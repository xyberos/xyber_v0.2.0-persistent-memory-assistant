# ---------------------------------------------------------------------------
# Xyber Chat — a minimal realtime chat application built on Xyberos.
#
# Stack:
#   * FastAPI  -> HTTP + WebSocket server
#   * Xyberos  -> the AI application framework (app.chat() pipeline)
#   * Ollama   -> local model server that serves qwen2.5:1.5b
#   * index.html -> the browser UI (served at GET /)
# ---------------------------------------------------------------------------

import asyncio            # asyncio.to_thread() runs blocking code off the event loop.
from pathlib import Path  # Cross-platform file paths (used to locate index.html).

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # Web server + WebSocket plumbing.
from fastapi.responses import HTMLResponse                  # Lets the "/" route return raw HTML.
from xyberos import create_app                              # Factory that builds a ready-to-use Xyberos app.
from xyberos.llm import OllamaEmbeddingLLM, OllamaLLM       # Adapters for local Ollama chat + embeddings.
from xyberos.llm.embeddings import embed_text               # Helper used to reload persisted turns on startup.
from xyberos.memory import MemoryEntry, VectorMemory        # Semantic + recency memory (remembers the conversation).
from xyberos.vector import SqliteVectorStore                # Persistent vector store that backs that memory.

# ---------------------------------------------------------------------------
# Xyberos has no built-in "persona"/system-prompt setting, so the assistant's
# identity is injected by simply prefixing every user message with this line.
# The "say so instead of guessing" clause is an anti-hallucination guard: the
# small local model should admit ignorance rather than invent facts.
# ---------------------------------------------------------------------------
PERSONA = (
    "You are Xyber, a witty assistant. Use the conversation history and your "
    "knowledge only; if you genuinely don't know something, say so instead of "
    "guessing. Never repeat yourself; always reply freshly and address the "
    "user's latest message directly."
)

# ---------------------------------------------------------------------------
# Persistent, semantic conversation memory.
#
# create_app() defaults to an in-memory memory that forgets everything on
# restart and grows without bound. We instead install a persistent semantic
# memory built from Xyberos' own components:
#   * SqliteVectorStore   -> writes to xyberos_memory.db, so the conversation
#                            survives restarts.
#   * OllamaEmbeddingLLM  -> real semantic embeddings from the local Ollama
#                            server (model: nomic-embed-text).
#   * VectorMemory        -> on every turn the Brain stores the exchange and
#                            later retrieves only the top_k most *relevant*
#                            past turns (similarity blended with recency).
#                            Keeping the injected context small and on-topic
#                            is what limits the model's hallucinations.
# NOTE: qwen2.5:1.5b is a very small model — see the README for its limits.
# NOTE: the embedding model must be available in Ollama first:
#       `ollama pull nomic-embed-text`
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> float:
    """Best-effort float timestamp for a stored ISO ``created_at`` value."""
    try:
        from datetime import datetime
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


class PersistentVectorMemory(VectorMemory):
    """VectorMemory that reloads past turns from the store at startup.

    Plain VectorMemory only ever recalls what was stored *in this process*:
    the vectors are written to SQLite but never read back, so a restart
    forgets the whole conversation. This subclass restores ``_entries`` from
    the persistent store on construction, so the conversation truly survives
    restarts while keeping the same semantic + recency recall.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restore()

    def _restore(self) -> None:
        """Seed the in-process history from the persistent vector store."""
        if self._embedder is None:  # no embedder -> recency-only, nothing to reload
            return
        try:
            probe = embed_text(self._embedder, " ")  # any vector to list the namespace
            hits = self._store.query(self._namespace, probe, top_k=100_000)
        except Exception:
            return  # embedding server unreachable at startup -> start empty
        restored: list[MemoryEntry] = []
        for hit in hits:
            payload = hit.payload or {}
            if payload.get("prompt") is None and payload.get("response") is None:
                continue
            restored.append(
                MemoryEntry(
                    prompt=payload.get("prompt"),
                    response=payload.get("response"),
                    created_at=payload.get("created_at") or "",
                )
            )
        restored.sort(key=lambda entry: _parse_timestamp(entry.created_at))
        self._entries = restored


PERSONA_HEADER = f"{PERSONA}\n\nUser: "  # the exact prefix _chat() prepends to each message


def _clean_prompt(prompt: str) -> str:
    """Strip the injected persona header, leaving just the user's message."""
    if prompt.startswith(PERSONA_HEADER):
        return prompt[len(PERSONA_HEADER):]
    return prompt


class PersonaMemory(PersistentVectorMemory):
    """Persistent semantic memory that stores only clean turns.

    The Brain persists ``context.prompt`` verbatim, which includes the persona
    header we prepend for generation. Storing that verbatim would repeat the
    whole persona inside every "user" line of the history and let the model
    echo stale assistant replies — a recipe for a repetitive small model.
    This wrapper strips the persona from the stored prompt and any echoed role
    prefix from the stored response, so the conversation history stays a clean
    transcript.
    """

    def store(self, context: object) -> None:
        original_prompt = getattr(context, "prompt", None)
        original_response = getattr(context, "response", None)
        if isinstance(original_prompt, str):
            context.prompt = _clean_prompt(original_prompt)
        if isinstance(original_response, str):
            context.response = _strip_prefix(original_response)
        try:
            super().store(context)
        finally:
            if isinstance(original_prompt, str):
                context.prompt = original_prompt
            if isinstance(original_response, str):
                context.response = original_response

    def retrieve(self, context: object) -> object:
        # Embed the clean message too, so the query vector matches the stored
        # (clean) vectors instead of a persona-prefixed one.
        original_prompt = getattr(context, "prompt", None)
        if isinstance(original_prompt, str):
            context.prompt = _clean_prompt(original_prompt)
        try:
            return super().retrieve(context)
        finally:
            if isinstance(original_prompt, str):
                context.prompt = original_prompt


memory = PersonaMemory(
    SqliteVectorStore("xyberos_memory.db"),
    embedder=OllamaEmbeddingLLM(model="nomic-embed-text"),
    top_k=5,
)
app = create_app(llm=OllamaLLM(model="qwen2.5:1.5b"), memory=memory)

# The FastAPI (ASGI) app that actually serves HTTP + WebSocket traffic.
# (uvicorn is pointed at "app:fast" — i.e. module "app", object "fast".)
fast = FastAPI()


def _strip_prefix(text: str) -> str:
    """Remove role prefixes the local model sometimes echoes back."""
    text = text.strip()                                  # Trim surrounding whitespace.
    for prefix in ("Xyber:", "Assistant:", "Baltz:", "User:"):  # Known echo prefixes.
        if text.startswith(prefix):                      # If the reply starts with one...
            text = text[len(prefix):].strip()            # ...chop it off and trim again.
    return text                                          # Return the cleaned-up reply.


def _chat(message: str) -> str:
    """Run the Xyberos pipeline with the Xyber persona applied."""
    # app.chat() is Xyberos' convenience API: it runs the whole cognitive
    # pipeline (memory, tools, planner, LLM, ...) and returns the text reply.
    # We prepend PERSONA so the model "knows" it is Xyber.
    reply = app.chat(f"{PERSONA}\n\nUser: {message}")
    return _strip_prefix(reply)  # Clean any "Assistant:"-style prefix out of the reply.


@fast.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the chat UI."""
    # Resolve index.html relative to THIS file (not the working directory), so
    # the app works no matter which folder uvicorn is launched from.
    html_file = Path(__file__).with_name("index.html")
    if html_file.exists():                                              # If the UI file exists...
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))  # ...serve it as HTML.
    return HTMLResponse(content="<h1>index.html missing</h1>", status_code=404)  # Otherwise 404.


@fast.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    """JSON chat protocol:

    Client -> Server:  {"message": "..."}
    Server -> Client:  {"type": "typing"} | {"type": "reply", "content": "..."}
                       | {"type": "error", "content": "..."}
    """
    await socket.accept()          # Accept the incoming WebSocket handshake.
    try:
        while True:                # Keep serving this connection until it closes.
            data = await socket.receive_json()                    # Wait for {"message": "..."} from the client.
            message = str(data.get("message", "")).strip()        # Read & trim the user's text.
            if not message:                                       # Ignore empty messages.
                continue

            # Tell the client to show the "typing..." indicator while we think.
            await socket.send_json({"type": "typing"})

            try:
                # app.chat() is synchronous and BLOCKING (it waits for the local
                # model). Running it on the event loop would freeze every other
                # request, so we push it onto a worker thread instead.
                reply = await asyncio.to_thread(_chat, message)
            except Exception as exc:  # e.g. Ollama is not running
                await socket.send_json({"type": "error", "content": str(exc)})
            else:
                # Send the final reply back; the UI appends it as a chat bubble.
                await socket.send_json({"type": "reply", "content": reply})
    except WebSocketDisconnect:
        pass  # Client went away — nothing to clean up, just stop the loop.


if __name__ == "__main__":
    import uvicorn  # Only import the dev server when run directly (python app.py).

    # Start the server. "app:fast" = import module "app", use object "fast".
    uvicorn.run("app:fast", host="127.0.0.1", port=8000, reload=True)
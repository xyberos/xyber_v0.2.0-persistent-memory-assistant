"""Xyberos agent — builds the app and exposes a chat() API.

The module-level ``agent`` is a ready-to-use :class:`~xyberos.Xyberos`
instance. It is consumed by the CLI (``app.py`` -> :func:`chat`) and by the
web layer via the :func:`chat` / :func:`achat` helpers below.

``chat``/``achat`` answer questions that need live data by routing them
through a query normalizer (:func:`backend.normalizer.classify`) into one of
three paths: ``'news'`` (broad news request -> :func:`~backend.tools.news_search`),
``'search'`` (specific topic -> :func:`~backend.tools.web_search`), or
``'chat'`` (normal agent pipeline). Web results are fed to the LLM for a
grounded, cited answer.
"""

import asyncio
import datetime

from xyberos import Xyberos, create_app
from xyberos.llm import OllamaLLM

from backend.normalizer import classify, normalize_query
from backend.tools import load_tools, news_search

# Local model served by Ollama (run `ollama pull qwen2.5:1.5b` once).
MODEL = "qwen2.5:1.5b"

_GROUNDING_PROMPT = (
    "Today is {date}. Answer the question using ONLY the search results below. "
    "Report concrete facts from the titles and snippets; do not just describe "
    "the websites. If the results do not contain the answer, say so. "
    "Cite each fact's source URL and end with the source list.\n\n"
    "Search results:\n{results}\n\nQuestion: {question}"
)

_NEWS_PROMPT = (
    "Today is {date}. The user asked: \"{question}\".\n"
    "Give a rundown of today's top news stories using ONLY the headlines below "
    "(the outlet is the part after the last \" - \"). Number each story. Do not "
    "invent details and do not describe the news websites themselves. End with "
    "the source URLs.\n\n"
    "Headlines:\n{items}"
)


def create_agent() -> Xyberos:
    """Create and configure the Xyberos AI agent."""
    agent = create_app(llm=OllamaLLM(MODEL))
    load_tools(agent)

    return agent


agent = create_agent()


def _format_results(results: list[dict]) -> str:
    lines = []
    for i, result in enumerate(results, 1):
        lines.append(
            f"{i}. {result.get('title', '')}\n"
            f"   {result.get('url', '')}\n"
            f"   {result.get('snippet', '')}"
        )
    return "\n".join(lines)


def _grounded_search(message: str) -> str | None:
    """Search the web for a specific topic and ground the LLM answer on it.

    Routes through the tool registry so the registered ``web_search`` is used
    — the commercial provider when an API key is set, otherwise the free
    Google News RSS fallback.
    """
    try:
        query = normalize_query(message) or message
        results = agent.tools.execute("web_search", None, query=query, top_k=5)
    except Exception:
        return None  # search unavailable -> fall back to normal chat
    if not results:
        return None
    prompt = _GROUNDING_PROMPT.format(
        date=datetime.date.today().strftime("%B %d, %Y"),
        results=_format_results(results),
        question=message,
    )
    return agent.llm.generate(prompt)


def _grounded_news(message: str) -> str | None:
    """Fetch real headlines (or topic news) and ask the LLM for a rundown."""
    try:
        items = news_search(query="", top_k=8)
    except Exception:
        return None
    if not items:
        return None
    prompt = _NEWS_PROMPT.format(
        date=datetime.date.today().strftime("%B %d, %Y"),
        items="\n".join(f"{i}. {it['title']}" for i, it in enumerate(items, 1)),
        question=message,
    )
    return agent.llm.generate(prompt)


def _answer_with_web(message: str) -> str | None:
    kind = classify(message)
    if kind == "news":
        return _grounded_news(message)
    if kind == "search":
        return _grounded_search(message)
    return None


def chat(message: str, *, user: str = "anon") -> str:
    """Run one user message through the agent and return the reply text."""
    if classify(message) in ("news", "search"):
        grounded = _answer_with_web(message)
        if grounded:
            return grounded
    return agent.chat(message, metadata={"user": user})


async def achat(message: str, *, user: str = "anon") -> str:
    """Async variant of :func:`chat` — handy for FastAPI/WebSocket handlers."""
    if classify(message) in ("news", "search"):
        grounded = await asyncio.to_thread(_answer_with_web, message)
        if grounded:
            return grounded
    return await agent.achat(message, metadata={"user": user})
"""Custom tools for the Xyberos agent.

This module is the home for tools beyond the built-in ones. Everything is
registered on the agent at startup via :func:`load_tools`, which is called by
:mod:`backend.agent` when the app is created.

Available tools (all query-driven):
* ``web_search(query, top_k=5)`` — commercial provider plugin (Tavily/Serper/
  Brave/Exa/Firecrawl) when a matching ``*_API_KEY`` env var is set, otherwise
  a free key-less fallback (Google News RSS search).
* ``news_search(query="", top_k=8)`` — news articles matching ``query``; empty
  ``query`` returns today's top stories (Google News RSS, free).
* ``fetch_page(url, max_chars=3000)`` — fetch a page's readable text (free).
"""

from __future__ import annotations

import html as _html
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from xyberos import Xyberos
from xyberos.tools import FunctionTool
from xyberos_web_search import WebSearchPlugin

__all__ = ["load_tools", "web_search", "news_search", "fetch_page"]

#: If any of these API keys is set, the commercial web_search plugin is used
#: instead of the free Google News RSS fallback.
_COMMERCIAL_KEYS = (
    "TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY",
    "EXA_API_KEY", "FIRECRAWL_API_KEY",
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def web_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search news via Google News RSS (free, no API key).

    Returns ``[{title, url, snippet, source, published}]``. When a commercial
    provider key is set, the plugin's ``web_search`` is registered instead.
    """
    return _search_google_news(query, top_k=top_k)


_GOOGLE_NEWS_RSS = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"


def _parse_news_items(root: ET.Element, top_k: int) -> list[dict[str, Any]]:
    """Convert a Google News RSS root into [{title, url, snippet, source, ...}]."""
    items: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:top_k]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source = title.rsplit(" - ", 1)[1].strip() if " - " in title else ""
        story = title.rsplit(" - ", 1)[0].strip() if " - " in title else title
        items.append(
            {
                "title": title,
                "url": (item.findtext("link") or "").strip(),
                "snippet": story,
                "source": source,
                "published": (item.findtext("pubDate") or "").strip(),
            }
        )
    return items


def _search_google_news(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search news via Google News RSS (free, no API key)."""
    url = (
        "https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    root = ET.fromstring(urllib.request.urlopen(request, timeout=15).read())
    return _parse_news_items(root, top_k)


def news_search(query: str = "", top_k: int = 8) -> list[dict[str, Any]]:
    """Search news via Google News RSS (free, no API key).

    A non-empty ``query`` returns matching articles; an empty ``query``
    returns today's top stories. Returns
    ``[{title, url, snippet, source, published}]``.
    """
    if query and query.strip():
        return _search_google_news(query, top_k=top_k)
    request = urllib.request.Request(_GOOGLE_NEWS_RSS, headers={"User-Agent": _UA})
    root = ET.fromstring(urllib.request.urlopen(request, timeout=15).read())
    return _parse_news_items(root, top_k)


def _extract_readable(html: str) -> str:
    """Strip scripts/styles/tags from an HTML page and return plain text."""
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S | re.I)
    if body:
        html = body.group(1)
    text = re.sub(r"<[^>]+>", " ", html)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_page(url: str, max_chars: int = 3000) -> dict[str, str]:
    """Fetch a web page and return its readable text (free, no API key).

    Returns ``{"url": ..., "content": ...}`` truncated to ``max_chars``.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    raw = urllib.request.urlopen(request, timeout=15).read().decode("utf-8", "ignore")
    return {"url": url, "content": _extract_readable(raw)[:max_chars]}


def _has_commercial_provider() -> bool:
    return any(os.getenv(key) for key in _COMMERCIAL_KEYS)


def load_tools(agent: Xyberos) -> None:
    """Register all tools on ``agent``.

    ``web_search`` uses the commercial provider plugin (Tavily/Serper/Brave/
    Exa/Firecrawl) when a matching ``*_API_KEY`` env var is set; otherwise the
    free Google News RSS fallback is registered. ``news_search`` and
    ``fetch_page`` are always available.
    """
    if _has_commercial_provider():
        agent.load_plugin(WebSearchPlugin())
    else:
        agent.tools.register(
            FunctionTool(
                "web_search",
                web_search,
                description=(
                    "Search the web and return up to top_k ranked results "
                    "(title, url, snippet)."
                ),
            )
        )
    agent.tools.register(
        FunctionTool(
            "news_search",
            news_search,
            description=(
                "Search news via Google News RSS. Empty query returns today's "
                "top stories; non-empty query returns matching articles. "
                "Returns up to top_k items (title, url, source, published)."
            ),
        )
    )
    agent.tools.register(
        FunctionTool(
            "fetch_page",
            fetch_page,
            description=(
                "Fetch a URL and return its readable article text (up to "
                "max_chars characters)."
            ),
        )
    )

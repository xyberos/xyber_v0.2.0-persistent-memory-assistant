"""Query/prompt normalization and lightweight intent classification.

Replaces the previous hard-coded ``SEARCH_HINTS`` / ``GENERIC_NEWS_WORDS``
lists in :mod:`backend.agent` with one normalizer plus a small set of
trigger / stop words.

Usage::

    from backend.normalizer import classify, normalize_query

    classify("current events today")   # -> "news"
    classify("openai stock price today")  # -> "search"
    classify("hello")                  # -> "chat"
    normalize_query("What's the News, Today?!")  # -> "what s the news today"
"""

from __future__ import annotations

import re

#: Words that signal a live web answer may be needed. Also excluded from the
#: extracted "topic" so they never turn a broad request into a topic query.
_TRIGGERS = frozenset({
    "news", "search", "web", "latest", "today", "current", "breaking",
    "headline", "headlines", "update", "updates", "weather", "forecast",
    "happened", "happening", "happen", "stories", "story", "events", "world",
    "top", "now", "right",
})

#: Pure filler words removed when extracting the topic of a message.
_STOPWORDS = frozenset({
    "a", "an", "and", "about", "around", "are", "for", "from", "give", "in",
    "is", "me", "of", "on", "please", "show", "tell", "the", "to", "what",
    "whats", "what's", "today's",
})

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9']+")


def normalize_query(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = _NON_ALNUM.sub(" ", (text or "").lower())
    return _WS.sub(" ", text).strip()


def topic_words(text: str) -> set[str]:
    """Meaningful tokens: the message minus trigger and stop words."""
    tokens = set(normalize_query(text).split())
    return tokens - _TRIGGERS - _STOPWORDS


def classify(text: str) -> str:
    """Classify a message as ``'news'``, ``'search'``, or ``'chat'``.

    * ``'chat'``   — no live web answer needed (no trigger words).
    * ``'news'``   — broad news request with no specific topic
                     (e.g. "current events today").
    * ``'search'`` — a specific topic query (e.g. "openai stock price today").
    """
    normalized = normalize_query(text)
    if not normalized:
        return "chat"
    if not (set(normalized.split()) & _TRIGGERS):
        return "chat"
    return "news" if not topic_words(text) else "search"

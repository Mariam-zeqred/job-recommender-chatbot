from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(Protocol):
    """One method. Return None when unavailable so callers fall back to rule-based text."""

    name: str

    def complete(self, messages: Sequence[Message], max_tokens: int = 200) -> str | None: ...


class NullProvider:
    """The default: no network, no key, always None."""

    name = "none"

    def complete(self, messages: Sequence[Message], max_tokens: int = 200) -> str | None:
        return None


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.environ.get("JOBREC_LLM_PROVIDER", "none")).lower()
    if name in ("", "none", "off", "null"):
        return NullProvider()
    if name == "http":
        from jobrec.providers.http_chat import HttpChatProvider

        return HttpChatProvider.from_env()
    raise ValueError(f"unknown LLM provider {name!r}; use none or http")

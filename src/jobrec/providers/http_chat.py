"""Any OpenAI-compatible chat-completions endpoint (self-hosted or commercial).

Enabled only when JOBREC_LLM_PROVIDER=http. Every failure returns None so the interview
keeps working rule-based; the demo must never depend on a key or on network access.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from jobrec.providers.base import Message


class HttpChatProvider:
    name = "http"

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, timeout: float = 20.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> HttpChatProvider:
        base = os.environ.get("JOBREC_LLM_BASE_URL")
        model = os.environ.get("JOBREC_LLM_MODEL")
        if not base or not model:
            raise ValueError("JOBREC_LLM_BASE_URL and JOBREC_LLM_MODEL must be set for http")
        return cls(base, model, os.environ.get("JOBREC_LLM_API_KEY") or None)

    def complete(self, messages: Sequence[Message], max_tokens: int = 200) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {"model": self.model, "messages": list(messages), "max_tokens": max_tokens}
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None
        return text.strip() or None

"""
Helpers for the local Ollama HTTP API: chat and health check utilities.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def ollama_host() -> str:
    """Return the base URL without a trailing slash."""
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_chat_model() -> str:
    """Return the default chat model name."""
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")


def _post_json(path: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    """Send a JSON POST request to the Ollama HTTP API."""
    url = f"{ollama_host()}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: int = 120,
) -> str:
    """
    Return a single text response from Ollama using the /api/chat endpoint.
    """
    selected_model = model or ollama_chat_model()
    data = _post_json(
        "/api/chat",
        {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout_s=timeout_s,
    )
    message = data.get("message") or {}
    content = message.get("content")
    return (content or "").strip()


def ollama_reachable(timeout_s: float = 3.0) -> bool:
    """Check whether the Ollama server is reachable."""
    url = f"{ollama_host()}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.getcode() < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
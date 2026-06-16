"""Helpers for building OpenAI-shaped responses and SSE stream chunks."""

from __future__ import annotations

import json
import time
import uuid


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def completion_response(
    model: str,
    content: str,
    *,
    cached: bool = False,
    finish_reason: str = "stop",
    usage: dict | None = None,
    completion_id: str | None = None,
) -> dict:
    """A non-streaming `chat.completion` body."""
    return {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_cache": "hit" if cached else "miss",
    }


def chunk(completion_id: str, model: str, delta: dict, finish_reason: str | None = None) -> dict:
    """A streaming `chat.completion.chunk` body."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


SSE_DONE = "data: [DONE]\n\n"


def extract_text(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-shaped completion."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""

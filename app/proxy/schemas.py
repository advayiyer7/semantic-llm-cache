"""OpenAI-compatible request schema.

Kept permissive (`extra="allow"`) so the proxy is a faithful passthrough — any
field we don't model is preserved and forwarded to the provider untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    n: int | None = None

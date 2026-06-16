"""OpenAI-compatible request schema.

Kept permissive (`extra="allow"`) so the proxy is a faithful passthrough — any
field we don't model is preserved and forwarded to the provider untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Proxy-only fields — the single source of truth for what gets stripped before
# a request is forwarded to a provider.
PROXY_ONLY_FIELDS: frozenset[str] = frozenset({"cache_profile", "cache_ttl"})


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Bounded + charset-constrained so it's safe to hash, log, and forward.
    model: str = Field(..., min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._:\-/]+$")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    n: int | None = None

    # Proxy extensions (stripped before forwarding to the provider).
    cache_profile: str | None = None  # strict | balanced | relaxed | off
    cache_ttl: int | None = Field(default=None, ge=1, le=2_592_000)  # ≤ 30 days

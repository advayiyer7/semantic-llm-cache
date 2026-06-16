"""Provider protocol.

A provider takes an OpenAI-shaped request and either returns a full
OpenAI-shaped response (`complete`) or yields assistant text deltas (`stream`).
Keeping the streaming contract to plain text deltas lets the proxy own the SSE
framing and the buffer-to-store-on-miss logic in one place.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from app.proxy.schemas import ChatCompletionRequest


class Provider(Protocol):
    async def complete(self, req: ChatCompletionRequest) -> dict: ...

    def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]: ...

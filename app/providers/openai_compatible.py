"""Provider for any OpenAI-compatible `/chat/completions` endpoint.

Both hosted OpenAI and a local Ollama server speak this contract, so one class
serves both — they differ only by base URL and auth header. A single pooled
`AsyncClient` is reused across requests and closed at shutdown via `aclose()`.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from app.proxy.schemas import ChatCompletionRequest

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # follow_redirects=False: don't let an upstream redirect bounce an
        # authenticated request to an unintended host.
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, req: ChatCompletionRequest, stream: bool) -> dict:
        # Drop proxy-only extension fields so they never reach the provider.
        body = req.model_dump(
            exclude_none=True, exclude={"stream", "cache_profile", "cache_ttl"}
        )
        body["stream"] = stream
        return body

    async def complete(self, req: ChatCompletionRequest) -> dict:
        resp = await self._client.post(
            f"{self._base}/chat/completions",
            json=self._payload(req, stream=False),
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            f"{self._base}/chat/completions",
            json=self._payload(req, stream=True),
            headers=self._headers,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = parsed.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta

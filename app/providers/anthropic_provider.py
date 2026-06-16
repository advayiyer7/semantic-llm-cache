"""Anthropic provider — translates the OpenAI chat contract to/from Anthropic.

Differences handled here:
  - Anthropic takes `system` as a top-level param, not a message role.
  - `max_tokens` is required (default applied if the caller omits it).
  - Sampling params (`temperature`/`top_p`/`top_k`) 400 on Opus 4.7/4.8/Fable,
    so they're dropped for those models.
  - Response content is a list of blocks; assistant text is the `text` blocks.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.proxy.openai_format import completion_response, text_of
from app.proxy.schemas import ChatCompletionRequest

_DEFAULT_MAX_TOKENS = 1024
_NO_SAMPLING = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable")
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _rejects_sampling(model: str) -> bool:
    return model.lower().startswith(_NO_SAMPLING)


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    def _build_kwargs(self, req: ChatCompletionRequest) -> dict:
        system_parts: list[str] = []
        messages: list[dict] = []
        for message in req.messages:
            content = text_of(message.content)
            if message.role == "system":
                system_parts.append(content)
            else:
                messages.append({"role": message.role, "content": content})

        kwargs: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if not _rejects_sampling(req.model):
            if req.temperature is not None:
                kwargs["temperature"] = req.temperature
            if req.top_p is not None:
                kwargs["top_p"] = req.top_p
        return kwargs

    async def complete(self, req: ChatCompletionRequest) -> dict:
        message = await self._client.messages.create(**self._build_kwargs(req))
        text = "".join(b.text for b in message.content if b.type == "text")
        usage = {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }
        return completion_response(
            req.model,
            text,
            finish_reason=_STOP_REASON_MAP.get(message.stop_reason, "stop"),
            usage=usage,
        )

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._build_kwargs(req)) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text

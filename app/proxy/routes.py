"""The drop-in `/v1/chat/completions` endpoint.

Cache flow per request:
  1. Split system prompt (→ namespace) from the conversation (→ embedded).
  2. Query the cache. On a hit, return the stored response (JSON or replayed SSE)
     without calling the provider.
  3. On a miss, call the provider. For streaming, frame deltas as SSE while
     buffering the full text, then store it once the stream ends.

Cache reads/writes embed via a (synchronous) network call, so they run in a
worker thread to avoid blocking the event loop. Provider errors are sanitized to
a 502 so upstream error bodies (which can echo key fragments) never reach the
caller; cache failures degrade to a plain miss rather than failing the request.
"""

from __future__ import annotations

import asyncio
import logging

import anthropic
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.providers.router import provider_key_for_model
from app.proxy.openai_format import (
    SSE_DONE,
    chunk,
    completion_response,
    extract_text,
    new_completion_id,
    sse,
    text_of,
)
from app.proxy.schemas import ChatCompletionRequest

logger = logging.getLogger("semantic_cache")
router = APIRouter()

_PROVIDER_ERRORS = (httpx.HTTPStatusError, httpx.RequestError, anthropic.APIError)


def _split_messages(messages) -> tuple[str | None, str]:
    """Return (system_prompt, serialized_conversation)."""
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for message in messages:
        content = text_of(message.content)
        if message.role == "system":
            system_parts.append(content)
        else:
            convo_parts.append(f"{message.role}: {content}")
    system = "\n".join(system_parts) or None
    return system, "\n".join(convo_parts)


async def _safe_query(cache, model, system, params, conversation):
    try:
        return await asyncio.to_thread(cache.query, model, system, params, conversation)
    except Exception as exc:  # noqa: BLE001 - cache must never fail the request
        logger.warning("cache.query failed (%s) — treating as miss", type(exc).__name__)
        return None


async def _safe_store(cache, namespace, vector, conversation, text):
    try:
        await asyncio.to_thread(cache.store, namespace, vector, conversation, text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.store failed (%s)", type(exc).__name__)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    try:
        req = ChatCompletionRequest(**body)
    except ValidationError as exc:
        # Strip Pydantic's doc URLs / echoed input; keep only location/message/type.
        detail = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        raise HTTPException(status_code=400, detail=detail)

    providers = request.app.state.providers
    key = provider_key_for_model(req.model)
    provider = providers.get(key)
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{key}' is not configured for model '{req.model}'.",
        )

    cache = getattr(request.app.state, "cache", None)
    system, conversation = _split_messages(req.messages)
    # n>1 yields multiple choices we don't model in the cache; skip caching it.
    cacheable = cache is not None and req.n in (None, 1)

    cache_result = None
    if cacheable:
        params = {"temperature": req.temperature, "top_p": req.top_p}
        cache_result = await _safe_query(cache, req.model, system, params, conversation)
        if cache_result is not None and cache_result.hit is not None:
            content = cache_result.hit.response
            if req.stream:
                return StreamingResponse(
                    _replay_stream(req.model, content),
                    media_type="text/event-stream",
                    headers={"X-Cache": "HIT"},
                )
            return JSONResponse(
                completion_response(req.model, content), headers={"X-Cache": "HIT"}
            )

    if req.stream:
        return StreamingResponse(
            _stream_and_store(provider, req, cache, cache_result, conversation),
            media_type="text/event-stream",
            headers={"X-Cache": "MISS"},
        )

    try:
        response = await provider.complete(req)
    except _PROVIDER_ERRORS as exc:
        logger.error("provider '%s' failed: %s", key, exc)
        raise HTTPException(status_code=502, detail="Upstream provider error.")

    text = extract_text(response)
    if cacheable and cache_result is not None and text:
        await _safe_store(cache, cache_result.namespace, cache_result.vector, conversation, text)
    return JSONResponse(response, headers={"X-Cache": "MISS"})


async def _replay_stream(model: str, content: str):
    """Emit a cached response as a single-delta SSE stream."""
    cid = new_completion_id()
    yield sse(chunk(cid, model, {"role": "assistant"}))
    yield sse(chunk(cid, model, {"content": content}))
    yield sse(chunk(cid, model, {}, finish_reason="stop"))
    yield SSE_DONE


async def _stream_and_store(provider, req, cache, cache_result, conversation):
    cid = new_completion_id()
    buffer: list[str] = []
    yield sse(chunk(cid, req.model, {"role": "assistant"}))
    try:
        async for delta in provider.stream(req):
            buffer.append(delta)
            yield sse(chunk(cid, req.model, {"content": delta}))
    except _PROVIDER_ERRORS as exc:
        logger.error("provider stream failed: %s", exc)
        yield sse({"error": {"message": "Upstream provider error.", "type": "upstream_error"}})
        yield SSE_DONE
        return

    yield sse(chunk(cid, req.model, {}, finish_reason="stop"))
    yield SSE_DONE

    text = "".join(buffer)
    if cache is not None and cache_result is not None and text:
        await _safe_store(cache, cache_result.namespace, cache_result.vector, conversation, text)

"""The drop-in `/v1/chat/completions` endpoint.

Per request:
  1. A cache policy decides cacheable / threshold / TTL from temperature (or an
     explicit `cache_profile`). Non-cacheable requests bypass the cache entirely.
  2. Query the cache. On a hit, return the stored response without the provider.
  3. On a miss, a single-flight lock collapses concurrent identical requests:
     the first calls the provider and stores; the rest re-check and reuse it.
  4. Streaming misses frame deltas as SSE while buffering the full text, then
     store it once the stream ends.

Cache embed/search run in worker threads. Provider errors are sanitized to 502
so upstream bodies (which can echo key fragments) never reach the caller; cache
failures degrade to a miss/no-op.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import anthropic
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.cache.policy import VALID_PROFILES, decide
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
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for message in messages:
        content = text_of(message.content)
        if message.role == "system":
            system_parts.append(content)
        else:
            convo_parts.append(f"{message.role}: {content}")
    return ("\n".join(system_parts) or None), "\n".join(convo_parts)


def _flight_key(namespace: str, conversation: str) -> str:
    digest = hashlib.sha256(conversation.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}:{digest}"


async def _safe_query(cache, model, system, params, conversation, threshold):
    try:
        return await asyncio.to_thread(
            cache.query, model, system, params, conversation, threshold
        )
    except Exception as exc:  # noqa: BLE001 - cache must never fail the request
        logger.warning("cache.query failed (%s) — treating as miss", type(exc).__name__)
        return None


async def _safe_lookup(cache, namespace, vector, threshold):
    try:
        return await asyncio.to_thread(cache.lookup, namespace, vector, threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.lookup failed (%s)", type(exc).__name__)
        return None


async def _safe_store(cache, namespace, vector, conversation, text, ttl):
    try:
        await asyncio.to_thread(cache.store, namespace, vector, conversation, text, ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.store failed (%s)", type(exc).__name__)


async def _provider_complete(provider, req, key):
    try:
        return await provider.complete(req)
    except _PROVIDER_ERRORS as exc:
        logger.error("provider '%s' failed: %s", key, exc)
        raise HTTPException(status_code=502, detail="Upstream provider error.")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    try:
        req = ChatCompletionRequest(**body)
    except ValidationError as exc:
        detail = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        raise HTTPException(status_code=400, detail=detail)

    if req.cache_profile is not None and req.cache_profile not in VALID_PROFILES:
        raise HTTPException(
            status_code=400, detail=f"Invalid cache_profile '{req.cache_profile}'."
        )

    app = request.app
    key = provider_key_for_model(req.model)
    provider = app.state.providers.get(key)
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{key}' is not configured for model '{req.model}'.",
        )

    cache = getattr(app.state, "cache", None)
    decision = decide(req.temperature, req.cache_profile, req.cache_ttl, app.state.settings)
    system, conversation = _split_messages(req.messages)
    cacheable = cache is not None and req.n in (None, 1) and decision.cacheable
    base_headers = {"X-Cache-Profile": decision.profile}

    cache_result = None
    if cacheable:
        params = {"temperature": req.temperature, "top_p": req.top_p}
        cache_result = await _safe_query(
            cache, req.model, system, params, conversation, decision.threshold
        )
        if cache_result is not None and cache_result.hit is not None:
            content = cache_result.hit.response
            if req.stream:
                return StreamingResponse(
                    _replay_stream(req.model, content),
                    media_type="text/event-stream",
                    headers={**base_headers, "X-Cache": "HIT"},
                )
            return JSONResponse(
                completion_response(req.model, content),
                headers={**base_headers, "X-Cache": "HIT"},
            )

    if req.stream:
        ttl = decision.ttl if cacheable else None
        store_ctx = cache_result if cacheable else None
        return StreamingResponse(
            _stream_and_store(provider, req, cache, store_ctx, conversation, ttl, key),
            media_type="text/event-stream",
            headers={**base_headers, "X-Cache": "MISS" if cacheable else "BYPASS"},
        )

    if not cacheable:
        response = await _provider_complete(provider, req, key)
        return JSONResponse(response, headers={**base_headers, "X-Cache": "BYPASS"})

    if cache_result is None:
        # Cache query failed (e.g. Redis blip) — serve from provider, don't store.
        response = await _provider_complete(provider, req, key)
        return JSONResponse(response, headers={**base_headers, "X-Cache": "MISS"})

    # Cacheable miss — collapse concurrent identical requests.
    flight = _flight_key(cache_result.namespace, conversation)
    async with app.state.singleflight(flight):
        recheck = await _safe_lookup(
            cache, cache_result.namespace, cache_result.vector, decision.threshold
        )
        if recheck is not None:
            return JSONResponse(
                completion_response(req.model, recheck.response),
                headers={**base_headers, "X-Cache": "HIT"},
            )
        response = await _provider_complete(provider, req, key)
        text = extract_text(response)
        if text:
            await _safe_store(
                cache, cache_result.namespace, cache_result.vector, conversation, text, decision.ttl
            )
        return JSONResponse(response, headers={**base_headers, "X-Cache": "MISS"})


async def _replay_stream(model: str, content: str):
    cid = new_completion_id()
    yield sse(chunk(cid, model, {"role": "assistant"}))
    yield sse(chunk(cid, model, {"content": content}))
    yield sse(chunk(cid, model, {}, finish_reason="stop"))
    yield SSE_DONE


async def _stream_and_store(provider, req, cache, cache_result, conversation, ttl, key):
    cid = new_completion_id()
    buffer: list[str] = []
    yield sse(chunk(cid, req.model, {"role": "assistant"}))
    try:
        async for delta in provider.stream(req):
            buffer.append(delta)
            yield sse(chunk(cid, req.model, {"content": delta}))
    except _PROVIDER_ERRORS as exc:
        logger.error("provider '%s' stream failed: %s", key, exc)
        yield sse({"error": {"message": "Upstream provider error.", "type": "upstream_error"}})
        yield SSE_DONE
        return

    yield sse(chunk(cid, req.model, {}, finish_reason="stop"))
    yield SSE_DONE

    text = "".join(buffer)
    if cache is not None and cache_result is not None and text:
        await _safe_store(
            cache, cache_result.namespace, cache_result.vector, conversation, text, ttl
        )

"""FastAPI entrypoint.

Wires the providers and the semantic cache at startup, then exposes the
drop-in proxy plus liveness/readiness. The cache is enabled only when an
OpenAI key is configured (it's needed for embeddings); without it the proxy
still works as a transparent passthrough.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import StreamingResponse

from app.cache.embedder import build_embedder
from app.metrics.prometheus import record_request
from app.cache.engine import SemanticCache
from app.cache.store import CacheStore
from app.config import get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.proxy.admin import admin_router
from app.proxy.routes import router as proxy_router
from app.proxy.singleflight import SingleFlight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_cache")

# Reject oversized bodies before buffering them into memory.
MAX_BODY_BYTES = 2_000_000


def _build_providers(settings) -> dict:
    providers: dict = {
        "ollama": OpenAICompatibleProvider(f"{settings.ollama_base_url.rstrip('/')}/v1")
    }
    if settings.openai_api_key:
        providers["openai"] = OpenAICompatibleProvider(
            "https://api.openai.com/v1", settings.openai_api_key
        )
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
    return providers


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    if not settings.admin_api_key:
        logger.warning("ADMIN_API_KEY not set — /admin/* endpoints are open to all callers.")
    app.state.providers = _build_providers(settings)
    app.state.singleflight = SingleFlight()
    app.state.store = None
    app.state.cache = None

    try:
        embedder = build_embedder(settings)
        if embedder is None:
            logger.warning(
                "No embedder (set OPENAI_API_KEY, or EMBEDDING_BACKEND=local) — "
                "cache disabled (passthrough only)"
            )
        else:
            store = CacheStore(
                redis_url=settings.redis_url,
                index_name=settings.index_name,
                index_prefix=settings.index_prefix,
                dims=embedder.dim,
            )
            store.ensure_index()
            app.state.store = store
            app.state.cache = SemanticCache(
                embedder=embedder,
                store=store,
                threshold=settings.similarity_threshold,
                default_ttl=settings.default_ttl_seconds,
            )
            logger.info(
                "Semantic cache enabled (backend=%s, dim=%d, threshold=%.2f)",
                settings.embedding_backend,
                embedder.dim,
                settings.similarity_threshold,
            )
    except Exception as exc:  # noqa: BLE001 - startup must stay resilient
        # Log the type only — the exception text can contain the Redis URL/creds.
        logger.warning("Cache unavailable at startup: %s", type(exc).__name__)

    yield

    for provider in app.state.providers.values():
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="Semantic LLM Cache", version="0.3.0", lifespan=lifespan)
app.include_router(proxy_router)
app.include_router(admin_router)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large."}, status_code=413)
    return await call_next(request)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    if request.url.path != "/v1/chat/completions":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    label = response.headers.get("X-Cache")
    if label:
        # Latency is only meaningful for non-streaming responses (streaming
        # returns the generator immediately, before the body is produced).
        latency = None if isinstance(response, StreamingResponse) else time.perf_counter() - start
        record_request(label.lower(), latency)
    return response


@app.get("/metrics")
def metrics(request: Request) -> Response:
    token = request.app.state.settings.metrics_auth_token
    if token and request.headers.get("Authorization") != f"Bearer {token}":
        return Response(status_code=401)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness — process is up."""
    return {"status": "ok"}


@app.get("/health/ready")
def ready(response: Response) -> dict[str, str]:
    """Readiness — vector store is reachable."""
    store = getattr(app.state, "store", None)
    if store is None or not store.ping():
        response.status_code = 503
        return {"status": "not_ready", "reason": "cache store unavailable"}
    return {"status": "ready"}

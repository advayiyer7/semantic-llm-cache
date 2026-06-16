"""FastAPI entrypoint.

Wires the providers and the semantic cache at startup, then exposes the
drop-in proxy plus liveness/readiness. The cache is enabled only when an
OpenAI key is configured (it's needed for embeddings); without it the proxy
still works as a transparent passthrough.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.cache.embedder import OpenAIEmbedder
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
    app.state.providers = _build_providers(settings)
    app.state.singleflight = SingleFlight()
    app.state.store = None
    app.state.cache = None

    try:
        store = CacheStore(
            redis_url=settings.redis_url,
            index_name=settings.index_name,
            index_prefix=settings.index_prefix,
            dims=settings.embedding_dim,
        )
        store.ensure_index()
        app.state.store = store
        if settings.openai_api_key:
            cache = SemanticCache(
                embedder=OpenAIEmbedder(settings),
                store=store,
                threshold=settings.similarity_threshold,
                default_ttl=settings.default_ttl_seconds,
            )
            app.state.cache = cache
            logger.info("Semantic cache enabled (threshold=%.2f)", settings.similarity_threshold)
        else:
            logger.warning("OPENAI_API_KEY not set — cache disabled (passthrough only)")
    except Exception as exc:  # noqa: BLE001 - startup must stay resilient
        # Log the type only — the exception text can contain the Redis URL/creds.
        logger.warning("Cache store unavailable at startup: %s", type(exc).__name__)

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

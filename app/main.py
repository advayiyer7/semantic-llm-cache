"""FastAPI entrypoint.

Wires the providers and the semantic cache at startup, then exposes the
drop-in proxy plus liveness/readiness. The cache is enabled only when an
OpenAI key is configured (it's needed for embeddings); without it the proxy
still works as a transparent passthrough.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from app.cache.embedder import OpenAIEmbedder
from app.cache.engine import SemanticCache
from app.cache.store import CacheStore
from app.config import get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.proxy.routes import router as proxy_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_cache")


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
        logger.warning("Cache store unavailable at startup: %s", exc)

    yield


app = FastAPI(title="Semantic LLM Cache", version="0.2.0", lifespan=lifespan)
app.include_router(proxy_router)


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

"""FastAPI entrypoint.

Phase 0 exposes liveness/readiness. The cache store is wired up at startup so
that readiness reflects whether the vector index is reachable. The proxy routes
(/v1/chat/completions) arrive in Phase 2.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from app.cache.store import CacheStore
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_cache")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.store = None

    # Connecting to Redis must not crash liveness — readiness reports the truth.
    try:
        store = CacheStore(
            redis_url=settings.redis_url,
            index_name=settings.index_name,
            index_prefix=settings.index_prefix,
            dims=settings.embedding_dim,
        )
        store.ensure_index()
        app.state.store = store
        logger.info("Cache index '%s' ready", settings.index_name)
    except Exception as exc:  # noqa: BLE001 - startup must stay resilient
        logger.warning("Cache store unavailable at startup: %s", exc)

    yield


app = FastAPI(title="Semantic LLM Cache", version="0.1.0", lifespan=lifespan)


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

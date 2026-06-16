"""Shared fixtures.

Integration tests need a running Redis Stack (vector search). When it isn't
reachable they skip cleanly rather than fail, so `pytest` stays green on a bare
checkout. Start Redis with: `docker-compose up -d redis`.
"""

from __future__ import annotations

import uuid

import pytest

from app.cache.store import CacheStore
from tests.fakes import FakeEmbedder

REDIS_URL = "redis://localhost:6379"
TEST_DIM = 8


def _redis_available() -> bool:
    try:
        import redis

        client = redis.from_url(REDIS_URL)
        client.ping()
        # RediSearch module must be loaded (redis-stack, not plain redis).
        modules = {m[b"name"] for m in client.module_list()}
        return b"search" in modules
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=TEST_DIM)


@pytest.fixture
def store():
    if not _redis_available():
        pytest.skip(
            "Redis Stack not reachable on localhost:6379 "
            "(run: docker-compose up -d redis)"
        )
    name = f"test_{uuid.uuid4().hex[:8]}"
    cache_store = CacheStore(REDIS_URL, name, name, TEST_DIM)
    cache_store.ensure_index()
    try:
        yield cache_store
    finally:
        try:
            cache_store.drop()
        except Exception:  # noqa: BLE001
            pass

"""Single-flight — concurrent identical misses collapse to one provider call."""

from __future__ import annotations

import asyncio

import httpx

from app.cache.engine import SemanticCache
from app.config import get_settings
from app.main import app
from app.proxy.singleflight import SingleFlight
from tests.fakes import FakeEmbedder, SlowFakeProvider


async def test_singleflight_collapses_concurrent_misses(store):
    provider = SlowFakeProvider(delay=0.05)
    app.state.settings = get_settings()
    app.state.singleflight = SingleFlight()
    app.state.cache = SemanticCache(FakeEmbedder(8), store, threshold=0.95, default_ttl=60)
    app.state.providers = {"ollama": provider}

    body = {"model": "llama3.2", "messages": [{"role": "user", "content": "same prompt"}]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first, second = await asyncio.gather(
            client.post("/v1/chat/completions", json=body),
            client.post("/v1/chat/completions", json=body),
        )

    assert provider.calls == 1  # second request reused the first's stored result
    assert {first.headers["X-Cache"], second.headers["X-Cache"]} == {"MISS", "HIT"}

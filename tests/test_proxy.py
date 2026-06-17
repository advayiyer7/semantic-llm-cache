"""Proxy + cache integration tests over the HTTP layer.

Uses a FakeProvider (counts calls, no network) and a FakeEmbedder so the
miss → store → hit path is verified deterministically against real Redis.
Requires Redis Stack (via the `store` fixture, which skips if unavailable).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.cache.engine import SemanticCache
from app.main import app
from tests.fakes import FailingStreamProvider, FakeProvider


def _wire(store, embedder, provider, threshold=0.95):
    cache = SemanticCache(embedder, store, threshold=threshold, default_ttl=60)
    app.state.cache = cache
    app.state.providers = {"ollama": provider, "openai": provider, "anthropic": provider}


def test_miss_then_hit_does_not_call_provider_twice(store, embedder):
    provider = FakeProvider()
    body = {"model": "llama3.2", "messages": [{"role": "user", "content": "hello world"}]}

    with TestClient(app) as client:
        _wire(store, embedder, provider)

        first = client.post("/v1/chat/completions", json=body)
        assert first.status_code == 200
        assert first.headers["X-Cache"] == "MISS"
        assert provider.calls == 1
        assert first.json()["choices"][0]["message"]["content"] == "FAKE:hello world"

        second = client.post("/v1/chat/completions", json=body)
        assert second.status_code == 200
        assert second.headers["X-Cache"] == "HIT"
        assert provider.calls == 1  # served from cache — provider not called again
        assert second.json()["choices"][0]["message"]["content"] == "FAKE:hello world"


def test_different_prompt_is_a_miss(store, embedder):
    provider = FakeProvider()
    with TestClient(app) as client:
        _wire(store, embedder, provider)
        client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "first"}]},
        )
        client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "second"}]},
        )
        assert provider.calls == 2


def test_streaming_miss_then_hit(store, embedder):
    provider = FakeProvider()
    body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": True,
    }
    with TestClient(app) as client:
        _wire(store, embedder, provider)

        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.headers["X-Cache"] == "MISS"
            payload = "".join(resp.iter_text())
        assert "[DONE]" in payload
        assert provider.calls == 1

        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.headers["X-Cache"] == "HIT"
            hit_payload = "".join(resp.iter_text())
        assert "stream me" in hit_payload
        assert provider.calls == 1  # cache hit — provider not called


def test_high_temperature_bypasses_cache(store, embedder):
    provider = FakeProvider()
    body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "be creative"}],
        "temperature": 0.9,
    }
    with TestClient(app) as client:
        _wire(store, embedder, provider)
        first = client.post("/v1/chat/completions", json=body)
        assert first.headers["X-Cache"] == "BYPASS"
        assert first.headers["X-Cache-Profile"] == "off"
        client.post("/v1/chat/completions", json=body)
        assert provider.calls == 2  # never cached


def test_cache_profile_off_bypasses(store, embedder):
    provider = FakeProvider()
    body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "x"}],
        "cache_profile": "off",
    }
    with TestClient(app) as client:
        _wire(store, embedder, provider)
        client.post("/v1/chat/completions", json=body)
        client.post("/v1/chat/completions", json=body)
        assert provider.calls == 2


def test_invalid_cache_profile_returns_400(store, embedder):
    with TestClient(app) as client:
        _wire(store, embedder, FakeProvider())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "x"}],
                "cache_profile": "bogus",
            },
        )
        assert resp.status_code == 400


def test_streaming_provider_error_terminates_cleanly(store, embedder):
    provider = FailingStreamProvider()
    body = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    with TestClient(app) as client:
        _wire(store, embedder, provider)
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            payload = "".join(resp.iter_text())
    # Clean SDK-safe termination: finish_reason error + [DONE], no top-level error key.
    assert "[DONE]" in payload
    assert "error" in payload
    assert '"error":' not in payload


def test_unconfigured_provider_returns_503(store, embedder):
    provider = FakeProvider()
    with TestClient(app) as client:
        _wire(store, embedder, provider)
        app.state.providers = {"ollama": provider}  # no anthropic configured
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 503

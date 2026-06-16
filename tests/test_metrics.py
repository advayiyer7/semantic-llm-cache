"""Metrics + near-miss analyzer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.metrics.prometheus import (
    estimate_saved_usd,
    record_near_miss,
    recent_near_misses,
)


def test_estimate_saved_usd():
    # (40 + 40) / 4 = 20 tokens; 20/1000 * 0.002 = 0.00004
    usd = estimate_saved_usd("p" * 40, "r" * 40, 0.002)
    assert abs(usd - (20 / 1000 * 0.002)) < 1e-12


def test_estimate_saved_usd_empty_is_zero():
    assert estimate_saved_usd("", "", 0.002) == 0.0


def test_near_miss_buffer_records():
    record_near_miss("ns-abc", 0.93, 0.95)
    latest = recent_near_misses()[-1]
    assert latest["namespace"] == "ns-abc"
    assert latest["similarity"] == 0.93
    assert latest["threshold"] == 0.95


def test_metrics_endpoint_exposes_series():
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "semantic_cache_requests_total" in resp.text
        assert "semantic_cache_similarity" in resp.text


def test_near_misses_endpoint_requires_admin_when_key_set():
    with TestClient(app) as client:
        app.state.settings.admin_api_key = "secret"
        try:
            unauthorized = client.get("/admin/near-misses")
            assert unauthorized.status_code == 401
            authorized = client.get("/admin/near-misses", headers={"X-Admin-Key": "secret"})
            assert authorized.status_code == 200
            assert "near_misses" in authorized.json()
        finally:
            app.state.settings.admin_api_key = None

"""Cache policy — pure unit tests."""

from __future__ import annotations

from app.cache.policy import decide, infer_profile
from app.config import Settings

S = Settings()


def test_infer_profile_from_temperature():
    assert infer_profile(None, S) == "balanced"
    assert infer_profile(0.0, S) == "relaxed"
    assert infer_profile(0.2, S) == "relaxed"
    assert infer_profile(0.5, S) == "balanced"
    assert infer_profile(0.9, S) == "off"


def test_decide_relaxed_uses_low_threshold_and_long_ttl():
    d = decide(0.0, None, None, S)
    assert d.cacheable is True
    assert d.threshold == 0.90
    assert d.ttl == S.long_ttl_seconds
    assert d.profile == "relaxed"


def test_decide_creative_is_not_cacheable():
    d = decide(0.95, None, None, S)
    assert d.cacheable is False
    assert d.profile == "off"


def test_explicit_profile_overrides_temperature():
    d = decide(0.95, "strict", None, S)  # high temp would infer "off"
    assert d.cacheable is True
    assert d.threshold == 0.98
    assert d.ttl == S.short_ttl_seconds


def test_ttl_override_wins():
    d = decide(None, "balanced", 42, S)
    assert d.ttl == 42

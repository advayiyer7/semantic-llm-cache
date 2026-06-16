"""Per-request cache policy.

Decides, for each request, whether to cache, at what similarity threshold, and
with what TTL. The profile is taken from an explicit `cache_profile` override
when present, otherwise inferred from temperature:

  - low temperature (deterministic) -> `relaxed`: looser match (0.90), long TTL
  - high temperature (creative)      -> `off`: don't cache (outputs vary)
  - otherwise                        -> `balanced`: 0.95, default TTL
"""

from __future__ import annotations

from dataclasses import dataclass

PROFILE_THRESHOLDS = {"strict": 0.98, "balanced": 0.95, "relaxed": 0.90}
PROFILE_TTL_TIER = {"strict": "short", "balanced": "default", "relaxed": "long"}
VALID_PROFILES = set(PROFILE_THRESHOLDS) | {"off"}


@dataclass(frozen=True)
class CacheDecision:
    cacheable: bool
    threshold: float
    ttl: int
    profile: str


def _ttl_for_tier(tier: str, settings) -> int:
    return {
        "short": settings.short_ttl_seconds,
        "default": settings.default_ttl_seconds,
        "long": settings.long_ttl_seconds,
    }[tier]


def infer_profile(temperature: float | None, settings) -> str:
    if temperature is None:
        return "balanced"
    if temperature <= settings.deterministic_temperature_max:
        return "relaxed"
    if temperature >= settings.creative_temperature_min:
        return "off"
    return "balanced"


def decide(
    temperature: float | None,
    profile_override: str | None,
    ttl_override: int | None,
    settings,
) -> CacheDecision:
    profile = profile_override or infer_profile(temperature, settings)
    if profile == "off":
        return CacheDecision(cacheable=False, threshold=0.0, ttl=0, profile="off")
    threshold = PROFILE_THRESHOLDS[profile]
    ttl = ttl_override or _ttl_for_tier(PROFILE_TTL_TIER[profile], settings)
    return CacheDecision(cacheable=True, threshold=threshold, ttl=ttl, profile=profile)

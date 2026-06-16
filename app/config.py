"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Providers
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Optional: when set, /admin/* endpoints require a matching X-Admin-Key header.
    admin_api_key: str | None = None
    # Optional: when set, /metrics requires `Authorization: Bearer <token>`.
    metrics_auth_token: str | None = None

    # Redis (vector store + cache)
    redis_url: str = "redis://localhost:6379"
    index_name: str = "llmcache"
    index_prefix: str = "llmcache"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Cache policy
    similarity_threshold: float = 0.95
    default_ttl_seconds: int = 3600
    short_ttl_seconds: int = 300       # time-sensitive / strict profile
    long_ttl_seconds: int = 86400      # stable / relaxed profile
    max_ttl_seconds: int = 604800      # hard ceiling for any caller TTL override (7d)
    # Rough blended $/1k tokens, used only for the "estimated cost saved" metric.
    estimated_price_per_1k_tokens: float = 0.002
    # Temperature bands that drive the inferred cache profile.
    deterministic_temperature_max: float = 0.3   # at/below -> relaxed (looser match OK)
    creative_temperature_min: float = 0.8        # at/above -> no caching


@lru_cache
def get_settings() -> Settings:
    return Settings()

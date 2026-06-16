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


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Prompt embedding.

The `Embedder` protocol keeps the cache engine decoupled from any single
provider — production uses `OpenAIEmbedder`; tests inject a deterministic fake.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Embeds prompts with OpenAI `text-embedding-3-small` (default)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — cannot create OpenAIEmbedder"
            )
        # Imported lazily so the package imports without the SDK configured.
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

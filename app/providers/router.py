"""Route a request to a provider by its `model` field."""

from __future__ import annotations

_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def provider_key_for_model(model: str) -> str:
    """Map a model name to a provider key: openai | anthropic | ollama.

    Unknown models default to Ollama (local), which fails cleanly if the model
    isn't pulled — preferable to silently mis-routing to a hosted provider.
    """
    name = model.lower()
    if name.startswith(_OPENAI_PREFIXES):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    return "ollama"

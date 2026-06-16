"""Cache namespace derivation.

Two requests may share a prompt but must NOT share a cache entry if anything
that changes the model output differs (model, system prompt, sampling params).
The namespace isolates those, preventing cross-contamination between use cases.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Sampling params that materially change the output distribution.
_KEY_PARAMS: tuple[str, ...] = ("temperature", "top_p")


def cache_namespace(
    model: str,
    system_prompt: str | None,
    params: dict[str, Any] | None,
) -> str:
    """Stable 32-char namespace for a (model, system prompt, params) tuple."""
    params = params or {}
    payload = {
        "model": model,
        "system": system_prompt or "",
        "params": {k: params.get(k) for k in _KEY_PARAMS},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

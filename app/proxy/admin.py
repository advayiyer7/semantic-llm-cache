"""Admin endpoints — threshold tuning.

The tuner turns caller input into paid embedding calls, so it is bounded
(pair count + string length) and can be gated behind an admin key
(`ADMIN_API_KEY`); when no key is configured it stays open for local dev.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.cache.tuner import cosine, sweep
from app.metrics.prometheus import recent_near_misses

admin_router = APIRouter()

_DEFAULT_THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
_MAX_PAIRS = 50
_MAX_TEXT = 4000


class TunerPair(BaseModel):
    query: str = Field(..., max_length=_MAX_TEXT)
    candidate: str = Field(..., max_length=_MAX_TEXT)
    should_hit: bool


class TunerRequest(BaseModel):
    pairs: list[TunerPair] = Field(..., min_length=1, max_length=_MAX_PAIRS)
    thresholds: list[float] | None = None


def _require_admin(request: Request, x_admin_key: str | None) -> None:
    configured = request.app.state.settings.admin_api_key
    if configured and x_admin_key != configured:
        raise HTTPException(status_code=401, detail="Unauthorized.")


@admin_router.post("/admin/threshold-tuner")
async def threshold_tuner(
    body: TunerRequest,
    request: Request,
    x_admin_key: str | None = Header(default=None),
):
    """Sweep similarity thresholds over labeled pairs; report hit-rate vs precision."""
    _require_admin(request, x_admin_key)

    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        raise HTTPException(
            status_code=503, detail="Cache/embedder not configured (set OPENAI_API_KEY)."
        )

    # Embed all texts concurrently (bounded by the thread pool and _MAX_PAIRS).
    texts = [p.query for p in body.pairs] + [p.candidate for p in body.pairs]
    vectors = await asyncio.gather(
        *(asyncio.to_thread(cache.embedder.embed, t) for t in texts)
    )
    n = len(body.pairs)
    scored = [
        (cosine(vectors[i], vectors[n + i]), body.pairs[i].should_hit) for i in range(n)
    ]

    thresholds = sorted(body.thresholds) if body.thresholds else _DEFAULT_THRESHOLDS
    rows = sweep(scored, thresholds)
    best = max(rows, key=lambda r: (r["f1"], r["threshold"])) if rows else None
    return {"results": rows, "recommended": best}


@admin_router.get("/admin/near-misses")
async def near_misses(request: Request, x_admin_key: str | None = Header(default=None)):
    """Recent lookups that missed but landed just below the threshold."""
    _require_admin(request, x_admin_key)
    return {"near_misses": recent_near_misses()}

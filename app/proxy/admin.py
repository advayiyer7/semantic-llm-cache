"""Admin endpoints — threshold tuning."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.cache.tuner import cosine, sweep

admin_router = APIRouter()

_DEFAULT_THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


class TunerPair(BaseModel):
    query: str
    candidate: str
    should_hit: bool


class TunerRequest(BaseModel):
    pairs: list[TunerPair] = Field(..., min_length=1)
    thresholds: list[float] | None = None


@admin_router.post("/admin/threshold-tuner")
async def threshold_tuner(body: TunerRequest, request: Request):
    """Sweep similarity thresholds over labeled pairs; report hit-rate vs precision."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="Cache/embedder not configured (set OPENAI_API_KEY).",
        )

    scored: list[tuple[float, bool]] = []
    for pair in body.pairs:
        query_vec = await asyncio.to_thread(cache.embedder.embed, pair.query)
        candidate_vec = await asyncio.to_thread(cache.embedder.embed, pair.candidate)
        scored.append((cosine(query_vec, candidate_vec), pair.should_hit))

    thresholds = sorted(body.thresholds) if body.thresholds else _DEFAULT_THRESHOLDS
    rows = sweep(scored, thresholds)
    best = max(rows, key=lambda r: r["f1"]) if rows else None
    return {"results": rows, "recommended": best}

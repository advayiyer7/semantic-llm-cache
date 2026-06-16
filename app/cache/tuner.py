"""Threshold tuner — hit-rate vs precision tradeoff across thresholds.

Given labeled (query, candidate, should_hit) pairs and their cosine
similarities, sweep candidate thresholds and report, for each:
  - hit_rate:  fraction of pairs that would be served from cache
  - precision: of served hits, the fraction that *should* have hit
  - recall:    of pairs that should hit, the fraction that did
  - f1:        harmonic mean of precision and recall

This is the data behind picking a similarity threshold deliberately rather than
guessing — reporting precision (not just hit rate) is the project's whole point.
"""

from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def sweep(scored_pairs: list[tuple[float, bool]], thresholds: list[float]) -> list[dict]:
    """scored_pairs: list of (similarity, should_hit)."""
    rows: list[dict] = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for similarity, should_hit in scored_pairs:
            predicted_hit = similarity >= threshold
            if predicted_hit and should_hit:
                tp += 1
            elif predicted_hit and not should_hit:
                fp += 1
            elif not predicted_hit and should_hit:
                fn += 1
            else:
                tn += 1
        total = tp + fp + fn + tn
        served = tp + fp
        hit_rate = served / total if total else 0.0
        precision = tp / served if served else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": round(threshold, 3),
                "hit_rate": round(hit_rate, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows

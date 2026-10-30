"""Semantic Pruning Layer (Gateway): the core contribution under test.

Pipeline per the research proposal: hierarchical chunking of the raw tool
JSON (one chunk per row) -> sentence-embedding each row -> cosine-similarity
relevance scoring against a query "intent vector" -> keep only rows above
an adaptive threshold, pruning the rest. Pruned rows are cached so a
"fetch-more" tool call can pull the next-most-relevant batch back in
instead of losing them outright.
"""
from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _row_to_text(row: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))


def rank_by_relevance(rows: list[dict], intent: str) -> list[tuple[dict, float]]:
    """Ranks rows by cosine similarity of their text representation to the intent."""
    if not rows:
        return []
    model = _embedder()
    texts = [_row_to_text(r) for r in rows]
    row_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    intent_embedding = model.encode([intent], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = row_embeddings @ intent_embedding
    ranked = sorted(zip(rows, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked


def adaptive_prune(
    ranked: list[tuple[dict, float]], min_keep: int = 5, max_keep: int = 15
) -> tuple[list[dict], list[dict]]:
    """Keeps rows scoring at/above (mean + 0.5*std), bounded to [min_keep, max_keep].

    The mean+0.5*std threshold is the "Adaptive Thresholding" step in the
    proposal: it moves with each observation's own score distribution
    rather than using a fixed cosine cutoff, since relevance spread varies
    a lot between e.g. a 3-row distance lookup and a 50-row hotel search.
    """
    if not ranked:
        return [], []
    scores = np.array([s for _, s in ranked])
    threshold = scores.mean() + 0.5 * scores.std()
    keep_count = int((scores >= threshold).sum())
    keep_count = max(min_keep, min(keep_count, max_keep, len(ranked)))
    kept = [r for r, _ in ranked[:keep_count]]
    pruned = [r for r, _ in ranked[keep_count:]]
    return kept, pruned


_PRICE_FIELDS = ["price", "Price", "Average Cost", "cost"]


def _price(row: dict) -> float | None:
    for field in _PRICE_FIELDS:
        val = row.get(field)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def constraint_aware_prune(
    rows: list[dict],
    intent: str,
    price_floor_keep: int = 6,
    min_keep: int = 5,
    max_keep: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Adaptive semantic pruning with a numeric-constraint guardrail.

    Pure cosine similarity to the intent text has no notion of "cheap" or
    "budget-compliant" - a listing's price is not part of what gets
    embedded as meaning. Diagnosed on a real pilot query: the single
    cheapest accommodation ($144/night) was pruned while $394-$1,034/night
    options were kept, breaking an otherwise-satisfiable budget constraint.

    This guardrail force-keeps the `price_floor_keep` lowest-priced rows
    (when a price field exists) regardless of semantic score, then fills
    the rest of the kept set by relevance as before.
    """
    if not rows:
        return [], []

    ranked = rank_by_relevance(rows, intent)
    semantic_kept, _ = adaptive_prune(ranked, min_keep=min_keep, max_keep=max_keep)

    priced = [(r, p) for r in rows if (p := _price(r)) is not None]
    floor_rows = [r for r, _ in sorted(priced, key=lambda x: x[1])[:price_floor_keep]]

    kept_ids = {id(r) for r in semantic_kept} | {id(r) for r in floor_rows}
    kept = [r for r, _ in ranked if id(r) in kept_ids]
    if len(kept) > max_keep:
        # Trim lowest-relevance non-floor rows first; never drop the price floor.
        floor_id_set = {id(r) for r in floor_rows}
        trimmed = [r for r, _ in ranked if id(r) in kept_ids]
        overflow = len(trimmed) - max_keep
        for r, _ in reversed(ranked):
            if overflow <= 0:
                break
            if id(r) in kept_ids and id(r) not in floor_id_set:
                kept_ids.discard(id(r))
                overflow -= 1
        kept = [r for r, _ in ranked if id(r) in kept_ids]

    pruned = [r for r, _ in ranked if id(r) not in kept_ids]
    return kept, pruned

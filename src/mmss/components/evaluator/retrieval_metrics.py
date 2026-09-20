"""Retrieval quality metrics: recall@k, reciprocal rank, computed against a
labeled (query -> expected chunk ids) set.

`mean_reciprocal_rank` computes the reciprocal rank for a single query's
retrieved list, despite the name -- matching the signature already committed
to this module (one ScoredChunk list in, one float out). The actual "mean"
happens one level up, where the eval runner averages this value across every
sample in a run to get the aggregate MRR.
"""

from __future__ import annotations

from mmss.components.base import ScoredChunk


def recall_at_k(retrieved: list[ScoredChunk], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 1.0  # nothing to find -- vacuously satisfied, avoids a 0/0
    top_k_ids = {sc.chunk.id for sc in retrieved[:k]}
    return len(expected_ids & top_k_ids) / len(expected_ids)


def mean_reciprocal_rank(retrieved: list[ScoredChunk], expected_ids: set[str]) -> float:
    if not expected_ids:
        return 1.0
    for rank, sc in enumerate(retrieved, start=1):
        if sc.chunk.id in expected_ids:
            return 1.0 / rank
    return 0.0

"""Query pipeline: retrieve (dense+bm25) -> RRF fuse -> [rerank] -> [pot] -> generate -> guardrail.

M4 adds a "reranked" strategy: hybrid retrieval fetches a wider candidate pool
(top_k * rerank_candidate_multiplier) than the final answer count, and a local
cross-encoder rescoring narrows it back down -- selectable alongside "dense"
and "hybrid" for side-by-side comparison, same pattern as M3. Still returns
ScoredChunks directly -- no generator/guardrail until M5.
"""

from __future__ import annotations

from typing import Literal

from mmss.components.base import ScoredChunk
from mmss.components.embedder import get_embedder
from mmss.components.reranker import get_reranker
from mmss.components.retriever.bm25_retriever import BM25Retriever
from mmss.components.retriever.dense_retriever import DenseRetriever
from mmss.components.retriever.hybrid import HybridRetriever
from mmss.components.vector_store import get_vector_store
from mmss.config import get_config


def run_query(
    question: str,
    top_k: int | None = None,
    strategy: Literal["dense", "hybrid", "reranked"] = "hybrid",
) -> list[ScoredChunk]:
    cfg = get_config().retrieval
    top_k = top_k if top_k is not None else cfg.final_top_k

    embedder = get_embedder()
    vector_store = get_vector_store()
    dense = DenseRetriever(embedder, vector_store)

    if strategy == "dense":
        return dense.retrieve(question, top_k=top_k)

    bm25 = BM25Retriever(vector_store.get_all())
    hybrid = HybridRetriever(
        [(dense, cfg.top_k_dense), (bm25, cfg.top_k_bm25)],
        rrf_k=cfg.rrf_k,
    )

    if strategy == "hybrid":
        return hybrid.retrieve(question, top_k=top_k)

    # "reranked": widen the hybrid fetch so the cross-encoder gets a real
    # candidate pool to rescore, then narrow back down to top_k.
    candidate_k = top_k * cfg.rerank_candidate_multiplier
    candidates = hybrid.retrieve(question, top_k=candidate_k)
    reranker = get_reranker()
    return reranker.rerank(question, candidates, top_k=top_k)

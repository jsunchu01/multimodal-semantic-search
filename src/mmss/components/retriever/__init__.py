"""Retriever package: dense, BM25, and hybrid (RRF-fused) retrieval.

Unlike the other component packages, these three aren't config-swapped
alternatives of each other -- the query pipeline composes them directly
(dense + bm25 feeding into hybrid), so there's no registry/build() here.
"""

from __future__ import annotations

from mmss.components.retriever.base import Retriever
from mmss.components.retriever.bm25_retriever import BM25Retriever
from mmss.components.retriever.dense_retriever import DenseRetriever
from mmss.components.retriever.hybrid import HybridRetriever

__all__ = ["Retriever", "BM25Retriever", "DenseRetriever", "HybridRetriever"]

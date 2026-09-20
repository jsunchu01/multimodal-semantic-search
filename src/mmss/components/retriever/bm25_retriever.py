"""BM25 keyword retriever, built fresh from the current corpus each time.

Uses the `rank_bm25` library (BM25Okapi) rather than a hand-rolled
implementation -- verified against its current README, standard usage is
`BM25Okapi(tokenized_corpus)` + `.get_scores(tokenized_query)`.

Our CLI is a fresh process per invocation, not a long-lived server, so
there's no persisted-index lifecycle to manage: rebuilding from all chunks
currently in the vector store costs milliseconds at personal-project data
volumes (one or a few documents, a few hundred chunks).
"""

from __future__ import annotations

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.retriever.base import Retriever
from mmss.utils.text import simple_tokenize


class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = None
        if chunks:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([simple_tokenize(c.text) for c in chunks])

    @property
    def name(self) -> str:
        return "bm25"

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(simple_tokenize(query))
        ranked = sorted(zip(self._chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            ScoredChunk(chunk=chunk, score=float(score), source="bm25")
            for chunk, score in ranked[:top_k]
            if score > 0  # a score of 0 means no query term matched at all
        ]

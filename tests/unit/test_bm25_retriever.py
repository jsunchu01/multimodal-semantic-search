"""Tests BM25Retriever against synthetic chunks -- no DB or embedding needed."""

from mmss.components.base import Chunk
from mmss.components.retriever.bm25_retriever import BM25Retriever


def _chunk(id_: str, text: str) -> Chunk:
    return Chunk(
        id=id_, doc_id="doc1", text=text, chunk_type="text",
        page_start=1, page_end=1, source_parser="test",
    )


def test_bm25_ranks_exact_term_match_higher() -> None:
    chunks = [
        _chunk("a", "the quick brown fox jumps over the lazy dog"),
        _chunk("b", "total revenue increased significantly this quarter"),
        _chunk("c", "completely unrelated text about weather patterns"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("total revenue", top_k=3)

    assert len(results) == 1  # "a" and "c" score exactly 0 -- no term overlap
    assert results[0].chunk.id == "b"


def test_bm25_excludes_zero_score_matches() -> None:
    chunks = [
        _chunk("a", "apples and oranges"),
        _chunk("b", "completely different subject matter"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("xyz nonexistent term", top_k=5)

    assert results == []


def test_bm25_empty_corpus_returns_empty() -> None:
    retriever = BM25Retriever([])
    assert retriever.retrieve("anything", top_k=5) == []

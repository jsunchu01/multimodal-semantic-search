"""Shared value types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawElement:
    """One unchunked unit of output from a DocumentParser, before chunking."""

    type: str  # "text" | "table" | "heading" | "caption" | "image"
    text: str
    page_number: int | None
    source_parser: str
    bbox: dict | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval unit produced by the chunker, ready to embed and index."""

    id: str
    doc_id: str
    text: str
    chunk_type: str  # "text" | "table" | "chart" | "caption"
    page_start: int
    page_end: int
    source_parser: str
    bbox: dict | None = None
    content_hash: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk with a retrieval/rerank score attached."""

    chunk: Chunk
    score: float
    source: str  # "dense" | "bm25" | "hybrid" | "rerank"


@dataclass
class GenerationResult:
    """The generator's answer plus citations back to source chunks."""

    text: str
    citations: list[str] = field(default_factory=list)
    raw_response: str | None = None


@dataclass
class VisionExtractionResult:
    """Structured data extracted from a chart/figure image."""

    description: str
    data_points: list[dict] = field(default_factory=list)
    chart_type: str | None = None
    raw_response: str | None = None

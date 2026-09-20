"""Shared chunker: normalizes RawElements from any parser backend into Chunks.

Modeled on the reference repo's verified layout_parser.py behavior: pair
tables/images with an adjacent caption, merge consecutive table chunks that
continue on the very next page, and split text runs at a character budget.

Deliberately simple for M1: char-budget splitting is a plain greedy
accumulator with no overlap-carry-forward yet -- that's a refinement for
later once this simpler version has been confirmed correct against a real
PDF, not something added now and left untested.
"""

from __future__ import annotations

import hashlib

from mmss.components.base import Chunk, RawElement
from mmss.components.parser.base import Chunker
from mmss.utils.table_flatten import flatten_html_table


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class DefaultChunker(Chunker):
    def __init__(self, max_characters: int = 2000) -> None:
        self._max_characters = max_characters

    def chunk(self, elements: list[RawElement], doc_id: str) -> list[Chunk]:
        chunks = self._build_chunks(elements, doc_id)
        return self._merge_cross_page_tables(chunks)

    def _build_chunks(self, elements: list[RawElement], doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        text_buffer: list[RawElement] = []
        buffer_chars = 0

        def flush_text() -> None:
            nonlocal buffer_chars
            if not text_buffer:
                return
            combined = "\n\n".join(e.text for e in text_buffer)
            pages = [e.page_number for e in text_buffer if e.page_number is not None]
            chunks.append(
                self._make_chunk(
                    doc_id,
                    len(chunks),
                    combined,
                    "text",
                    min(pages) if pages else 0,
                    max(pages) if pages else 0,
                    text_buffer[0].source_parser,
                )
            )
            text_buffer.clear()
            buffer_chars = 0

        for i, el in enumerate(elements):
            if el.type == "caption":
                continue  # only ever consumed by an adjacent table/image below

            if el.type in ("table", "image"):
                flush_text()
                caption = self._find_adjacent_caption(elements, i)
                chunk_type = "chart" if el.type == "image" else "table"
                # Tables are flattened to natural-language rows -- see
                # table_flatten.py for why (BM25/dense/reranker all read raw
                # HTML markup as noise otherwise). Original kept in metadata.
                body = flatten_html_table(el.text) if el.type == "table" else el.text
                text = f"{caption}\n\n{body}" if caption else body
                page = el.page_number or 0
                # Tables carry the original HTML; images carry image_path
                # (set by DoclingParser when Docling could export the
                # picture) for the ingest pipeline's vision-extraction stage.
                metadata = {"raw_html": el.text} if el.type == "table" else dict(el.metadata)
                chunks.append(
                    self._make_chunk(
                        doc_id, len(chunks), text, chunk_type, page, page, el.source_parser, metadata
                    )
                )
                continue

            if buffer_chars + len(el.text) > self._max_characters:
                flush_text()
            text_buffer.append(el)
            buffer_chars += len(el.text)

        flush_text()
        return chunks

    @staticmethod
    def _find_adjacent_caption(elements: list[RawElement], index: int) -> str | None:
        """Looks one element back and one forward on the same page for a caption."""
        el = elements[index]
        for j in (index - 1, index + 1):
            if 0 <= j < len(elements):
                neighbor = elements[j]
                if neighbor.type == "caption" and neighbor.page_number == el.page_number:
                    return neighbor.text
        return None

    @staticmethod
    def _merge_cross_page_tables(chunks: list[Chunk]) -> list[Chunk]:
        """Merges consecutive table chunks that continue on the very next page."""
        merged: list[Chunk] = []
        for c in chunks:
            if (
                merged
                and merged[-1].chunk_type == "table"
                and c.chunk_type == "table"
                and c.page_start == merged[-1].page_end + 1
            ):
                prev = merged.pop()
                combined_text = f"{prev.text}\n{c.text}"
                combined_raw_html = f"{prev.metadata.get('raw_html', '')}\n{c.metadata.get('raw_html', '')}"
                merged.append(
                    Chunk(
                        id=prev.id,
                        doc_id=prev.doc_id,
                        text=combined_text,
                        chunk_type="table",
                        page_start=prev.page_start,
                        page_end=c.page_end,
                        source_parser=prev.source_parser,
                        content_hash=_content_hash(combined_text),
                        metadata={"raw_html": combined_raw_html},
                    )
                )
            else:
                merged.append(c)
        return merged

    @staticmethod
    def _make_chunk(
        doc_id: str,
        index: int,
        text: str,
        chunk_type: str,
        page_start: int,
        page_end: int,
        source_parser: str,
        metadata: dict | None = None,
    ) -> Chunk:
        return Chunk(
            id=f"{doc_id}_{index:04d}",
            doc_id=doc_id,
            text=text,
            chunk_type=chunk_type,
            page_start=page_start,
            page_end=page_end,
            source_parser=source_parser,
            content_hash=_content_hash(text),
            metadata=metadata or {},
        )

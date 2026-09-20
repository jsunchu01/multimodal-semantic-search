"""Tests the chunking algorithm directly against synthetic RawElements --
no Docling/Unstructured install needed, since DefaultChunker only depends on
our own RawElement/Chunk types. This is the one part of M1 we can verify
without a live parser library or a real PDF.
"""

from mmss.components.base import RawElement
from mmss.components.parser.chunker import DefaultChunker


def _el(type_: str, text: str, page: int) -> RawElement:
    return RawElement(type=type_, text=text, page_number=page, source_parser="test")


def test_text_splits_at_char_budget() -> None:
    # budget=20: "a"*8 + "b"*8 fit together (16<=20), adding "c"*8 would hit
    # 24>20 so it flushes and starts a new buffer with "c"+"d" (16<=20).
    chunker = DefaultChunker(max_characters=20)
    elements = [
        _el("text", "a" * 8, 1),
        _el("text", "b" * 8, 1),
        _el("text", "c" * 8, 1),
        _el("text", "d" * 8, 1),
    ]
    chunks = chunker.chunk(elements, doc_id="doc1")

    assert [c.chunk_type for c in chunks] == ["text", "text"]
    assert "a" * 8 in chunks[0].text and "b" * 8 in chunks[0].text
    assert "c" * 8 in chunks[1].text and "d" * 8 in chunks[1].text


def test_caption_pairs_with_following_table() -> None:
    chunker = DefaultChunker()
    elements = [
        _el("text", "intro paragraph", 1),
        _el("caption", "Table 1: Revenue by segment", 1),
        _el("table", "<table><tr><td>1</td></tr></table>", 1),
    ]
    chunks = chunker.chunk(elements, doc_id="doc1")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Table 1: Revenue by segment" in table_chunks[0].text
    # Table body is flattened to plain text (no header row detected in this
    # single-cell table, so it falls back to tag-stripped text) -- raw HTML
    # is preserved separately rather than living in .text.
    assert "<table>" not in table_chunks[0].text
    assert "<table>" in table_chunks[0].metadata["raw_html"]


def test_caption_pairs_with_preceding_image() -> None:
    chunker = DefaultChunker()
    elements = [
        _el("image", "", 2),
        _el("caption", "Figure 3: Stock price trend", 2),
    ]
    chunks = chunker.chunk(elements, doc_id="doc1")

    chart_chunks = [c for c in chunks if c.chunk_type == "chart"]
    assert len(chart_chunks) == 1
    assert "Figure 3: Stock price trend" in chart_chunks[0].text


def test_cross_page_tables_merge() -> None:
    chunker = DefaultChunker()
    elements = [
        _el("table", "<table><tr><td>row1</td></tr></table>", 5),
        _el("table", "<table><tr><td>row2</td></tr></table>", 6),
    ]
    chunks = chunker.chunk(elements, doc_id="doc1")

    assert len(chunks) == 1
    assert chunks[0].page_start == 5
    assert chunks[0].page_end == 6
    assert "row1" in chunks[0].text and "row2" in chunks[0].text
    assert "row1" in chunks[0].metadata["raw_html"] and "row2" in chunks[0].metadata["raw_html"]


def test_non_adjacent_page_tables_do_not_merge() -> None:
    chunker = DefaultChunker()
    elements = [
        _el("table", "<table>first</table>", 5),
        _el("text", "unrelated paragraph in between", 6),
        _el("table", "<table>second</table>", 8),
    ]
    chunks = chunker.chunk(elements, doc_id="doc1")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 2


def test_chunk_ids_and_content_hash_are_populated() -> None:
    chunker = DefaultChunker()
    elements = [_el("text", "some text", 1)]
    chunks = chunker.chunk(elements, doc_id="doc1")

    assert chunks[0].id == "doc1_0000"
    assert chunks[0].content_hash is not None
    assert len(chunks[0].content_hash) == 12

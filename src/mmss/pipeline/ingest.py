"""Ingestion pipeline: parse -> chunk -> vision -> embed -> index.

M1 implemented parse + chunk, writing chunks to data/processed/<doc_id>.json.
M2 adds embedding + indexing into the vector store. M6 adds the vision stage:
chart chunks carry an image_path in metadata (set by DoclingParser, carried
through by the chunker) -- each gets sent to the configured VisionExtractor,
replacing the empty placeholder text with a real description (and, if
structured extraction is on, chart_type/data_points too).

M7 runs chart descriptions concurrently (bounded thread pool, see
utils/concurrency.py) instead of one at a time -- a real wall-clock win for
a document with several charts and an API vision vendor (OpenAI/Together),
since each call is pure network wait. For a local Ollama model this doesn't
meaningfully speed anything up (one local model serializing requests either
way), but it's harmless there too -- not worth a separate code path for.
"""

from __future__ import annotations

from pathlib import Path

from mmss.components.base import Chunk
from mmss.components.embedder import get_embedder
from mmss.components.parser import get_chunker, get_parser
from mmss.components.vector_store import get_vector_store
from mmss.components.vision import get_vision_extractor
from mmss.utils.concurrency import run_concurrently
from mmss.utils.io import write_json_list
from mmss.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_CONCURRENT_VISION_CALLS = 4


def _describe_charts(chunks: list[Chunk]) -> list[Chunk]:
    chart_chunks = [c for c in chunks if c.chunk_type == "chart" and c.metadata.get("image_path")]
    if not chart_chunks:
        return chunks

    vision = get_vision_extractor()

    def _describe_one(chunk: Chunk) -> None:
        image_path = chunk.metadata["image_path"]
        try:
            result = vision.extract(image_path, context_text=chunk.text or None)
            chunk.text = result.description
            if result.chart_type is not None:
                chunk.metadata["chart_type"] = result.chart_type
            if result.data_points:
                chunk.metadata["data_points"] = result.data_points
        except Exception:
            logger.warning(
                "Vision extraction failed for %s (chunk %s) -- leaving it as a placeholder chunk",
                image_path,
                chunk.id,
            )

    run_concurrently(_describe_one, chart_chunks, max_workers=_MAX_CONCURRENT_VISION_CALLS)
    return chunks


def run_ingest(file_path: str) -> None:
    doc_id = Path(file_path).stem
    parser = get_parser()
    chunker = get_chunker()
    embedder = get_embedder()
    vector_store = get_vector_store()

    elements = parser.parse(file_path)
    chunks = chunker.chunk(elements, doc_id=doc_id)
    chunks = _describe_charts(chunks)

    out_path = Path("data/processed") / f"{doc_id}.json"
    write_json_list(chunks, out_path)

    if chunks:
        embeddings = embedder.embed_documents([c.text for c in chunks])
        vector_store.add(chunks, embeddings)
        vector_store.persist()

    logger.info(
        "Ingested %s: %d elements -> %d chunks (parser=%s) -> %s, embedded with %s into %s",
        file_path,
        len(elements),
        len(chunks),
        parser.name,
        out_path,
        embedder.name,
        vector_store.name,
    )


def run_delete(doc_id: str) -> None:
    """Removes a previously-ingested document: its rows in the vector store,
    its processed JSON, and its raw source file(s) -- the inverse of
    run_ingest(). Nothing currently does this automatically; documents
    persist until explicitly deleted."""
    get_vector_store().delete(doc_id)

    processed_path = Path("data/processed") / f"{doc_id}.json"
    if processed_path.exists():
        processed_path.unlink()

    for raw_path in Path("data/raw").glob(f"{doc_id}.*"):
        raw_path.unlink()

    logger.info("Deleted %s: vector store rows, processed JSON, and raw file(s) removed", doc_id)

"""Streamlit front end: upload PDFs (financial filings, research papers, or
anything else), ask questions, see grounded answers with the actual source
PDF pages shown for verification -- no new pipeline logic, this just calls
run_ingest()/run_generate()/run_delete() directly in-process and renders
what they already return.

Verified precedent from the reference repo's own demo (src/rag_system's
demo/app.py): it's Streamlit too, and it also bypasses their FastAPI layer
entirely, importing the pipeline as a plain Python library instead. We do
the same, more directly still -- no SDK/pipeline wrapper class of our own to
go through, just the pipeline.* functions the CLI already calls.

Caching note: get_embedder()/get_vector_store()/get_generator()/
get_reranker()/get_vision_extractor() are @lru_cache(maxsize=1) (see each
components/*/__init__.py) specifically so this long-lived process builds
each heavy local model once, not once per question.

The sidebar's document list is read fresh from the vector store itself
(get_vector_store().get_all(), deduped by doc_id) rather than tracked in
st.session_state -- session state would only show what THIS browser session
uploaded and forget everything on a restart, when the actual ingested data
lives in Postgres and persists across restarts regardless. Nothing deletes
data automatically (no expiry, no TTL) -- the delete button next to each
document is the only way data ever leaves the store today.

Styling: color comes from .streamlit/config.toml's [theme] (Streamlit's own
palette system -- every button/slider/selectbox picks it up automatically),
not hand-placed CSS. Icons are Streamlit's built-in Material Symbols
(`icon=":material/name:"`, verified against the current st.button/st.markdown
docs) rather than emoji -- vector icons that match the theme, not cartoon
glyphs.

Display is deliberately plain: no chat bubbles/avatars (st.chat_message
always shows one or the other by design -- verified, there's no way to turn
it off -- so this skips that widget entirely rather than fight it), no
metadata dump. Just the question, the answer, a grounding percentage, and
the actual source PDF page(s) rendered as images (via pymupdf, already a
base dependency) so a claim can be visually checked against the real page
instead of trusting a page-number citation on faith.

Run with: streamlit run src/mmss/ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import streamlit as st

from mmss.components.vector_store import get_vector_store
from mmss.config import get_config
from mmss.pipeline.generate import NOT_FOUND_PHRASE, AnswerReport, run_generate
from mmss.pipeline.ingest import run_delete, run_ingest

st.set_page_config(page_title="Document Search", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _list_documents() -> list[str]:
    try:
        chunks = get_vector_store().get_all()
    except Exception:
        return []
    return sorted({c.doc_id for c in chunks})


def _render_pdf_page(doc_id: str, page_number: int) -> bytes | None:
    """Renders one page of the original PDF to a PNG so a source citation
    can be visually checked, not just trusted as a page number. Returns
    None (caller just skips it) if the raw file or page isn't available --
    e.g. the file was deleted, or an older ingest ran before pages were
    tracked correctly."""
    pdf_path = Path("data/raw") / f"{doc_id}.pdf"
    if not pdf_path.exists() or page_number < 1:
        return None
    try:
        with pymupdf.open(pdf_path) as doc:
            if page_number > len(doc):
                return None
            page = doc[page_number - 1]  # 0-indexed
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5))  # 2.5x zoom for readability
            return pix.tobytes("png")
    except Exception:
        return None


def _build_answer_entry(question: str, report: AnswerReport) -> dict:
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []
    for scored in report.retrieved_chunks:
        key = (scored.chunk.doc_id, scored.chunk.page_start)
        if key not in seen:
            seen.add(key)
            # .score here is the reranker's relevance score for this chunk
            # (or the RRF fusion score if strategy isn't "reranked") -- a
            # different thing from the answer-level grounding_ratio below,
            # and not naturally a 0-100 percentage (cross-encoder scores are
            # raw, unbounded), so it's shown as a plain score, not a %.
            sources.append({"doc_id": key[0], "page": key[1], "score": scored.score})
        if len(sources) == 3:  # top 3 unique source pages -- enough to verify, not a wall of images
            break
    return {
        "question": question,
        "answer": report.answer.text,
        "grounding_ratio": report.guardrail.grounding_ratio,
        "sources": sources,
        # Retrieval always returns the top-k NEAREST chunks with no
        # relevance threshold, even when nothing in the corpus actually
        # answers the question -- so a chunk/score always exists even for
        # an answer the model itself says isn't backed by any source.
        # Detected via the same NOT_FOUND_PHRASE the system prompt
        # instructs the model to use (imported, not duplicated, so this
        # can't silently drift out of sync with the actual prompt).
        "not_found": NOT_FOUND_PHRASE.lower() in report.answer.text.lower(),
    }


def _render_source_image(source: dict) -> None:
    """Just the page image + its caption -- no score/text attached here
    anymore, that's shown as its own metric beside the image instead so it
    doesn't get squeezed into a tiny caption under a (previously) tiny
    image."""
    image_bytes = _render_pdf_page(source["doc_id"], source["page"])
    if image_bytes is not None:
        st.image(image_bytes, caption=f"{source['doc_id']} -- page {source['page']}", width="stretch")


def _render_source_row(text_col, image_col, source: dict) -> None:
    with text_col:
        st.metric("Relevance score", f"{source['score']:.2f}")
    with image_col:
        with st.container(border=True):
            _render_source_image(source)


def _render_answer_entry(entry: dict) -> None:
    if entry["not_found"]:
        # No real source backed this answer -- showing a relevance score,
        # a grounding percentage, or "top sources" here would misrepresent
        # an irrelevant nearest-neighbor match as if it were evidence.
        st.write(entry["answer"])
        return

    sources = entry["sources"]
    top_source = sources[0] if sources else None

    col_text, col_image = st.columns([3, 2])
    with col_text:
        st.write(entry["answer"])
        if top_source is not None:
            st.metric("Relevance score", f"{top_source['score']:.2f}")
        st.markdown(f"**Numeric grounding confidence:** :blue[{entry['grounding_ratio']:.0%}]")
    with col_image:
        if top_source is not None:
            with st.container(border=True):
                _render_source_image(top_source)

    if len(sources) > 1:
        with st.expander("Next top 2 sources"):
            for source in sources[1:]:
                col_text2, col_image2 = st.columns([3, 2])
                _render_source_row(col_text2, col_image2, source)
                st.divider()


with st.sidebar:
    st.subheader("Settings")
    strategy = st.selectbox("Retrieval strategy", ["reranked", "hybrid", "dense"], index=0)
    top_k = st.number_input(
        "Chunks per answer", min_value=1, max_value=50, value=get_config().retrieval.final_top_k
    )

    st.divider()
    st.subheader("Your documents")
    documents = _list_documents()
    if not documents:
        st.caption("Nothing uploaded yet.")
    for doc_id in documents:
        col_name, col_delete = st.columns([4, 1])
        col_name.write(doc_id)
        if col_delete.button(
            "",
            icon=":material/delete_outline:",
            key=f"delete_{doc_id}",
            help=f"Delete {doc_id}",
            type="tertiary",
        ):
            with st.spinner(f"Deleting {doc_id}..."):
                run_delete(doc_id)
            st.rerun()

st.title("Document Search")
st.caption("Ask questions about your documents, with cited, numerically-verified answers.")

st.subheader("Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
if uploaded_files and st.button("Upload File(s)"):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        dest = raw_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                run_ingest(str(dest))
            except Exception as exc:
                st.error(f"Failed to ingest {uploaded.name}: {exc}")
            else:
                st.success(f"Ingested {uploaded.name}")
    st.rerun()

for entry in st.session_state.messages:
    st.markdown(f"**{entry['question']}**")
    _render_answer_entry(entry)
    st.divider()

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.spinner("Thinking..."):
        try:
            report = run_generate(question, top_k=int(top_k), strategy=strategy)  # type: ignore[arg-type]
        except Exception as exc:
            st.error(f"Failed to answer: {exc}")
            report = None

    if report is not None:
        entry = _build_answer_entry(question, report)
        st.session_state.messages.append(entry)
        st.markdown(f"**{entry['question']}**")
        _render_answer_entry(entry)

"""Command-line entrypoint for the mmss RAG pipeline."""

from __future__ import annotations

import typer

app = typer.Typer(help="Personal multimodal RAG system for document analysis (financial filings, research papers).")


@app.command()
def ingest(file_path: str) -> None:
    """Parse, chunk, embed, and index a document into the vector store."""
    from mmss.pipeline.ingest import run_ingest

    run_ingest(file_path)


@app.command()
def delete(doc_id: str) -> None:
    """Remove a previously-ingested document (vector store rows, processed
    JSON, and raw file(s)). doc_id is the filename without its extension,
    e.g. 'tesla_test' for tesla_test.pdf."""
    from mmss.pipeline.ingest import run_delete

    run_delete(doc_id)
    typer.echo(f"Deleted {doc_id}")


@app.command()
def query(
    question: str,
    top_k: int | None = typer.Option(
        None, help="How many results to return (default: config's retrieval.final_top_k)"
    ),
    strategy: str = typer.Option(
        "hybrid", help="'dense', 'hybrid', or 'reranked' -- for comparing retrieval quality"
    ),
) -> None:
    """Search the indexed documents (retrieval only until M5 adds generation)."""
    from mmss.pipeline.query import run_query

    if strategy not in ("dense", "hybrid", "reranked"):
        typer.echo(f"strategy must be 'dense', 'hybrid', or 'reranked', got '{strategy}'")
        raise typer.Exit(code=1)

    results = run_query(question, top_k=top_k, strategy=strategy)  # type: ignore[arg-type]
    if not results:
        typer.echo("No results (has anything been ingested yet?)")
        return
    for i, scored in enumerate(results, start=1):
        c = scored.chunk
        snippet = c.text[:200].replace("\n", " ")
        typer.echo(f"[{i}] score={scored.score:.4f} {c.doc_id} p{c.page_start}-{c.page_end} ({c.chunk_type})")
        typer.echo(f"    {snippet}{'...' if len(c.text) > 200 else ''}")


@app.command()
def ask(
    question: str,
    top_k: int | None = typer.Option(
        None,
        help="How many chunks to feed the generator (default: query analyzer's suggestion, "
        "e.g. wider for comparative/temporal questions)",
    ),
    strategy: str = typer.Option(
        "reranked", help="'dense', 'hybrid', or 'reranked' -- which retrieval feeds the answer"
    ),
) -> None:
    """Full RAG: analyze -> retrieve -> generate a cited answer -> guardrails."""
    from mmss.pipeline.generate import run_generate

    if strategy not in ("dense", "hybrid", "reranked"):
        typer.echo(f"strategy must be 'dense', 'hybrid', or 'reranked', got '{strategy}'")
        raise typer.Exit(code=1)

    report = run_generate(question, top_k=top_k, strategy=strategy)  # type: ignore[arg-type]

    typer.echo(report.answer.text)
    typer.echo("")
    typer.echo(f"Sources: {', '.join(report.answer.citations)}")
    typer.echo(f"Query intent: {report.query_intent} | Generator: {report.generator_used}")
    if report.pot_result is not None:
        status = "succeeded" if report.pot_result.success else f"failed ({report.pot_result.error})"
        typer.echo(f"Program-of-thought: {status}")
    typer.echo(
        f"Numeric grounding: {'passed' if report.guardrail.passed else 'FAILED'} "
        f"(ratio={report.guardrail.grounding_ratio:.2f})"
    )
    if report.guardrail.ungrounded_claims:
        typer.echo(f"  Ungrounded claims: {report.guardrail.ungrounded_claims}")
    if report.pii_entities_found:
        typer.echo(f"PII/financial identifiers redacted: {report.pii_entities_found}")


@app.command()
def evaluate(
    dataset: str = typer.Option(
        "evals/golden_dataset.jsonl", help="Path to a golden JSONL dataset"
    ),
    use_ragas: bool = typer.Option(
        True,
        help="Score with RAGAS (faithfulness/answer_relevancy/context_precision/context_recall) "
        "-- requires `pip install -e '.[eval]'` and OPENAI_API_KEY",
    ),
    top_k: int | None = typer.Option(
        None, help="Chunks retrieved per sample (default: config's retrieval.final_top_k)"
    ),
    strategy: str = typer.Option(
        "reranked", help="'dense', 'hybrid', or 'reranked' -- which retrieval feeds each sample"
    ),
) -> None:
    """Run the retrieval/answer evaluation harness against a golden dataset."""
    from mmss.pipeline.evaluate import run_eval

    if strategy not in ("dense", "hybrid", "reranked"):
        typer.echo(f"strategy must be 'dense', 'hybrid', or 'reranked', got '{strategy}'")
        raise typer.Exit(code=1)

    report = run_eval(
        dataset_path=dataset, use_ragas=use_ragas, top_k=top_k, strategy=strategy  # type: ignore[arg-type]
    )

    typer.echo(f"Run {report.run_id} -- {report.num_samples} samples")
    typer.echo(f"Pass rate: {report.pass_rate:.2%}  Groundedness rate: {report.groundedness_rate:.2%}")

    def _fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    typer.echo(f"Recall@k: {_fmt(report.avg_recall_at_k)}  MRR: {_fmt(report.avg_mrr)}")
    if use_ragas:
        typer.echo(
            f"Faithfulness: {_fmt(report.avg_faithfulness)}  "
            f"Answer relevancy: {_fmt(report.avg_answer_relevancy)}  "
            f"Context precision: {_fmt(report.avg_context_precision)}  "
            f"Context recall: {_fmt(report.avg_context_recall)}"
        )

    if report.regression_detected:
        typer.echo("REGRESSION DETECTED vs. previous run:")
        for note in report.regression_notes:
            typer.echo(f"  - {note}")

    for i, result in enumerate(report.results, start=1):
        status = "PASS" if result.passed else "FAIL"
        typer.echo(f"[{i}] {status} -- {result.question}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

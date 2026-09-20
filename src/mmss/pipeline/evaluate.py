"""Eval harness: runs a golden JSONL dataset through the real pipeline
(run_generate, which internally retrieves via run_query), scores each sample
with the deterministic metrics plus RAGAS, aggregates into an EvalReport, and
flags a simple regression against the previous run's numbers.

Regression detection is intentionally much lighter than the reference repo's
(which keeps the last 50 runs): this just compares against the single most
recent saved run, since a personal project won't have CI re-running this
continuously and a longer history wouldn't get used for anything here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mmss.components.evaluator.answer_metrics import groundedness_rate
from mmss.components.evaluator.base import EvalReport, EvalResult, EvalSample
from mmss.components.evaluator.ragas_evaluator import RagasEvaluator
from mmss.components.evaluator.retrieval_metrics import mean_reciprocal_rank, recall_at_k
from mmss.config import get_config
from mmss.pipeline.generate import run_generate
from mmss.utils.logging import get_logger

logger = get_logger(__name__)

_HISTORY_KEEP = 20


def _load_samples(path: Path) -> list[EvalSample]:
    samples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        samples.append(
            EvalSample(
                question=raw["question"],
                expected_chunk_ids=set(raw.get("expected_chunk_ids", [])),
                ground_truth=raw.get("ground_truth"),
                tags=raw.get("tags", []),
            )
        )
    return samples


def _average(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _score_sample(
    sample: EvalSample,
    ragas: RagasEvaluator | None,
    top_k: int | None,
    strategy: Literal["dense", "hybrid", "reranked"],
) -> EvalResult:
    start = time.perf_counter()
    report = run_generate(sample.question, top_k=top_k, strategy=strategy)
    elapsed_ms = (time.perf_counter() - start) * 1000

    retrieved = report.retrieved_chunks
    retrieved_ids = [sc.chunk.id for sc in retrieved]

    ragas_scores: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    if ragas is not None:
        ragas_scores = ragas.score(
            sample.question,
            report.answer.text,
            [sc.chunk.text for sc in retrieved],
            sample.ground_truth,
        )

    k = top_k or get_config().retrieval.final_top_k
    return EvalResult(
        question=sample.question,
        answer_text=report.answer.text,
        ground_truth=sample.ground_truth,
        retrieved_chunk_ids=retrieved_ids,
        recall_at_k=recall_at_k(retrieved, sample.expected_chunk_ids, k)
        if sample.expected_chunk_ids
        else None,
        mrr=mean_reciprocal_rank(retrieved, sample.expected_chunk_ids)
        if sample.expected_chunk_ids
        else None,
        guardrail=report.guardrail,
        latency_ms=elapsed_ms,
        tags=sample.tags,
        **ragas_scores,
    )


def _build_report(run_id: str, results: list[EvalResult]) -> EvalReport:
    pass_count = sum(1 for r in results if r.passed)
    return EvalReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=len(results),
        pass_rate=pass_count / len(results) if results else 1.0,
        groundedness_rate=groundedness_rate([r.guardrail for r in results if r.guardrail is not None]),
        avg_recall_at_k=_average([r.recall_at_k for r in results]),
        avg_mrr=_average([r.mrr for r in results]),
        avg_faithfulness=_average([r.faithfulness for r in results]),
        avg_answer_relevancy=_average([r.answer_relevancy for r in results]),
        avg_context_precision=_average([r.context_precision for r in results]),
        avg_context_recall=_average([r.context_recall for r in results]),
        results=results,
    )


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read eval history at %s -- treating as empty", path)
        return []


def _check_regression(report: EvalReport, history: list[dict], threshold: float) -> None:
    if not history:
        return
    previous = history[-1]
    notes = []
    for field_name in ("groundedness_rate", "avg_faithfulness"):
        prev_value = previous.get(field_name)
        curr_value = getattr(report, field_name)
        if prev_value is None or curr_value is None:
            continue
        drop = prev_value - curr_value
        if drop > threshold:
            notes.append(
                f"{field_name} dropped {drop:.3f} vs. previous run "
                f"({prev_value:.3f} -> {curr_value:.3f})"
            )
    if notes:
        report.regression_detected = True
        report.regression_notes = notes


def _save_history(path: Path, report: EvalReport) -> None:
    history = _load_history(path)
    entry = asdict(report)
    entry.pop("results")  # per-sample detail isn't needed for trend comparison
    history.append(entry)
    history = history[-_HISTORY_KEEP:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def run_eval(
    dataset_path: str = "evals/golden_dataset.jsonl",
    use_ragas: bool = True,
    top_k: int | None = None,
    strategy: Literal["dense", "hybrid", "reranked"] = "reranked",
) -> EvalReport:
    cfg = get_config()
    samples = _load_samples(Path(dataset_path))
    ragas = RagasEvaluator(judge_model=cfg.eval.judge_model) if use_ragas else None

    results = [_score_sample(sample, ragas, top_k, strategy) for sample in samples]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = _build_report(run_id, results)

    history_path = Path(cfg.eval.history_path)
    history = _load_history(history_path)
    _check_regression(report, history, cfg.eval.regression_threshold)
    _save_history(history_path, report)

    return report

"""Tests the eval runner's own logic -- JSONL loading, averaging, report
aggregation, regression detection, and history persistence -- plus one
end-to-end run_eval() test with run_generate mocked out (no real retrieval/
generation/DB needed, same style as other pipeline-level tests in this repo).
"""

from __future__ import annotations

import json

import pytest

from mmss.components.base import Chunk, GenerationResult, ScoredChunk
from mmss.components.evaluator.base import EvalReport, EvalResult, EvalSample
from mmss.components.guardrails.base import GuardrailResult
from mmss.config import get_config, reset_config
from mmss.pipeline import evaluate as evaluate_module
from mmss.pipeline.evaluate import (
    _average,
    _build_report,
    _check_regression,
    _load_history,
    _load_samples,
    _save_history,
    run_eval,
)
from mmss.pipeline.generate import AnswerReport


def _result(passed: bool = True, faithfulness: float | None = None,
            recall: float | None = None, mrr: float | None = None) -> EvalResult:
    guardrail = GuardrailResult(passed=passed, grounding_ratio=1.0 if passed else 0.0)
    return EvalResult(
        question="q", answer_text="a", ground_truth=None, retrieved_chunk_ids=[],
        recall_at_k=recall, mrr=mrr, guardrail=guardrail, faithfulness=faithfulness,
    )


def _empty_report(**overrides) -> EvalReport:
    base = dict(
        run_id="r", timestamp="t", num_samples=1, pass_rate=1.0, groundedness_rate=1.0,
        avg_recall_at_k=None, avg_mrr=None, avg_faithfulness=None,
        avg_answer_relevancy=None, avg_context_precision=None, avg_context_recall=None,
    )
    base.update(overrides)
    return EvalReport(**base)


# ── _load_samples ────────────────────────────────────────────────────────────

def test_load_samples_parses_jsonl_and_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_chunk_ids": ["a", "b"], "ground_truth": "GT1", "tags": ["t1"]}\n'
        "\n"
        '{"question": "Q2"}\n'
    )

    samples = _load_samples(path)

    assert samples == [
        EvalSample(question="Q1", expected_chunk_ids={"a", "b"}, ground_truth="GT1", tags=["t1"]),
        EvalSample(question="Q2", expected_chunk_ids=set(), ground_truth=None, tags=[]),
    ]


# ── _average ──────────────────────────────────────────────────────────────

def test_average_ignores_none_values() -> None:
    assert _average([0.5, None, 0.7]) == pytest.approx(0.6)


def test_average_all_none_returns_none() -> None:
    assert _average([None, None]) is None


def test_average_empty_returns_none() -> None:
    assert _average([]) is None


# ── _build_report ────────────────────────────────────────────────────────

def test_build_report_aggregates_pass_rate_and_groundedness() -> None:
    results = [_result(passed=True), _result(passed=False), _result(passed=True)]

    report = _build_report("run1", results)

    assert report.num_samples == 3
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.groundedness_rate == pytest.approx(2 / 3)


def test_build_report_averages_present_metrics_only() -> None:
    results = [_result(faithfulness=0.8), _result(faithfulness=None), _result(faithfulness=0.6)]

    report = _build_report("run1", results)

    assert report.avg_faithfulness == pytest.approx(0.7)


def test_build_report_low_faithfulness_fails_even_if_guardrail_passed() -> None:
    results = [_result(passed=True, faithfulness=0.4)]

    report = _build_report("run1", results)

    assert report.pass_rate == 0.0


# ── _check_regression ────────────────────────────────────────────────────

def test_check_regression_flags_drop_beyond_threshold() -> None:
    report = _empty_report(groundedness_rate=0.5, avg_faithfulness=0.5)
    history = [{"groundedness_rate": 0.9, "avg_faithfulness": 0.9}]

    _check_regression(report, history, threshold=0.05)

    assert report.regression_detected is True
    assert len(report.regression_notes) == 2


def test_check_regression_no_flag_within_threshold() -> None:
    report = _empty_report(groundedness_rate=0.88, avg_faithfulness=0.87)
    history = [{"groundedness_rate": 0.9, "avg_faithfulness": 0.9}]

    _check_regression(report, history, threshold=0.05)

    assert report.regression_detected is False


def test_check_regression_empty_history_never_flags() -> None:
    report = _empty_report(groundedness_rate=0.1)

    _check_regression(report, [], threshold=0.05)

    assert report.regression_detected is False


def test_check_regression_skips_metrics_missing_from_either_side() -> None:
    # avg_faithfulness is None on this run (RAGAS wasn't used) -- shouldn't
    # be compared even if the previous run had a value.
    report = _empty_report(groundedness_rate=0.9, avg_faithfulness=None)
    history = [{"groundedness_rate": 0.9, "avg_faithfulness": 0.9}]

    _check_regression(report, history, threshold=0.05)

    assert report.regression_detected is False


# ── history persistence ──────────────────────────────────────────────────

def test_save_and_load_history_round_trips(tmp_path) -> None:
    path = tmp_path / "history.json"
    report = _build_report("run1", [_result(passed=True)])

    _save_history(path, report)
    history = _load_history(path)

    assert len(history) == 1
    assert history[0]["run_id"] == "run1"
    assert "results" not in history[0]


def test_save_history_keeps_only_last_n_entries(tmp_path, monkeypatch) -> None:
    path = tmp_path / "history.json"
    monkeypatch.setattr(evaluate_module, "_HISTORY_KEEP", 2)

    for i in range(3):
        _save_history(path, _build_report(f"run{i}", [_result(passed=True)]))

    history = _load_history(path)
    assert [h["run_id"] for h in history] == ["run1", "run2"]


def test_load_history_missing_file_returns_empty(tmp_path) -> None:
    assert _load_history(tmp_path / "nope.json") == []


def test_load_history_corrupt_file_returns_empty_not_raise(tmp_path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not valid json{{{")

    assert _load_history(path) == []


# ── run_eval end-to-end (run_generate mocked) ────────────────────────────

def test_run_eval_end_to_end_with_mocked_pipeline(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        json.dumps(
            {"question": "What was revenue?", "expected_chunk_ids": ["c1"], "ground_truth": "Revenue was $100."}
        )
        + "\n"
    )

    chunk = Chunk(
        id="c1", doc_id="doc1", text="Revenue was $100.", chunk_type="text",
        page_start=1, page_end=1, source_parser="test",
    )
    fake_report = AnswerReport(
        answer=GenerationResult(text="Revenue was $100.", citations=["doc1 p.1"]),
        guardrail=GuardrailResult(passed=True, grounding_ratio=1.0),
        query_intent="factual",
        generator_used="fake",
        retrieved_chunks=[ScoredChunk(chunk=chunk, score=1.0, source="rerank")],
    )
    monkeypatch.setattr(evaluate_module, "run_generate", lambda *a, **kw: fake_report)

    reset_config()
    cfg = get_config()
    history_path = tmp_path / "history.json"
    monkeypatch.setattr(cfg.eval, "history_path", str(history_path))

    report = run_eval(dataset_path=str(dataset_path), use_ragas=False, top_k=5)

    assert report.num_samples == 1
    assert report.avg_recall_at_k == 1.0
    assert report.avg_mrr == 1.0
    assert report.groundedness_rate == 1.0
    assert report.avg_faithfulness is None  # use_ragas=False -- never scored
    assert history_path.exists()

    reset_config()

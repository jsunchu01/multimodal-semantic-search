"""Tests RagasEvaluator's degrade path and scoring/aggregation logic. Fake
scorer classes stand in for ragas.metrics.collections.{Faithfulness,...},
assigned directly to the private _Faithfulness/_AnswerRelevancy/
_ContextPrecision/_ContextRecall attributes (same pattern as the reranker's
._model injection) -- ragas itself never has to be installed for these to
run, and no real network/API call is made.
"""

from mmss.components.evaluator.ragas_evaluator import RagasEvaluator


class _FakeScore:
    def __init__(self, value: float) -> None:
        self.value = value


def _fake_scorer_class(value: float | None = None, raises: bool = False):
    class _Scorer:
        def __init__(self, **kwargs) -> None:
            pass

        async def ascore(self, **kwargs) -> _FakeScore:
            if raises:
                raise RuntimeError("judge call failed")
            return _FakeScore(value)

    return _Scorer


def _evaluator_with_fakes(
    faithfulness: float | None = 0.9,
    answer_relevancy: float | None = 0.8,
    context_precision: float | None = 0.7,
    context_recall: float | None = 0.6,
    faithfulness_raises: bool = False,
) -> RagasEvaluator:
    evaluator = RagasEvaluator()
    evaluator._llm = object()  # bypasses _ensure_clients' real import entirely
    evaluator._embeddings = object()
    evaluator._Faithfulness = _fake_scorer_class(faithfulness, raises=faithfulness_raises)
    evaluator._AnswerRelevancy = _fake_scorer_class(answer_relevancy)
    evaluator._ContextPrecision = _fake_scorer_class(context_precision)
    evaluator._ContextRecall = _fake_scorer_class(context_recall)
    return evaluator


def test_score_returns_all_none_when_ragas_not_installed() -> None:
    evaluator = RagasEvaluator()
    evaluator._ensure_clients = lambda: False

    scores = evaluator.score("q", "a", ["ctx"], None)

    assert scores == {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }


def test_score_without_ground_truth_skips_context_metrics() -> None:
    evaluator = _evaluator_with_fakes()

    scores = evaluator.score("q", "a", ["ctx"], None)

    assert scores["faithfulness"] == 0.9
    assert scores["answer_relevancy"] == 0.8
    assert scores["context_precision"] is None
    assert scores["context_recall"] is None


def test_score_with_ground_truth_runs_all_four_metrics() -> None:
    evaluator = _evaluator_with_fakes()

    scores = evaluator.score("q", "a", ["ctx"], "reference answer")

    assert scores == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
    }


def test_one_failing_metric_does_not_crash_the_others() -> None:
    evaluator = _evaluator_with_fakes(faithfulness_raises=True)

    scores = evaluator.score("q", "a", ["ctx"], "reference answer")

    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] == 0.8
    assert scores["context_precision"] == 0.7
    assert scores["context_recall"] == 0.6


def test_ensure_clients_short_circuits_when_llm_already_set() -> None:
    # If _llm is already populated (as tests do above), _ensure_clients must
    # not attempt the real `import ragas...` at all.
    evaluator = RagasEvaluator()
    evaluator._llm = object()

    assert evaluator._ensure_clients() is True

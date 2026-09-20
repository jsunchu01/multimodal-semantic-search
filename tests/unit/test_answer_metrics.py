"""Tests groundedness_rate -- fraction of GuardrailResults that passed."""

from mmss.components.evaluator.answer_metrics import groundedness_rate
from mmss.components.guardrails.base import GuardrailResult


def test_groundedness_rate_all_passed() -> None:
    results = [GuardrailResult(passed=True, grounding_ratio=1.0) for _ in range(3)]
    assert groundedness_rate(results) == 1.0


def test_groundedness_rate_partial() -> None:
    results = [
        GuardrailResult(passed=True, grounding_ratio=1.0),
        GuardrailResult(passed=False, grounding_ratio=0.5),
        GuardrailResult(passed=True, grounding_ratio=0.9),
        GuardrailResult(passed=False, grounding_ratio=0.3),
    ]
    assert groundedness_rate(results) == 0.5


def test_groundedness_rate_empty_is_vacuously_satisfied() -> None:
    assert groundedness_rate([]) == 1.0

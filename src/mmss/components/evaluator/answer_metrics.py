"""Answer quality metrics: groundedness rate (fraction of answers passing the
numeric guardrail)."""

from __future__ import annotations

from mmss.components.guardrails.base import GuardrailResult


def groundedness_rate(results: list[GuardrailResult]) -> float:
    if not results:
        return 1.0
    return sum(1 for r in results if r.passed) / len(results)

"""Shared value types for the eval harness: one labeled query (EvalSample),
one sample's scored outcome (EvalResult), and a full run's aggregate
(EvalReport).

Trimmed from the reference repo's actual dataclasses (verified from its real
source): dropped `cost_usd` (nothing in mmss tracks token/API cost) and
`citation_faithfulness` (never a defined metric in their code either, just an
unused field). Kept `passed`/`pass_rate` as a simple, useful top-line signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mmss.components.guardrails.base import GuardrailResult


@dataclass
class EvalSample:
    question: str
    expected_chunk_ids: set[str] = field(default_factory=set)
    ground_truth: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    question: str
    answer_text: str
    ground_truth: str | None
    retrieved_chunk_ids: list[str]
    # Deterministic retrieval metrics -- None when the sample has no
    # expected_chunk_ids (nothing to score against, not "scored zero").
    recall_at_k: float | None = None
    mrr: float | None = None
    # The real GuardrailResult from this sample's run_generate() call (not
    # reconstructed after the fact) -- carries grounding_ratio and
    # ungrounded_claims too, not just pass/fail.
    guardrail: GuardrailResult | None = None
    # RAGAS LLM-judge metrics -- None when RAGAS wasn't run, isn't
    # installed, or the judge call failed for this sample; context_precision/
    # context_recall are also None whenever ground_truth is unset, since both
    # require a reference answer.
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    latency_ms: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.guardrail is None or not self.guardrail.passed:
            return False
        if self.faithfulness is not None and self.faithfulness < 0.7:
            return False
        return True


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    num_samples: int
    pass_rate: float
    groundedness_rate: float
    avg_recall_at_k: float | None
    avg_mrr: float | None
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_context_precision: float | None
    avg_context_recall: float | None
    results: list[EvalResult] = field(default_factory=list)
    regression_detected: bool = False
    regression_notes: list[str] = field(default_factory=list)

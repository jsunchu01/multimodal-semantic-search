"""Guardrail interface: checks a generated answer against its source chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mmss.components.base import GenerationResult, ScoredChunk


@dataclass
class GuardrailResult:
    passed: bool
    grounding_ratio: float
    ungrounded_claims: list[str] = field(default_factory=list)


class Guardrail(ABC):
    @abstractmethod
    def check(
        self, answer: GenerationResult, source_chunks: list[ScoredChunk]
    ) -> GuardrailResult: ...

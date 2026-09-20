"""Generator interface: prompt -> answer text (+ citations)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmss.components.base import GenerationResult


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

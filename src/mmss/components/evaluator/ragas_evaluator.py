"""Wraps RAGAS to score one generated answer on faithfulness, answer
relevancy, context precision, and context recall.

Verified against ragas's own current docs (docs.ragas.io, package version
0.4.3 on PyPI as of writing) -- NOT the reference repo's actual usage, which
targeted an older ragas release (`evaluate(dataset, metrics=[faithfulness,
answer_relevancy, context_precision])` over a HuggingFace Dataset, with bare
lowercase metric singletons imported from `ragas.metrics`). That pattern no
longer matches ragas's current API, so this is a fresh implementation against
the current one rather than a port:

  - Metrics are classes under `ragas.metrics.collections`: Faithfulness,
    AnswerRelevancy, ContextPrecision, ContextRecall.
  - Each needs an `llm=` (and AnswerRelevancy also `embeddings=`), built via
    `ragas.llms.llm_factory` / `ragas.embeddings.embedding_factory`, both of
    which wrap a plain `openai.AsyncOpenAI()` client directly -- no
    LangChain dependency needed for this.
  - Scoring is `await scorer.ascore(**fields) -> result` where `result.value`
    is the 0..1 float. Faithfulness needs (user_input, response,
    retrieved_contexts); AnswerRelevancy needs (user_input, response);
    ContextPrecision/ContextRecall need (user_input, retrieved_contexts,
    reference) -- `reference` is our EvalSample.ground_truth, so those two
    are skipped whenever a sample has none.

ascore() is async (ragas is built on AsyncOpenAI). Rather than adopting async
project-wide -- deliberately declined at M7 for everywhere else in mmss --
one asyncio.run() per sample gathers all four calls concurrently; eval runs
are small batches, not a hot path, and nothing here is called from inside an
already-running event loop (the whole CLI is synchronous end to end).

Graceful degrade, same shape as LocalCrossEncoderReranker/PIIRedactor: if
`ragas` isn't installed, or a judge call fails for a given sample, the
affected metric(s) come back as None (never a fabricated placeholder score)
so the rest of the eval run -- deterministic metrics, the guardrail pass
rate -- is unaffected.
"""

from __future__ import annotations

import asyncio

from mmss.utils.logging import get_logger

logger = get_logger(__name__)

_EMPTY_SCORES: dict[str, float | None] = {
    "faithfulness": None,
    "answer_relevancy": None,
    "context_precision": None,
    "context_recall": None,
}


class RagasEvaluator:
    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self._judge_model = judge_model
        self._llm = None
        self._embeddings = None
        self._Faithfulness = None
        self._AnswerRelevancy = None
        self._ContextPrecision = None
        self._ContextRecall = None

    def _ensure_clients(self) -> bool:
        """Lazily build the ragas-wrapped LLM/embeddings clients and pull in
        the metric classes. Returns False (leaving everything None) if
        `ragas` isn't installed, without raising."""
        if self._llm is not None:
            return True
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings import embedding_factory
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError:
            logger.warning(
                "ragas not installed -- RAGAS metrics disabled for this run "
                "(pip install -e '.[eval]')"
            )
            return False

        client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        self._llm = llm_factory(self._judge_model, client=client)
        self._embeddings = embedding_factory("openai", client=client)
        self._Faithfulness = Faithfulness
        self._AnswerRelevancy = AnswerRelevancy
        self._ContextPrecision = ContextPrecision
        self._ContextRecall = ContextRecall
        return True

    def score(
        self,
        question: str,
        answer_text: str,
        retrieved_contexts: list[str],
        ground_truth: str | None,
    ) -> dict[str, float | None]:
        if not self._ensure_clients():
            return dict(_EMPTY_SCORES)
        try:
            return asyncio.run(
                self._score_async(question, answer_text, retrieved_contexts, ground_truth)
            )
        except Exception:
            logger.warning("RAGAS scoring failed for this sample", exc_info=True)
            return dict(_EMPTY_SCORES)

    async def _score_async(
        self,
        question: str,
        answer_text: str,
        retrieved_contexts: list[str],
        ground_truth: str | None,
    ) -> dict[str, float | None]:
        tasks: dict[str, object] = {
            "faithfulness": self._Faithfulness(llm=self._llm).ascore(
                user_input=question, response=answer_text, retrieved_contexts=retrieved_contexts
            ),
            "answer_relevancy": self._AnswerRelevancy(llm=self._llm, embeddings=self._embeddings).ascore(
                user_input=question, response=answer_text
            ),
        }
        if ground_truth:
            tasks["context_precision"] = self._ContextPrecision(llm=self._llm).ascore(
                user_input=question, reference=ground_truth, retrieved_contexts=retrieved_contexts
            )
            tasks["context_recall"] = self._ContextRecall(llm=self._llm).ascore(
                user_input=question, retrieved_contexts=retrieved_contexts, reference=ground_truth
            )

        keys = list(tasks.keys())
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)

        scores = dict(_EMPTY_SCORES)
        for key, outcome in zip(keys, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.warning("RAGAS metric %r failed: %s", key, outcome)
                continue
            scores[key] = outcome.value
        return scores

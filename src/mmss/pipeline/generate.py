"""Generation pipeline: analyze query -> retrieve -> build a cited prompt ->
generate (routed to a complex-query generator if configured) -> [PoT
verification for numeric queries] -> run guardrails (numeric grounding, then
PII/financial-identifier redaction) -> return a report the CLI can print.

M8 adds query analysis (QueryAnalyzer) driving three things: retrieval
breadth (suggested_top_k), generator routing (is_complex -> generator_complex
if configured), and whether to invoke the PoT executor for arithmetic
(use_pot). Ordering matters for two real interaction risks, both handled
below: the PoT code block is extracted and stripped out of the answer text
*before* the numeric-grounding guardrail runs (so the guardrail never scans
-- and potentially mangles via the "strip" policy -- numeric literals that
are part of the code, not a claim), and the verified-computation footer is
appended *after* both the guardrail and PII redaction (so a freshly-computed
derived number, e.g. a growth rate that doesn't appear verbatim in any
source chunk, never gets flagged as "ungrounded").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from mmss.components.agentic.pot_executor import PoTExecutor, PoTResult
from mmss.components.agentic.query_analyzer import QueryAnalyzer
from mmss.components.base import GenerationResult, ScoredChunk
from mmss.components.generator import get_generator
from mmss.components.guardrails.base import GuardrailResult
from mmss.components.guardrails.numeric_grounding import NumericGroundingGuardrail
from mmss.components.guardrails.pii_redactor import PIIRedactor
from mmss.config import get_config
from mmss.pipeline.query import run_query
from mmss.registry import build

# Exposed as a named constant (not just inline in the prompt string below) so
# callers -- e.g. the Streamlit UI -- can detect a "not found" answer and
# suppress source/score display for it, without duplicating this exact
# string and risking it drifting out of sync with what the LLM is actually
# instructed to say.
NOT_FOUND_PHRASE = "This information is not available in the provided documents."

# Genericized from an earlier finance-only version ("expert financial analyst
# assistant") once this project started being used for research papers too --
# the strict source-grounding discipline (rules 1-5) is what actually matters
# and isn't domain-specific; only the framing/vocabulary needed to change.
RAG_SYSTEM_PROMPT = f"""\
You are a rigorous research and analysis assistant. Answer the user's question
STRICTLY based on the provided source passages. Follow these rules:

1. ONLY use information explicitly present in the source passages.
2. If the answer is not in the sources, say "{NOT_FOUND_PHRASE}"
3. For quantitative claims (numbers, statistics, results), ALWAYS include the exact figure from the source -- do not convert units or round.
4. ALWAYS cite your sources with [Source N] after each claim, matching the source numbers given below.
5. NEVER extrapolate, estimate, or invent numbers.
"""

_POT_PROMPT_ADDENDUM = """
Additionally, since this question requires a calculation: after your prose answer, include a
fenced Python code block that computes the final numeric answer using ONLY the exact figures
from the source passages above. Assign the final computed value to a variable named `result`.
Example:
```python
v_old = 41831
v_new = 50623
result = ((v_new - v_old) / v_old) * 100
```
"""

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n.*?```", re.DOTALL)


@dataclass
class AnswerReport:
    answer: GenerationResult
    guardrail: GuardrailResult
    pii_entities_found: list[str] = field(default_factory=list)
    query_intent: str = ""
    generator_used: str = ""
    pot_result: PoTResult | None = None
    # The chunks actually fed to the generator -- exposed for the eval
    # harness (recall@k/MRR need chunk ids, RAGAS needs the chunk text as
    # retrieved_contexts). Not printed by `mmss ask`; citations above already
    # cover that CLI's needs.
    retrieved_chunks: list[ScoredChunk] = field(default_factory=list)


def _build_context_block(chunks: list[ScoredChunk]) -> str:
    parts = []
    for i, scored in enumerate(chunks, 1):
        c = scored.chunk
        parts.append(f"[Source {i}: {c.doc_id}, Page {c.page_start}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _select_generator(is_complex: bool):
    cfg = get_config()
    if is_complex and cfg.generator_complex is not None:
        return build("generator", cfg.generator_complex.provider, **cfg.generator_complex.params)
    return get_generator()


def run_generate(
    question: str,
    top_k: int | None = None,
    strategy: Literal["dense", "hybrid", "reranked"] = "reranked",
) -> AnswerReport:
    analysis = QueryAnalyzer().analyze(question)
    resolved_top_k = top_k if top_k is not None else analysis.suggested_top_k

    chunks = run_query(question, top_k=resolved_top_k, strategy=strategy)

    context_block = _build_context_block(chunks)
    pot_cfg = get_config().pot
    use_pot = analysis.use_pot and pot_cfg.enabled
    system_prompt = RAG_SYSTEM_PROMPT + (_POT_PROMPT_ADDENDUM if use_pot else "")
    prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    generator = _select_generator(analysis.is_complex)
    result = generator.generate(prompt, system=system_prompt)
    result.citations = [f"{c.chunk.doc_id} p.{c.chunk.page_start}" for c in chunks]

    pot_result: PoTResult | None = None
    if use_pot:
        pot_result = PoTExecutor(timeout_seconds=pot_cfg.timeout_seconds).execute_from_llm_response(
            result.text
        )
        # Strip the code block regardless of success -- raw Python isn't a
        # useful thing to show the user, and it must not reach the
        # numeric-grounding guardrail below (see module docstring).
        result.text = _CODE_FENCE_RE.sub("", result.text, count=1).strip()

    guardrail_cfg = get_config().guardrail
    guardrail = NumericGroundingGuardrail(
        numeric_tolerance=guardrail_cfg.numeric_tolerance,
        min_grounding_ratio=guardrail_cfg.min_grounding_ratio,
    )
    guardrail_result = guardrail.check(result, chunks)

    if not guardrail_result.passed:
        if guardrail_cfg.on_violation == "reject":
            result.text = (
                "This answer could not be verified against the source documents "
                "and has been withheld."
            )
        elif guardrail_cfg.on_violation == "strip":
            for span in guardrail_result.ungrounded_claims:
                result.text = result.text.replace(span, "[unverified]")
        # "flag": leave result.text as-is -- the caller reads guardrail_result directly.

    pii_cfg = get_config().pii
    pii_found: list[str] = []
    if pii_cfg.enabled:
        redactor = PIIRedactor(
            pii_entities=pii_cfg.entities,
            enable_financial_patterns=pii_cfg.enable_financial_patterns,
        )
        result.text, pii_found = redactor.redact(result.text)

    if pot_result is not None and pot_result.success:
        result.text = f"{result.text}\n\n**Verified calculation**: {pot_result.formatted()}"

    return AnswerReport(
        answer=result,
        guardrail=guardrail_result,
        pii_entities_found=pii_found,
        query_intent=analysis.intent.value,
        generator_used=generator.name,
        pot_result=pot_result,
        retrieved_chunks=chunks,
    )

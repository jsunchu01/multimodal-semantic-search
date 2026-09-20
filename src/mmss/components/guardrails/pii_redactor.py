r"""PII + financial-identifier redaction, applied to the generated answer text
(not the retrieved context) -- redacting context before generation risks
removing figures the numeric-grounding guardrail and the generator both need,
and this project's source documents are public SEC filings with no real
customer PII in them to begin with; redacting the answer is the safer, still-
useful integration point if a future document set does carry real PII.

Wraps Microsoft Presidio when installed (pip install presidio-analyzer
presidio-anonymizer, plus `python -m spacy download en_core_web_lg` -- a
separate model download Presidio's own install docs require, run it
yourself since there's no CLI execution here), falling back to regex-only
redaction otherwise.

Verified against Presidio's actual docs (not just the reference repo's
usage, which happens to match): AnalyzerEngine().analyze(text=..., entities=
[...], language=...) returns a list of RecognizerResult (entity_type/score/
start/end); AnonymizerEngine().anonymize(text=..., analyzer_results=...)
returns an EngineResult whose .text holds the redacted string.

Two deliberate changes from the reference repo:

1. _try_init_presidio() there only catches ImportError. A missing spaCy
   model (a separate, easy-to-forget install step) raises at
   AnalyzerEngine() construction, not at import time -- so a narrow
   ImportError-only catch would let that specific failure crash the whole
   answer pipeline instead of degrading to the regex fallback. Same category
   of fix as FallbackParser in components/parser/__init__.py.

2. The TICKER pattern's leading \b (matching "$AAPL") doesn't actually work:
   \b only matches at a transition between a word character and a non-word
   character, and both whitespace and "$" are non-word, so \b\$ never
   matches after a space or at the start of a sentence -- only when "$" is
   glued directly to a preceding letter/digit, which isn't how ticker
   mentions are normally written. Replaced with a negative lookbehind for a
   word character so it matches the realistic case.
"""

from __future__ import annotations

import re

from mmss.utils.logging import get_logger

logger = get_logger(__name__)

# Financial-specific regex patterns, always applied regardless of Presidio.
_FINANCIAL_PATTERNS: dict[str, re.Pattern] = {
    "CUSIP": re.compile(r"\b[0-9A-Z]{9}\b"),
    "ISIN": re.compile(r"\b[A-Z]{2}[0-9A-Z]{10}\b"),
    "BANK_ACCOUNT_US": re.compile(r"\b\d{8,17}\b"),
    "ROUTING_NUMBER": re.compile(r"\b\d{9}\b"),
    # $AAPL style. A leading \b won't fire directly before "$", since "$" is
    # not a word character -- \b only matches at a transition between a word
    # and a non-word character, and whitespace/punctuation before "$" is
    # also non-word, so \b\$ never matches after a space in practice. Using
    # a negative lookbehind for a word character instead, so this actually
    # matches "$AAPL" preceded by whitespace, punctuation, or string start.
    "TICKER": re.compile(r"(?<!\w)\$[A-Z]{1,5}\b"),
}

# Core PII regex patterns, used only when Presidio is unavailable.
_PII_FALLBACK_PATTERNS: dict[str, re.Pattern] = {
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_NUMBER": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{1,3}\b"),
}


def _redact_with_regex(text: str, entity_map: dict[str, re.Pattern]) -> tuple[str, list[str]]:
    found: list[str] = []
    for entity_type, pattern in entity_map.items():
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub(f"<{entity_type}>", text)
            found.extend([entity_type] * len(matches))
    return text, found


class PIIRedactor:
    """Detects and redacts PII + financial identifiers from text.

    Uses Microsoft Presidio when available; falls back to regex-only mode.
    """

    def __init__(
        self,
        pii_entities: list[str] | None = None,
        enable_financial_patterns: bool = True,
        language: str = "en",
    ) -> None:
        self._language = language
        self._enable_financial = enable_financial_patterns
        self._pii_entities = pii_entities or [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_BANK_NUMBER",
        ]
        self._presidio_available = False
        self._analyzer = None
        self._anonymizer = None
        self._try_init_presidio()

    def _try_init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
        except Exception:
            logger.warning(
                "Presidio unavailable (not installed, or its spaCy model "
                "wasn't downloaded) -- falling back to regex-only PII "
                "redaction. Install with: pip install presidio-analyzer "
                "presidio-anonymizer && python -m spacy download en_core_web_lg"
            )

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact PII from text. Returns (redacted_text, entity_types_found)."""
        found_entities: list[str] = []

        if self._presidio_available and self._analyzer:
            try:
                results = self._analyzer.analyze(
                    text=text, entities=self._pii_entities, language=self._language
                )
                if results:
                    anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
                    text = anonymized.text
                    found_entities = [r.entity_type for r in results]
            except Exception:
                logger.warning("Presidio redaction failed at call time -- text left unredacted for this pass")
        else:
            text, pii_found = _redact_with_regex(text, _PII_FALLBACK_PATTERNS)
            found_entities.extend(pii_found)

        if self._enable_financial:
            text, fin_found = _redact_with_regex(text, _FINANCIAL_PATTERNS)
            found_entities.extend(fin_found)

        return text, found_entities

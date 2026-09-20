"""Tests PIIRedactor's regex-fallback path deterministically -- presidio isn't
installed by default (it's the optional [guardrails] extra), and even when it
is, forcing _presidio_available=False keeps this test independent of whether
Presidio + its spaCy model happen to be present in the environment.
"""

from mmss.components.guardrails.pii_redactor import PIIRedactor


def test_regex_fallback_redacts_ssn_email_phone() -> None:
    redactor = PIIRedactor()
    redactor._presidio_available = False

    text, found = redactor.redact(
        "Contact John at john@example.com or 555-123-4567, SSN 123-45-6789."
    )

    assert "<EMAIL_ADDRESS>" in text
    assert "<PHONE_NUMBER>" in text
    assert "<US_SSN>" in text
    assert set(found) >= {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"}


def test_ticker_redacted_when_preceded_by_whitespace() -> None:
    # Regression test for the leading-\b bug found in the reference repo's
    # equivalent pattern: \b\$ never matches after a space, since both are
    # non-word characters, so realistic input like "the ticker $AAPL" needs
    # the negative-lookbehind version to actually match.
    redactor = PIIRedactor()
    redactor._presidio_available = False

    text, found = redactor.redact("The ticker $AAPL is up today.")

    assert "<TICKER>" in text
    assert "TICKER" in found


def test_financial_patterns_can_be_disabled() -> None:
    redactor = PIIRedactor(enable_financial_patterns=False)
    redactor._presidio_available = False

    text, found = redactor.redact("The ticker $AAPL is up today.")

    assert "$AAPL" in text
    assert found == []


def test_no_pii_returns_text_unchanged() -> None:
    redactor = PIIRedactor()
    redactor._presidio_available = False

    text, found = redactor.redact("Total revenues increased 24% year over year.")

    assert text == "Total revenues increased 24% year over year."
    assert found == []

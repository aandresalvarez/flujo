import pytest
from pydantic import SecretStr

from flujo.utils.redact import summarize_and_redact_prompt
from flujo.infra.settings import Settings


@pytest.mark.fast
def test_telemetry_pii_redaction() -> None:
    """Redaction helper should mask API keys in telemetry span payloads."""

    settings = Settings(openai_api_key=SecretStr("sk-12345"))

    redacted = summarize_and_redact_prompt(
        "My secret is sk-12345", max_length=200, settings=settings
    )
    assert "[REDACTED]" in redacted
    assert "sk-12345" not in redacted

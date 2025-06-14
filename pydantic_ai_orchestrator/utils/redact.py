"""Utilities for redacting sensitive information."""

import re


def redact_string(text: str, secret: str) -> str:
    """Replaces occurrences of a secret string or its prefix with a redacted placeholder."""
    if not text:
        return text
    if secret:
        # Redact both the full secret and any string starting with the first 8 chars and plausible suffix
        pattern = re.escape(secret[:8]) + r"[A-Za-z0-9_-]{5,}"  # e.g. sk-XXXX...
        text = re.sub(pattern, "[REDACTED]", text)
        text = text.replace(secret, "[REDACTED]")
    return text


def redact_url_password(url: str) -> str:
    """Redacts the password from a URL."""
    return re.sub(r"://[^@]+@", "://[REDACTED]@", url)


def summarize_and_redact_prompt(prompt_text: str, max_length: int = 200) -> str:
    """Return a truncated and redacted version of a prompt."""
    if not prompt_text:
        return ""

    from pydantic_ai_orchestrator.infra.settings import settings

    text = prompt_text
    for secret in (
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
        settings.google_api_key.get_secret_value() if settings.google_api_key else None,
        settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None,
    ):
        if secret:
            text = redact_string(text, secret)

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text

"""Utilities for redacting sensitive information."""

import re
from flujo.infra.settings import settings as global_settings, Settings
from typing import Optional


def redact_string(text: str, secret: Optional[str]) -> str:
    """Replaces occurrences of a secret string or its prefix with a redacted placeholder."""
    if not text:
        return text
    if secret:
        # First, replace the full secret
        text = text.replace(secret, "[REDACTED]")
        # Then, redact any string starting with the first 8 chars and at least 5 more plausible chars
        pattern = re.escape(secret[:8]) + r"[A-Za-z0-9_-]{5,}"
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def redact_url_password(url: str) -> str:
    """Redacts the password from a URL."""
    return re.sub(r"://[^@]+@", "://[REDACTED]@", url)


def summarize_and_redact_prompt(
    prompt_text: str, max_length: int = 200, settings: Optional[Settings] = None
) -> str:
    """Return a truncated and redacted version of a prompt."""
    if not prompt_text:
        return ""

    if settings is None:
        settings = global_settings

    text = prompt_text
    for secret in (
        settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
        settings.google_api_key.get_secret_value() if settings.google_api_key else None,
        settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None,
    ):
        if secret:
            text = redact_string(text, secret)

    # Generic fallback for obvious API-key patterns when explicit settings are unavailable.
    text = re.sub(r"sk-[A-Za-z0-9_-]{5,}", "[REDACTED]", text)

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text

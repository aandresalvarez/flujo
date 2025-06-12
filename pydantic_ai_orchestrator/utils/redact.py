"""Utilities for redacting sensitive information."""

import re

def redact_string(text: str, secret: str) -> str:
    """Replaces occurrences of a secret string with a redacted placeholder."""
    if not text:
        return text
    if secret:
        return text.replace(secret, "[REDACTED]")
    return text

def redact_url_password(url: str) -> str:
    """Redacts the password from a URL."""
    return re.sub(r"://[^@]+@", "://[REDACTED]@", url) 
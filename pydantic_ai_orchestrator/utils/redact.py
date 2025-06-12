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
"""Error utilities for consistent feedback formatting and error extraction."""


def format_feedback(exc: Exception) -> str:
    """Return canonical feedback string expected by legacy tests."""
    return f"Agent execution failed with {exc.__class__.__name__}: {exc}"


def extract_original_error(exc: Exception) -> str:
    """Return the innermost, human error message (no prefixes)."""
    # Walk the exception chain until we hit the first non-wrapper
    current_exc: BaseException = exc
    while current_exc.__cause__ is not None:
        current_exc = current_exc.__cause__
    return str(current_exc)

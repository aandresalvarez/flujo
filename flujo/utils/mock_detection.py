from __future__ import annotations


def is_mock_like(obj: object) -> bool:
    """Return True when an object looks like a unittest.mock instance.

    This avoids importing unittest.mock in runtime modules while still
    guarding against mock objects flowing through execution paths.
    """
    if obj is None:
        return False
    try:
        module = getattr(getattr(obj, "__class__", None), "__module__", "")
        if module == "unittest.mock":
            return True
    except Exception:
        pass
    try:
        if getattr(obj, "_is_mock", False):
            return True
    except Exception:
        pass
    try:
        if getattr(obj, "_mock_name", None) is not None:
            return True
    except Exception:
        pass
    try:
        if hasattr(obj, "assert_called") and hasattr(obj, "mock_calls"):
            return True
    except Exception:
        pass
    return False

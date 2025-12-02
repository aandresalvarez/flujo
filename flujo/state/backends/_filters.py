from __future__ import annotations

from typing import Any, Mapping


def metadata_contains(candidate: Any, subset: Mapping[str, Any]) -> bool:
    """Return True when ``candidate`` contains ``subset`` using JSON-style containment."""
    if not isinstance(candidate, Mapping):
        return False

    for key, expected in subset.items():
        if key not in candidate:
            return False
        actual = candidate[key]
        if isinstance(expected, Mapping):
            if not metadata_contains(actual, expected):
                return False
        else:
            if actual != expected:
                return False
    return True


__all__ = ["metadata_contains"]

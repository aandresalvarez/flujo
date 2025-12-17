from __future__ import annotations

from typing import Any, Final
from collections.abc import Mapping

SCRATCHPAD_REMOVED_MESSAGE: Final[str] = (
    "scratchpad field has been removed; migrate data to typed fields"
)

SCRATCHPAD_PATH_REMOVED_MESSAGE: Final[str] = (
    "scratchpad has been removed; migrate paths to import_artifacts or typed context fields."
)


def is_scratchpad_path(path: str) -> bool:
    """Return True if a dot-path targets the removed 'scratchpad' root."""
    stripped = path.strip()
    return stripped == "scratchpad" or stripped.startswith("scratchpad.")


def update_contains_scratchpad(update_data: Mapping[str, Any]) -> bool:
    """Return True if an update payload attempts to set the removed 'scratchpad' field."""
    return "scratchpad" in update_data


def is_merge_scratchpad(value: object) -> bool:
    """Return True if a merge strategy value references the removed scratchpad merge mode."""
    if isinstance(value, str):
        return value.strip().lower() == "merge_scratchpad"
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip().lower() == "merge_scratchpad":
        return True
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str) and enum_value.strip().lower() == "merge_scratchpad":
        return True
    return False

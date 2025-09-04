from __future__ import annotations

from typing import Any, Optional
import importlib

try:
    _mod = importlib.import_module("pydantic")
    PydTypeAdapter = getattr(_mod, "TypeAdapter", None)
except Exception:  # pragma: no cover - defensive for environments without pydantic v2
    PydTypeAdapter = None


def derive_json_schema_from_type(tp: Any) -> Optional[dict[str, Any]]:
    """Best-effort derivation of a JSON Schema from a Python type.

    Uses Pydantic v2 TypeAdapter when available. Returns None if schema cannot
    be derived or if type is too generic (e.g., Any).
    """
    try:
        if tp is None:
            return None
        # Heuristic: avoid Any
        if getattr(tp, "__name__", "") == "Any":
            return None
        if PydTypeAdapter is None:
            return None
        # Narrow the Optional
        assert PydTypeAdapter is not None
        ta = PydTypeAdapter(tp)
        schema = ta.json_schema()
        # Ensure object form
        if isinstance(schema, dict) and schema:
            return schema
        return None
    except Exception:
        return None

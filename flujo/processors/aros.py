"""
AROS processors: deterministic JSON extraction/unescape (Stage 0) and
schema-aware smart coercion (placeholder, filled in a later step).

These processors integrate with the existing processor pipeline and should be
auto-injected by AgentStep policy based on per-step processing config and
global [aros] defaults (see FSD.md v1.2).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from flujo.type_definitions.common import JSONObject

from .base import Processor
from ..infra.telemetry import logfire
from ..tracing.manager import get_active_trace_manager


class JsonRegionExtractorProcessor(Processor):
    """Stage 0: Extract the largest balanced JSON object/array and bounded-unescape.

    - Scans input text to find the maximal balanced {...} or [...] region, ignoring
      braces/brackets inside quoted strings.
    - If both object and array candidates exist, prefer the one matching the schema
      root when provided via `expected_root`.
    - Applies bounded unescape for double-encoded JSON (up to `max_unescape_depth`).

    This is a safe, deterministic pre-stage that often enables plain JSON decoding
    to succeed without LLM-based repair.
    """

    name: str = "JsonRegionExtractor"

    def __init__(self, *, max_unescape_depth: int = 2, expected_root: Optional[str] = None) -> None:
        self.max_unescape_depth = max_unescape_depth
        self.expected_root = expected_root  # "object" | "array" | None

    async def process(self, data: Any, context: Optional[Any] = None) -> Any:
        tm = get_active_trace_manager()
        try:
            if not isinstance(data, str):
                return data

            text = data
            obj_span = _find_largest_balanced(text, kind="object")
            arr_span = _find_largest_balanced(text, kind="array")

            chosen: Optional[tuple[int, int]] = None
            if self.expected_root == "object" and obj_span is not None:
                chosen = obj_span
            elif self.expected_root == "array" and arr_span is not None:
                chosen = arr_span
            else:
                # Prefer the longer candidate if both exist
                if obj_span and arr_span:
                    chosen = (
                        obj_span
                        if (obj_span[1] - obj_span[0]) >= (arr_span[1] - arr_span[0])
                        else arr_span
                    )
                else:
                    chosen = obj_span or arr_span

            if not chosen:
                # As a fallback, try bounded unescape on the whole text to handle double-encoded payloads
                unescaped = _bounded_json_unescape(text, max_depth=self.max_unescape_depth)
                if (
                    unescaped is not text
                    and isinstance(unescaped, str)
                    and unescaped.strip()[:1] in ("{", "[")
                ):
                    if tm is not None:
                        tm.add_event(
                            "output.coercion.attempt",
                            {
                                "stage": "extract",
                                "reason": "whole_string_unescape",
                                "expected_type": self.expected_root,
                                "actual_type": "string",
                            },
                        )
                    return unescaped
                return data

            start, end = chosen
            candidate = text[start:end]

            # Bounded unescape for double-encoded JSON strings
            candidate = _bounded_json_unescape(candidate, max_depth=self.max_unescape_depth)

            if tm is not None:
                tm.add_event(
                    "output.coercion.attempt",
                    {
                        "stage": "extract",
                        "reason": "largest_balanced_region",
                        "expected_type": self.expected_root,
                        "actual_type": "string",
                    },
                )
            return candidate
        except Exception as e:
            try:
                logfire.warning(f"JsonRegionExtractor failed: {e}")
                if tm is not None:
                    tm.add_event(
                        "output.coercion.fail",
                        {"stage": "extract", "error_preview": str(e)[:240]},
                    )
            except Exception:
                pass
            return data


def _find_largest_balanced(s: str, *, kind: str) -> Optional[tuple[int, int]]:
    """Return (start, end) slice of the largest balanced region for the given kind.

    kind = "object" → {...}
    kind = "array"  → [...]

    Ignores braces/brackets inside quoted strings with escape handling.
    Returns None if no balanced region is found.
    """
    if kind == "object":
        open_ch, close_ch = "{", "}"
    elif kind == "array":
        open_ch, close_ch = "[", "]"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    max_start = -1
    max_end = -1
    max_len = 0

    depth = 0
    start_idx = -1
    in_str = False
    esc = False
    quote = ""

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue

        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue

        if ch == open_ch:
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start_idx >= 0:
                end_idx = i + 1
                cur_len = end_idx - start_idx
                if cur_len >= max_len:
                    max_len = cur_len
                    max_start = start_idx
                    max_end = end_idx
                start_idx = -1

    if max_len > 0:
        return (max_start, max_end)
    return None


def _bounded_json_unescape(candidate: str, *, max_depth: int = 2) -> str:
    """Attempt to unescape double-encoded JSON up to max_depth.

    Strategy:
    - If candidate starts with a quote and looks like JSON-encoded JSON, try json.loads → str.
    - If the decoded string starts with '{' or '[', accept and iterate (bounded by max_depth).
    - Otherwise, return the original candidate.
    """
    try:
        import json

        depth = 0
        out = candidate
        while depth < max_depth and len(out) > 1 and out[0] in ('"', "'") and out[-1] == out[0]:
            try:
                decoded = json.loads(out)
                if (
                    isinstance(decoded, str)
                    and decoded[:1] in ("{", "[")
                    and decoded[-1:] in ("}", "]")
                ):
                    out = decoded
                    depth += 1
                else:
                    break
            except Exception:
                break
        return out
    except Exception:
        return candidate


def _coerce_recursive(data: Any, allow: dict[str, list[str]]) -> tuple[Any, set[str]]:
    """Recursively apply safe, whitelisted coercions and return (new_data, transforms).

    This implementation does not require a JSON Schema; it applies conservative
    conversions based on allowlists per JSON type. It is intentionally minimal
    to avoid over-coercion. Transforms is a set of string markers like
    'str->int', 'str->float', 'str->bool', 'str->array'.
    """
    transforms: set[str] = set()

    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            nv, tf = _coerce_recursive(v, allow)
            if tf:
                transforms |= tf
            out[k] = nv
        return out, transforms
    if isinstance(data, list):
        res: list[Any] = []
        for v in data:
            nv, tf = _coerce_recursive(v, allow)
            if tf:
                transforms |= tf
            res.append(nv)
        return res, transforms

    # Scalars: try whitelisted conversions
    if isinstance(data, str):
        s = data.strip()
        # boolean
        if "str->bool" in (allow.get("boolean") or []) and s.lower() in {"true", "false", "0", "1"}:
            transforms.add("str->bool")
            return (s.lower() in {"true", "1"}), transforms
        # integer
        if "str->int" in (allow.get("integer") or []) and _looks_like_int(s):
            try:
                iv = int(s, 10)
                transforms.add("str->int")
                return iv, transforms
            except Exception:
                pass
        # number
        if "str->float" in (allow.get("number") or []) and _looks_like_float(s):
            try:
                fv = float(s)
                transforms.add("str->float")
                return fv, transforms
            except Exception:
                pass
        # array from JSON string
        if "str->array" in (allow.get("array") or []) and s[:1] == "[" and s[-1:] == "]":
            try:
                import json as _json

                arr = _json.loads(s)
                if isinstance(arr, list):
                    transforms.add("str->array")
                    return arr, transforms
            except Exception:
                pass
    return data, transforms


def _looks_like_int(s: str) -> bool:
    if not s:
        return False
    if s[0] in "+-":
        return s[1:].isdigit()
    return s.isdigit()


def _looks_like_float(s: str) -> bool:
    # Accept standard float forms, with optional sign and decimal/exponent
    import re as _re

    return bool(_re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", s))


def _schema_root_type(schema: dict[str, Any]) -> Optional[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return t
    return None


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_against_schema(value: Any, schema: dict[str, Any]) -> bool:
    # Simple validator supporting: type, enum, properties(required subset), items, anyOf, oneOf
    try:
        # enum check (strict equality)
        if "enum" in schema and isinstance(schema["enum"], list):
            if value in schema["enum"]:
                return True
            # Do not coerce here; validator is strict
            return False
        if "anyOf" in schema and isinstance(schema["anyOf"], list):
            return any(_validate_against_schema(value, s) for s in schema["anyOf"])
        if "oneOf" in schema and isinstance(schema["oneOf"], list):
            matches = sum(1 for s in schema["oneOf"] if _validate_against_schema(value, s))
            return matches == 1 if matches > 0 else False
        t = schema.get("type")
        if isinstance(t, str):
            if t == "object":
                if not isinstance(value, dict):
                    return False
                props = schema.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                req = schema.get("required") or []
                for r in req:
                    if r not in value:
                        return False
                # Validate provided keys against their schemas if present
                for k, v in value.items():
                    if k in props and isinstance(props[k], dict):
                        if not _validate_against_schema(v, props[k]):
                            return False
                return True
            if t == "array":
                if not isinstance(value, list):
                    return False
                items = schema.get("items")
                if isinstance(items, dict):
                    return all(_validate_against_schema(v, items) for v in value)
                return True
            # primitives
            return _json_type(value) == t or (
                t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        return True
    except Exception:
        return False


def _coerce_with_schema(
    value: Any,
    schema: dict[str, Any],
    allow: dict[str, List[str]],
    *,
    path: Tuple[str, ...],
) -> Tuple[Any, set[str], dict[str, int]]:
    """Attempt to coerce value to satisfy schema. Returns (new_value, transforms, branch_choices).

    - Supports type, properties/required, items, anyOf/oneOf (first-pass), and allowlist conversions.
    - Records branch choices as a mapping from path ("/a/b") to chosen index.
    """
    transforms: set[str] = set()
    branch_choices: dict[str, int] = {}

    # enum coercion: try to coerce string to allowed enum values when whitelisted
    if isinstance(schema.get("enum"), list):
        enum_vals = schema["enum"]
        if value in enum_vals:
            return value, transforms, branch_choices
        if isinstance(value, str):
            s = value.strip()
            # boolean
            if "str->bool" in (allow.get("boolean") or []):
                if s.lower() in {"true", "false", "0", "1"}:
                    b = s.lower() in {"true", "1"}
                    if b in enum_vals:
                        transforms.add("str->bool")
                        return b, transforms, branch_choices
            # integer
            if "str->int" in (allow.get("integer") or []) and _looks_like_int(s):
                try:
                    iv = int(s, 10)
                    if iv in enum_vals:
                        transforms.add("str->int")
                        return iv, transforms, branch_choices
                except Exception:
                    pass
            # number
            if "str->float" in (allow.get("number") or []) and _looks_like_float(s):
                try:
                    fv = float(s)
                    if fv in enum_vals:
                        transforms.add("str->float")
                        return fv, transforms, branch_choices
                except Exception:
                    pass
        # If enum present and not matched, do not attempt deeper coercion here
        return value, transforms, branch_choices

    # anyOf / oneOf handling: Prefer ordered branch trial before shortcut acceptance
    for key in ("oneOf", "anyOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            for idx, br in enumerate(branches):
                try:
                    coerced, tf, choices = _coerce_with_schema(value, br, allow, path=path)
                    if _validate_against_schema(coerced, br):
                        transforms |= tf
                        branch_choices["/" + "/".join(path) if path else "/"] = idx
                        return coerced, transforms, branch_choices | choices
                except Exception:
                    continue
            # None matched via coercion; if original already valid for overall schema, accept as-is
            if _validate_against_schema(value, schema):
                return value, transforms, branch_choices
            # else fall through to base coercion by type

    # Shortcut: already valid for non-union schemas
    if _validate_against_schema(value, schema):
        return value, transforms, branch_choices

    # Apply by type
    t = schema.get("type")
    if isinstance(t, str):
        if t == "object" and isinstance(value, dict):
            props = schema.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            out: JSONObject = dict(value)
            for k, sub in props.items():
                if k in out and isinstance(sub, dict):
                    nv, tf, bc = _coerce_with_schema(out[k], sub, allow, path=(*path, k))
                    out[k] = nv
                    transforms |= tf
                    branch_choices.update(bc)
            return out, transforms, branch_choices
        if t == "array":
            if isinstance(value, list):
                items = schema.get("items")
                if isinstance(items, dict):
                    res: List[Any] = []
                    for i, it in enumerate(value):
                        nv, tf, bc = _coerce_with_schema(it, items, allow, path=(*path, str(i)))
                        res.append(nv)
                        transforms |= tf
                        branch_choices.update(bc)
                    return res, transforms, branch_choices
                return value, transforms, branch_choices
            # string→array when allowed and looks like JSON array
            if (
                isinstance(value, str)
                and "str->array" in (allow.get("array") or [])
                and value.strip().startswith("[")
            ):
                try:
                    import json as _json

                    arr = _json.loads(value)
                    if isinstance(arr, list):
                        transforms.add("str->array")
                        return arr, transforms, branch_choices
                except Exception:
                    pass
        if (
            t == "integer"
            and isinstance(value, str)
            and "str->int" in (allow.get("integer") or [])
            and _looks_like_int(value.strip())
        ):
            try:
                iv = int(value.strip(), 10)
                transforms.add("str->int")
                return iv, transforms, branch_choices
            except Exception:
                pass
        if (
            t == "number"
            and isinstance(value, str)
            and "str->float" in (allow.get("number") or [])
            and _looks_like_float(value.strip())
        ):
            try:
                fv = float(value.strip())
                transforms.add("str->float")
                return fv, transforms, branch_choices
            except Exception:
                pass
        if (
            t == "boolean"
            and isinstance(value, str)
            and "str->bool" in (allow.get("boolean") or [])
        ):
            s = value.strip().lower()
            if s in {"true", "false", "0", "1"}:
                transforms.add("str->bool")
                return (s in {"true", "1"}), transforms, branch_choices

    # Fallback: no changes
    return value, transforms, branch_choices


class SmartTypeCoercionProcessor(Processor):
    """Stage 3: Schema-aware, safe type coercion (Ajv-style, minimal version).

    This minimal implementation applies whitelisted, unambiguous conversions
    without requiring a full JSON Schema. It supports per-type allowlists:
      - integer: ["str->int"]
      - number:  ["str->float"]
      - boolean: ["str->bool"]
      - array:   ["str->array"] (string that looks like JSON array)

    A future iteration can accept and use a JSON Schema to narrow conversions
    by field paths and support anyOf/oneOf branch selection.
    """

    name: str = "SmartTypeCoercion"

    def __init__(
        self,
        *,
        allow: Optional[dict[str, list[str]]] = None,
        anyof_strategy: str = "first-pass",
        schema: Optional[dict[str, Any]] = None,
    ) -> None:
        self.allow = allow or {}
        self.anyof_strategy = anyof_strategy
        self.schema = schema or {}

    async def process(self, data: Any, context: Optional[Any] = None) -> Any:
        tm = get_active_trace_manager()
        try:
            # Prefer schema-aware coercion when schema is provided
            if self.schema:
                converted, transforms, branch_choices = _coerce_with_schema(
                    data, self.schema, self.allow, path=()
                )
                if transforms and tm is not None:
                    payload: JSONObject = {
                        "stage": "semantic",
                        "transforms": sorted(list(transforms))[:10],
                    }
                    if branch_choices:
                        payload["branch_choices"] = branch_choices
                    tm.add_event("output.coercion.success", payload)
                elif tm is not None:
                    tm.add_event(
                        "output.coercion.attempt",
                        {
                            "stage": "semantic",
                            "reason": "no-op",
                            "expected_type": _schema_root_type(self.schema),
                            "actual_type": type(data).__name__,
                        },
                    )
                return converted

            # Fallback: allowlist-only coercion
            converted, transforms = _coerce_recursive(data, self.allow)
            if transforms and tm is not None:
                tm.add_event(
                    "output.coercion.success",
                    {"stage": "semantic", "transforms": sorted(list(transforms))[:10]},
                )
            elif tm is not None:
                tm.add_event(
                    "output.coercion.attempt",
                    {
                        "stage": "semantic",
                        "reason": "no-op",
                        "expected_type": None,
                        "actual_type": type(data).__name__,
                    },
                )
            return converted
        except Exception as e:
            try:
                if tm is not None:
                    tm.add_event(
                        "output.coercion.fail",
                        {"stage": "semantic", "error_preview": str(e)[:240]},
                    )
            except Exception:
                pass
            return data


class TolerantJsonDecoderProcessor(Processor):
    """Stage 1: Tiered tolerant JSON decode.

    Order:
    - orjson.loads (fast path) → fallback to json.loads
    - if tolerant_level >= 1: try json5/pyjson5
    - if tolerant_level >= 2: try json-repair (log patch preview)

    Returns decoded Python object on success; otherwise returns the original input.
    """

    name: str = "TolerantJsonDecoder"

    def __init__(self, *, tolerant_level: int = 0) -> None:
        self.tolerant_level = tolerant_level

    async def process(self, data: Any, context: Optional[Any] = None) -> Any:
        tm = get_active_trace_manager()
        if not isinstance(data, (str, bytes)):
            return data
        text = data.decode() if isinstance(data, bytes) else data

        # Try orjson → json
        try:
            try:
                import orjson as _orjson

                obj = _orjson.loads(text)
                if tm is not None:
                    tm.add_event(
                        "output.coercion.success",
                        {"stage": "tolerant", "transforms": ["orjson.loads"]},
                    )
                return obj
            except Exception:
                import json as _json

                obj = _json.loads(text)
                if tm is not None:
                    tm.add_event(
                        "output.coercion.success",
                        {"stage": "tolerant", "transforms": ["json.loads"]},
                    )
                return obj
        except Exception:
            pass

        # Tolerant decoders (opt-in)
        if self.tolerant_level >= 1:
            # Try json5/pyjson5
            try:
                try:
                    import pyjson5 as _json5
                except Exception:
                    import importlib as _importlib

                    _json5 = _importlib.import_module("json5")

                obj = _json5.loads(text)
                if tm is not None:
                    tm.add_event(
                        "output.coercion.success",
                        {"stage": "tolerant", "transforms": ["json5.loads"]},
                    )
                return obj
            except Exception:
                pass

        if self.tolerant_level >= 2:
            # Try json-repair
            try:
                from json_repair import repair_json

                repaired = repair_json(text)
                # Patch preview (bounded)
                preview = repaired[:240] if isinstance(repaired, str) else str(repaired)[:240]
                if tm is not None:
                    tm.add_event(
                        "output.coercion.attempt",
                        {
                            "stage": "tolerant",
                            "reason": "json_repair",
                            "expected_type": None,
                            "actual_type": "string",
                        },
                    )
                import json as _json

                obj = _json.loads(repaired)
                if tm is not None:
                    tm.add_event(
                        "output.coercion.success",
                        {
                            "stage": "tolerant",
                            "transforms": ["json_repair", "json.loads"],
                            "patch_preview": preview,
                        },
                    )
                return obj
            except Exception:
                pass

        # Failed all attempts: emit fail and return input
        try:
            if tm is not None:
                tm.add_event(
                    "output.coercion.fail",
                    {"stage": "tolerant", "error_preview": "all decode attempts failed"},
                )
        except Exception:
            pass
        return data

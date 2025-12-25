from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..domain.models import BaseModel
from ..type_definitions.common import JSONObject

PatchOp = dict[str, Any]


def _escape_path(segment: object) -> str:
    text = str(segment)
    return text.replace("~", "~0").replace("/", "~1")


def _normalize_json(value: object) -> object:
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, tuple):
        return [_normalize_json(v) for v in value]
    if isinstance(value, set):
        try:
            return [_normalize_json(v) for v in sorted(value, key=str)]
        except Exception:
            return [_normalize_json(v) for v in list(value)]
    if isinstance(value, list):
        return [_normalize_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    return value


def _diff_values(before: object, after: object, path: str, ops: list[PatchOp]) -> None:
    if before == after:
        return

    if isinstance(before, dict) and isinstance(after, dict):
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        for key in sorted(before_keys - after_keys, key=str):
            ops.append({"op": "remove", "path": f"{path}/{_escape_path(key)}"})
        for key in sorted(after_keys - before_keys, key=str):
            ops.append(
                {
                    "op": "add",
                    "path": f"{path}/{_escape_path(key)}",
                    "value": after[key],
                }
            )
        for key in sorted(before_keys & after_keys, key=str):
            _diff_values(before[key], after[key], f"{path}/{_escape_path(key)}", ops)
        return

    if isinstance(before, list) and isinstance(after, list):
        common_len = min(len(before), len(after))
        for idx in range(common_len):
            _diff_values(before[idx], after[idx], f"{path}/{idx}", ops)
        if len(before) > len(after):
            for idx in range(len(before) - 1, len(after) - 1, -1):
                ops.append({"op": "remove", "path": f"{path}/{idx}"})
        elif len(after) > len(before):
            for idx in range(len(before), len(after)):
                ops.append({"op": "add", "path": f"{path}/{idx}", "value": after[idx]})
        return

    ops.append({"op": "replace", "path": path or "", "value": after})


class DiffProcessor:
    """Compute JSON patch operations between two JSON-like payloads."""

    name: str = "DiffProcessor"

    async def process(self, data: object, context: BaseModel | None = None) -> JSONObject:
        before, after = self._extract_pair(data)
        norm_before = _normalize_json(before)
        norm_after = _normalize_json(after)
        ops: list[PatchOp] = []
        _diff_values(norm_before, norm_after, "", ops)
        return {"patch": ops, "before": norm_before, "after": norm_after}

    @staticmethod
    def _extract_pair(data: object) -> tuple[object, object]:
        if isinstance(data, dict) and "before" in data and "after" in data:
            return data["before"], data["after"]
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return data[0], data[1]
        missing = object()
        before = getattr(data, "before", missing)
        after = getattr(data, "after", missing)
        if (
            before is not missing
            and after is not missing
            and before is not None
            and after is not None
        ):
            return before, after
        raise ValueError("DiffProcessor expects {'before':..., 'after':...} or (before, after)")

from __future__ import annotations

from typing import Any, Optional

from flujo.type_definitions.common import JSONObject
from flujo.infra.skill_models import SkillRegistration

from .support import return_yaml_for_cli


def register_transform_skills(reg: Any) -> None:
    """Register data transform builtins into the provided registry."""

    async def to_csv(rows: Any, *, headers: Optional[list[str]] = None) -> str:
        import csv
        import io

        norm: list[JSONObject]
        if isinstance(rows, dict):
            norm = [rows]
        elif isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
            norm = rows
        else:
            if isinstance(rows, list):
                norm = [x if isinstance(x, dict) else {"value": x} for x in rows]
            else:
                norm = [rows if isinstance(rows, dict) else {"value": rows}]
        if headers and isinstance(headers, list) and all(isinstance(h, str) for h in headers):
            cols = list(headers)
        else:
            keys: set[str] = set()
            for row in norm:
                try:
                    keys.update(k for k in row.keys() if isinstance(k, str))
                except Exception:
                    continue
            cols = sorted(keys)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        if cols:
            writer.writeheader()
        for row in norm:
            try:
                writer.writerow({k: row.get(k, "") for k in cols})
            except Exception:
                continue
        return buf.getvalue()

    reg.register(
        **SkillRegistration(
            id="flujo.builtins.to_csv",
            factory=lambda **_params: to_csv,
            description="Convert list[dict] into CSV string (deterministic headers).",
            input_schema={
                "type": "object",
                "properties": {
                    "rows": {"type": ["array", "object"]},
                    "headers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rows"],
            },
            side_effects=False,
        ).__dict__
    )

    async def aggregate(
        data: Any,
        *,
        operation: str,
        field: Optional[str] = None,
    ) -> float | int:
        op = (operation or "").strip().lower()
        items: list[JSONObject]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            items = []

        def _nums() -> list[float]:
            out: list[float] = []
            if not field:
                return out
            for obj in items:
                try:
                    val = obj.get(field)
                    if isinstance(val, (int, float)):
                        out.append(float(val))
                except Exception:
                    continue
            return out

        if op == "count":
            if field:
                c = 0
                for obj in items:
                    try:
                        if field in obj and obj.get(field) is not None:
                            c += 1
                    except Exception:
                        continue
                return int(c)
            return int(len(items))
        if op == "sum":
            nums = _nums()
            return float(sum(nums)) if nums else 0.0
        if op in {"avg", "average", "mean"}:
            nums = _nums()
            return float(sum(nums)) / float(len(nums)) if nums else 0.0
        return 0

    reg.register(
        **SkillRegistration(
            id="flujo.builtins.aggregate",
            factory=lambda **_params: aggregate,
            description="Aggregate numeric field across list[dict]: sum/avg/count.",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": ["array", "object"]},
                    "operation": {"type": "string"},
                    "field": {"type": "string"},
                },
                "required": ["data", "operation"],
            },
            side_effects=False,
        ).__dict__
    )

    async def select_fields(
        data: Any,
        *,
        include: Optional[list[str]] = None,
        rename: Optional[dict[str, str]] = None,
    ) -> Any:
        includes = list(include) if include else None
        ren = dict(rename) if rename else {}

        def _project(obj: JSONObject) -> JSONObject:
            try:
                keys = list(obj.keys()) if includes is None else [k for k in includes]
                out: JSONObject = {}
                for k in keys:
                    if k in obj:
                        out[ren.get(k, k)] = obj.get(k)
                if includes is None and ren:
                    for k, newk in ren.items():
                        if k in obj:
                            out[newk] = obj.get(k)
                return out
            except Exception:
                return {}

        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [_project(x) for x in data]
        if isinstance(data, dict):
            return _project(data)
        return data

    reg.register(
        "flujo.builtins.select_fields",
        lambda **_params: select_fields,
        description="Project/rename fields on dict or list[dict] using include/rename.",
        arg_schema={
            "type": "object",
            "properties": {
                "data": {"type": ["object", "array"]},
                "include": {"type": "array", "items": {"type": "string"}},
                "rename": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["data"],
        },
        side_effects=False,
    )

    async def flatten(items: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        out: list[Any] = []
        for sub in items:
            if isinstance(sub, list):
                out.extend(sub)
            elif isinstance(sub, tuple):
                out.extend(list(sub))
            else:
                out.append(sub)
        return out

    reg.register(
        "flujo.builtins.flatten",
        lambda **_params: flatten,
        description="Flatten one level of nesting in a list of lists.",
        arg_schema={
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
        },
        side_effects=False,
    )

    reg.register(
        "flujo.builtins.return_yaml_for_cli",
        lambda **_params: return_yaml_for_cli,
        description=(
            "Return YAML in the format that the CLI expects to find "
            "(with generated_yaml and yaml_text keys)."
        ),
    )


__all__ = ["register_transform_skills"]

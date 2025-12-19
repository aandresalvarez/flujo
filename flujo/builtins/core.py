from __future__ import annotations

from typing import Any, Dict, List, Optional

from flujo.type_definitions.common import JSONObject

# Ensure framework primitives (like StateMachine) are registered when builtins load
try:  # pragma: no cover - best-effort for import order
    import flujo.framework as _framework  # noqa: F401
except Exception:
    pass
from flujo.infra.skill_registry import get_skill_registry
from .context import context_get, context_merge, context_set, register_context_builtins
from .support import (
    always_valid_key,
    capture_validation_report,
    check_user_confirmation,
    compute_validity_key,
    extract_decomposed_steps,
    extract_validation_errors,
    extract_yaml_text,
    passthrough,
    repair_yaml_ruamel,
    return_yaml_for_cli,
    select_by_yaml_shape,
    select_validity_branch,
    shape_to_validity_flag,
    validation_report_to_flag,
    welcome_agent,
)
from .architect import DiscoverSkillsAgent
from .optional import register_optional_builtins
from .extras import _DDGSAsync, _DDGS_CLASS, render_jinja_template


def register_core_builtins() -> None:
    """Public entrypoint for registering core builtins."""
    _register_builtins()


def _register_builtins() -> None:
    """Register builtin skills with the global registry."""
    reg = get_skill_registry()
    register_context_builtins()
    register_optional_builtins()

    async def _stringify(x: Any) -> str:
        try:
            if isinstance(x, (str, bytes)):
                return x.decode() if isinstance(x, bytes) else x
            return str(x)
        except Exception:
            return str(x)

    if reg.get("flujo.builtins.stringify") is None:
        reg.register(
            "flujo.builtins.stringify",
            _stringify,
            description="Return input as string",
            input_schema={"type": ["string", "object", "array", "number", "boolean", "null"]},
            side_effects=False,
        )

    def _context_set_factory(**params: Any) -> Any:
        if params:

            async def _runner(_data: Any = None, *, context: Any = None, **kwargs: Any) -> Any:
                merged = dict(params)
                merged.update(kwargs)
                if "value" not in merged and _data is not None:
                    merged["value"] = _data
                return await context_set(
                    path=merged.get("path", ""), value=merged.get("value"), context=context
                )

            return _runner
        return context_set

    def _context_merge_factory(**params: Any) -> Any:
        if params:

            async def _runner(_data: Any = None, *, context: Any = None, **kwargs: Any) -> Any:
                merged = dict(params)
                merged.update(kwargs)
                value_obj = merged.get("value", {})
                if value_obj is None and _data is not None:
                    value_obj = _data
                if not isinstance(value_obj, dict):
                    try:
                        value_obj = dict(value_obj)
                    except Exception:
                        value_obj = {}
                return await context_merge(
                    path=str(merged.get("path", "")), value=value_obj, context=context
                )

            return _runner
        return context_merge

    reg.register(
        "flujo.builtins.context_set",
        _context_set_factory,
        description="Set a context field at a dot-separated path (e.g., 'import_artifacts.counter')",
        arg_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "value": {}},
            "required": ["path", "value"],
        },
        side_effects=True,
    )
    reg.register(
        "flujo.builtins.context_merge",
        _context_merge_factory,
        description="Merge a dictionary into context at a path (e.g., 'import_artifacts.extras.settings')",
        arg_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "value": {"type": "object"}},
            "required": ["path", "value"],
        },
        side_effects=True,
    )
    reg.register(
        "flujo.builtins.context_get",
        lambda **_p: context_get,
        description="Get a value from context at a dot-separated path with optional default",
        arg_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "default": {}},
            "required": ["path"],
        },
        side_effects=False,
    )

    reg.register(
        "flujo.builtins.welcome_agent",
        lambda **_p: welcome_agent,
        description="Returns a fun welcome message for new users.",
        arg_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "default": "Developer"}},
        },
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.discover_skills",
        lambda directory=".": DiscoverSkillsAgent(directory=directory),
        description="Discover local and packaged skills; returns available_skills list.",
    )

    reg.register(
        "flujo.builtins.extract_decomposed_steps",
        lambda **_p: extract_decomposed_steps,
        description="Extract list of step dicts from decomposer output into prepared_steps_for_mapping",
    )
    reg.register(
        "flujo.builtins.extract_yaml_text",
        lambda **_p: extract_yaml_text,
        description="Extract YAML string from YamlWriter output object or dict.",
    )
    reg.register(
        "flujo.builtins.capture_validation_report",
        lambda **_p: capture_validation_report,
        description="Capture full validation report in context for later error extraction.",
    )
    reg.register(
        "flujo.builtins.validation_report_to_flag",
        lambda **_p: validation_report_to_flag,
        description="Map validation report to {'yaml_is_valid': bool} and update context.",
    )
    reg.register(
        "flujo.builtins.select_validity_branch",
        lambda **_p: select_validity_branch,
        description="Return 'valid' if context.yaml_is_valid else 'invalid'.",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.select_by_yaml_shape",
        lambda **_p: select_by_yaml_shape,
        description="Return 'invalid' when context.yaml_text uses inline list for steps, else 'valid'.",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.extract_validation_errors",
        lambda **_p: extract_validation_errors,
        description="Extract error messages from a validation report into context.validation_errors.",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.shape_to_validity_flag",
        lambda **_p: shape_to_validity_flag,
        description="Return {'yaml_is_valid': bool} based on simple inline-list shape after steps:",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.return_yaml_for_cli",
        lambda **_p: return_yaml_for_cli,
        description="Return YAML in the format that the CLI expects to find (with generated_yaml and yaml_text keys).",
    )
    reg.register(
        "flujo.builtins.always_valid_key",
        lambda **_p: always_valid_key,
        description="Return 'valid' unconditionally for post-repair branch logging.",
        side_effects=False,
    )

    async def to_csv(
        rows: list[JSONObject] | JSONObject, *, headers: Optional[List[str]] = None
    ) -> str:
        import csv
        import io

        norm: List[JSONObject]
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
        "flujo.builtins.to_csv",
        lambda **_p: to_csv,
        description="Convert list[dict] into CSV string (deterministic headers).",
        arg_schema={
            "type": "object",
            "properties": {
                "rows": {"type": ["array", "object"]},
                "headers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["rows"],
        },
        side_effects=False,
    )

    async def aggregate(
        data: list[JSONObject] | JSONObject,
        *,
        operation: str,
        field: Optional[str] = None,
    ) -> float | int:
        op = (operation or "").strip().lower()
        items: List[JSONObject]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            items = []

        def _nums() -> List[float]:
            out: List[float] = []
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
        "flujo.builtins.aggregate",
        lambda **_p: aggregate,
        description="Aggregate numeric field across list[dict]: sum/avg/count.",
        arg_schema={
            "type": "object",
            "properties": {
                "data": {"type": ["array", "object"]},
                "operation": {"type": "string"},
                "field": {"type": "string"},
            },
            "required": ["data", "operation"],
        },
        side_effects=False,
    )

    async def select_fields(
        data: list[JSONObject] | JSONObject,
        *,
        include: Optional[List[str]] = None,
        rename: Optional[Dict[str, str]] = None,
    ) -> list[JSONObject] | JSONObject:
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
        lambda **_p: select_fields,
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

    async def flatten(items: list[Any]) -> list[Any]:
        if not isinstance(items, list):
            return []
        out: List[Any] = []
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
        lambda **_p: flatten,
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
        lambda **_p: return_yaml_for_cli,
        description="Return YAML in the format that the CLI expects to find (with generated_yaml and yaml_text keys).",
    )

    async def web_search(query: str, max_results: int = 3) -> List[JSONObject]:
        # Allow tests to monkeypatch the facade module while keeping lazy optional deps.
        try:
            import importlib

            _mod = importlib.import_module("flujo.builtins")
            ddgs_async = getattr(_mod, "_DDGSAsync", _DDGSAsync)
            ddgs_class = getattr(_mod, "_DDGS_CLASS", _DDGS_CLASS)
        except Exception:
            ddgs_async = _DDGSAsync
            ddgs_class = _DDGS_CLASS
        results: List[JSONObject] = []
        try:
            if ddgs_async is None and ddgs_class is None:
                return []
            if ddgs_async is not None:
                async with ddgs_async() as ddgs:
                    if hasattr(ddgs, "atext"):
                        async for r in ddgs.atext(query, max_results=max_results):
                            try:
                                results.append(
                                    {
                                        "title": r.get("title"),
                                        "link": r.get("href") or r.get("url"),
                                        "snippet": r.get("body") or r.get("snippet"),
                                    }
                                )
                            except Exception:
                                continue
                    elif hasattr(ddgs, "text"):
                        async for r in ddgs.text(query, max_results=max_results):
                            try:
                                results.append(
                                    {
                                        "title": r.get("title"),
                                        "link": r.get("href") or r.get("url"),
                                        "snippet": r.get("body") or r.get("snippet"),
                                    }
                                )
                            except Exception:
                                continue
            elif ddgs_class is not None:
                ddgs = ddgs_class()
                data = ddgs.text(query, max_results=max_results) or []
                for r in data:
                    if not isinstance(r, dict):
                        continue
                    results.append(
                        {
                            "title": r.get("title"),
                            "link": r.get("href") or r.get("url"),
                            "snippet": r.get("body") or r.get("snippet"),
                        }
                    )
        except Exception:
            results = []
        return results

    reg.register(
        "flujo.builtins.web_search",
        lambda **_p: web_search,
        description="Perform a web search and return summarized results",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
        side_effects=False,
    )

    reg.register(
        "flujo.builtins.render_jinja_template",
        lambda **_p: render_jinja_template,
        description="Render a Jinja2 template string with a variables mapping.",
        arg_schema={
            "type": "object",
            "properties": {"template": {"type": "string"}, "variables": {"type": "object"}},
            "required": ["template"],
        },
        side_effects=False,
    )

    reg.register(
        "flujo.builtins.repair_yaml_ruamel",
        lambda **_p: repair_yaml_ruamel,
        description="Conservatively repair YAML text via ruamel.yaml round-trip load/dump.",
    )
    reg.register(
        "flujo.builtins.compute_validity_key",
        lambda **_p: compute_validity_key,
        description="Return 'valid' or 'invalid' based on context.yaml_is_valid.",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.check_user_confirmation",
        lambda **_p: check_user_confirmation,
        description="Interpret user input as 'approved' when affirmative (y/yes/empty), else 'denied'.",
        side_effects=False,
    )
    reg.register(
        "flujo.builtins.passthrough",
        lambda **_p: passthrough,
        description="Identity adapter that returns input unchanged.",
        side_effects=False,
    )


_register_builtins()

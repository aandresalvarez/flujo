from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _parse_openapi_text(*, text: str, suffix_hint: str | None) -> dict[str, Any]:
    hint = (suffix_hint or "").lower()
    if hint in {".json"}:
        return dict(json.loads(text))

    # Best-effort: try JSON first, then YAML.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return dict(parsed)
    except Exception:
        pass

    if yaml is None:
        raise RuntimeError("PyYAML is required to load non-JSON OpenAPI specs")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("OpenAPI spec did not load to a mapping")
    return dict(loaded)


def _read_spec_text(spec: str) -> tuple[str, str | None]:
    """Return (text, suffix_hint) for a local file path or http(s) URL."""
    parsed = urlparse(spec)
    if parsed.scheme in {"http", "https"}:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"httpx is required to load OpenAPI specs from URL: {exc}") from exc
        try:
            resp = httpx.get(spec, timeout=30.0)
            resp.raise_for_status()
            return resp.text, Path(parsed.path).suffix or None
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch OpenAPI spec from URL: {exc}") from exc

    path = Path(spec)
    return path.read_text(encoding="utf-8"), path.suffix or None


def load_openapi_spec(spec: str) -> dict[str, Any]:
    """Load an OpenAPI spec from a local file path or URL (JSON or YAML)."""
    text, suffix_hint = _read_spec_text(spec)
    return _parse_openapi_text(text=text, suffix_hint=suffix_hint)


def _safe_name(name: str) -> str:
    """Sanitize a name into snake_case."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower() or "operation"


def _iter_operations(spec: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    paths = spec.get("paths", {}) or {}
    for path, ops in paths.items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not isinstance(op, dict):
                continue
            yield path, method.lower(), op


def generate_openapi_agents(
    *,
    spec: dict[str, Any],
    models_package: str,
    output_dir: Path,
    agents_filename: str = "openapi_agents.py",
    models_module: str = "generated_models",
) -> Path:
    """Generate agent factory code for OpenAPI operations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").touch(exist_ok=True)
    agents_path = output_dir / agents_filename

    ops = list(_iter_operations(spec))
    if not ops:
        raise ValueError("No operations found in OpenAPI spec")

    functions: list[str] = []
    response_models: dict[str, str] = {}
    for path, method, op in ops:
        op_id = op.get("operationId") or f"{method}_{path}"
        fname = _safe_name(op_id)
        resp_model = "dict"
        try:
            responses = op.get("responses") or {}
            for status_code in ("200", "201", "202"):
                resp_obj = responses.get(status_code)
                if not isinstance(resp_obj, dict):
                    continue
                content = resp_obj.get("content") or {}
                if not isinstance(content, dict):
                    continue
                json_content = content.get("application/json")
                if not isinstance(json_content, dict):
                    continue
                schema = json_content.get("schema")
                if not isinstance(schema, dict):
                    continue
                ref = schema.get("$ref")
                if isinstance(ref, str):
                    resp_model = ref.split("/")[-1]
                    break
                type_hint = schema.get("type")
                if type_hint == "array":
                    items = schema.get("items")
                    if isinstance(items, dict):
                        item_ref = items.get("$ref")
                        if isinstance(item_ref, str):
                            resp_model = f"list[{item_ref.split('/')[-1]}]"
                            break
                        item_type = items.get("type")
                        if item_type in {"string", "integer", "number", "boolean", "object"}:
                            scalar_map = {
                                "string": "str",
                                "integer": "int",
                                "number": "float",
                                "boolean": "bool",
                                "object": "dict",
                            }
                            resp_model = f"list[{scalar_map[item_type]}]"
                            break
                    resp_model = "list"
                    break
                if type_hint == "object":
                    resp_model = "dict"
                    break
        except Exception:
            resp_model = "dict"
        response_models[fname] = resp_model
        functions.append(
            f"""
async def {fname}(
    *,
    base_url: str,
    path_params: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: Any | None = None,
) -> dict[str, Any]:
    \"\"\"Call {method.upper()} {path} from the generated agent.\"\"\"
    import re

    _path = "{path}"
    for _k, _v in (path_params or {{}}).items():
        _path = re.sub(r"\\{{" + _k + r"\\}}", str(_v), _path)
    url_path = _path
    return await _http_request(
        method="{method.upper()}",
        base_url=base_url,
        url_path=url_path,
        query=query,
        headers=headers,
        json_body=body,
    )
"""
        )

    func_list = ", ".join(_safe_name(op.get("operationId") or f"{m}_{p}") for p, m, op in ops)

    resp_map_lines = "\n".join(
        f'    "{name}": "{model}",' for name, model in response_models.items()
    )
    op_func_lines = "\n".join(f'    "{name}": {name},' for name in response_models.keys())
    content = f'''"""
Auto-generated OpenAPI agent wrappers.

This file was generated by flujo dev import-openapi.
"""
from __future__ import annotations

from typing import Any

import httpx

from flujo.agents.wrapper import make_agent_async, AsyncAgentWrapper
from .{models_module} import *  # noqa: F401,F403

# Response model names per operation (if schema was detected)
RESPONSE_MODEL_NAMES: dict[str, str] = {{
{resp_map_lines}
}}

def _resolve_output_type(name: str | None) -> type[Any]:
    if name is None:
        return dict
    if name == "dict":
        return dict
    if name == "list":
        return list
    if name.startswith("list[") and name.endswith("]"):
        inner = name[5:-1].strip()
        if inner == "dict":
            return list[dict]
        if inner == "str":
            return list[str]
        if inner == "int":
            return list[int]
        if inner == "float":
            return list[float]
        if inner == "bool":
            return list[bool]
        obj_inner = globals().get(inner)
        if isinstance(obj_inner, type):
            return list[obj_inner]
        return list
    obj = globals().get(name)
    return obj if isinstance(obj, type) else dict


async def _http_request(
    *,
    method: str,
    base_url: str,
    url_path: str,
    query: dict[str, Any] | None,
    headers: dict[str, str] | None,
    json_body: Any | None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + url_path
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, params=query, headers=headers, json=json_body)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {{"status_code": response.status_code, "content": response.text}}


{"".join(functions)}

OPERATION_FUNCS = {{
{op_func_lines}
}}


def make_openapi_agent(
    *,
    base_url: str,
    model: str = "openai:gpt-4o-mini",
    system_prompt: str | None = None,
) -> AsyncAgentWrapper:
    \"\"\"Create an agent with tools for all operations in the spec.\"\"\"
    tools = [{func_list}]
    prompt = system_prompt or "You are an API caller that uses provided tools to call the target OpenAPI service."
    return make_agent_async(
        model=model,
        system_prompt=prompt,
        tools=tools,
        output_type=dict,
    )


def make_openapi_operation_agent(
    *,
    base_url: str,
    operation: str,
    model: str = "openai:gpt-4o-mini",
    system_prompt: str | None = None,
) -> AsyncAgentWrapper:
    \"\"\"Create an agent for a single OpenAPI operation with typed output when known.\"\"\"
    if operation not in OPERATION_FUNCS:
        raise ValueError(f"Unknown operation: {{operation}}")
    tool = OPERATION_FUNCS[operation]
    prompt = system_prompt or "You are an API caller for this OpenAPI operation."
    output_type = _resolve_output_type(RESPONSE_MODEL_NAMES.get(operation))
    return make_agent_async(
        model=model,
        system_prompt=prompt,
        tools=[tool],
        output_type=output_type,
    )
'''
    agents_path.write_text(content, encoding="utf-8")
    return agents_path

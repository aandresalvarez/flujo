from __future__ import annotations

from typing import Any, Optional, Type

from flujo.type_definitions.common import JSONObject

from flujo.infra.skill_registry import get_skill_registry
from flujo.infra.jinja_utils import create_jinja_environment

_DDGSAsync: Optional[Type[Any]] = None
_DDGS_CLASS: Optional[Type[Any]] = None
try:  # pragma: no cover - optional dependency
    _ddgs_module = None
    try:
        import ddgs

        _ddgs_module = ddgs
    except Exception:
        try:
            import duckduckgo_search  # deprecated upstream, kept for compatibility

            _ddgs_module = duckduckgo_search
        except Exception:
            pass

    if _ddgs_module is not None:
        _async = getattr(_ddgs_module, "AsyncDDGS", None)
        _sync = getattr(_ddgs_module, "DDGS", None)
    else:
        _async = None
        _sync = None
    if _async is not None:
        _DDGSAsync = _async
    if _sync is not None:
        _DDGS_CLASS = _sync
except Exception:
    _DDGSAsync = None
    _DDGS_CLASS = None

_httpx: Any = None
try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except Exception:
    pass

_jinja2: Any = None
try:
    import jinja2 as _jinja2
except ImportError:
    pass


def register_optional_builtins() -> None:
    reg = get_skill_registry()

    async def _web_search(*, query: str, max_results: int = 5) -> str:
        try:
            if _DDGSAsync is not None:
                async with _DDGSAsync() as ddgs:
                    results = []
                    async for r in ddgs.atext(query, max_results=max_results):
                        try:
                            title = r.get("title") or ""
                            href = r.get("href") or r.get("url") or ""
                            body = r.get("body") or r.get("snippet") or ""
                            results.append(f"- {title}\n  {href}\n  {body}")
                        except Exception:
                            continue
                    return "\n".join(results) if results else ""
            elif _DDGS_CLASS is not None:
                with _DDGS_CLASS() as ddgs:
                    data = ddgs.text(query, max_results=max_results) or []
                    lines = []
                    for r in data:
                        try:
                            title = r.get("title") or ""
                            href = r.get("href") or r.get("url") or ""
                            body = r.get("body") or r.get("snippet") or ""
                            lines.append(f"- {title}\n  {href}\n  {body}")
                        except Exception:
                            continue
                    return "\n".join(lines)
        except Exception:
            pass
        return f"(web_search stub) query='{query}'"

    reg.register(
        "flujo.builtins.web_search",
        lambda: _web_search,
        description="Perform a web search and return summarized results",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
        side_effects=False,
    )

    async def _http_get(*, url: str, timeout_s: int = 10) -> str:
        try:
            if _httpx is not None:
                async with _httpx.AsyncClient(timeout=timeout_s) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return str(resp.text)
        except Exception:
            pass
        return f"(http_get stub) url='{url}'"

    reg.register(
        "flujo.builtins.http_get",
        lambda: _http_get,
        description="Fetch content from a URL",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}, "timeout_s": {"type": "integer"}},
            "required": ["url"],
        },
        side_effects=False,
    )

    async def render_jinja_template(template: str, variables: Optional[JSONObject] = None) -> str:
        if _jinja2 is None:
            return template
        try:
            env = create_jinja_environment(_jinja2)
            tmpl = env.from_string(template)
            result = tmpl.render(**(variables or {}))
            return str(result)
        except Exception:
            return template

    reg.register(
        "flujo.builtins.render_jinja_template",
        lambda **_params: render_jinja_template,
        description="Render a Jinja2 template string with a variables mapping.",
        arg_schema={
            "type": "object",
            "properties": {
                "template": {"type": "string"},
                "variables": {"type": "object"},
            },
            "required": ["template"],
        },
        side_effects=False,
    )


__all__ = ["register_optional_builtins"]

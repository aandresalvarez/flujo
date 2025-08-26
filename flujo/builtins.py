from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, cast

# Ensure framework primitives (like StateMachine) are registered when builtins load
try:  # pragma: no cover - best-effort for import order
    import flujo.framework as _framework  # noqa: F401
except Exception:
    pass
from pydantic import BaseModel as PydanticBaseModel
from flujo.domain.models import BaseModel as DomainBaseModel

from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.infra.skill_registry import get_skill_registry
from flujo.domain.agent_protocol import AsyncAgentProtocol
from .agents.wrapper import make_agent_async

# Lazy imports for optional dependencies
_jinja2: Any = None
try:
    import jinja2 as _jinja2
except ImportError:
    pass

_ruamel_yaml: Any = None
try:
    import ruamel.yaml as _ruamel_yaml
except ImportError:
    pass

# Optional dependency: pyfiglet for ASCII art (installed via extras: flujo[skills])
_pyfiglet: Any = None
try:  # pragma: no cover - optional dependency
    import pyfiglet as _pyfiglet
except Exception:
    pass

# Optional dependency: web search client (ddgs preferred; fallback to duckduckgo_search)
# Prefer async client if available; otherwise use sync DDGS in a thread pool.
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

# Optional dependency for HTTP client
_httpx: Any = None
try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except Exception:
    pass


# --- Core builtin skills registration ---


def _register_core_skills() -> None:
    reg = get_skill_registry()

    # Stringify: identity formatter as a safe default
    async def _stringify(x: Any) -> Dict[str, Any] | str:
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

    # Web search: use ddgs/duckduckgo if available, otherwise stub
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
        # Fallback: no-op string
        return f"(web_search stub) query='{query}'"

    if reg.get("flujo.builtins.web_search") is None:
        reg.register(
            "flujo.builtins.web_search",
            lambda: _web_search,  # factory returning an async callable
            description="Perform a web search and return summarized results",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"],
            },
            side_effects=False,
        )

    # HTTP GET: simple fetcher
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

    if reg.get("flujo.builtins.http_get") is None:
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

    # Write file: side-effect skill
    async def _fs_write_file(data: Any, *, path: str, encoding: str = "utf-8") -> str:
        from pathlib import Path as _Path

        p = _Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            text = (
                data
                if isinstance(data, str)
                else (data.decode() if isinstance(data, bytes) else str(data))
            )
            p.write_text(text, encoding=encoding)
            return f"wrote:{p.as_posix()}"
        except Exception as e:
            return f"error:{type(e).__name__}:{e}"

    if reg.get("flujo.builtins.fs_write_file") is None:
        reg.register(
            "flujo.builtins.fs_write_file",
            lambda: _fs_write_file,
            description="Write string content to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string"},
                },
                "required": ["path"],
            },
            side_effects=True,
        )


# Ensure skills are registered when this module is imported by the CLI
try:
    _register_core_skills()
except Exception:
    # Never fail module import due to registration
    pass


# --- Architect agent stubs (Planner, ToolMatcher, YAML Writer) ---


def _register_architect_agents() -> None:
    """Register stub implementations for architect agents.

    These stubs enable local iteration without external LLMs.
    They follow the contracts defined in flujo/architect/models.py.
    """
    reg = get_skill_registry()

    # Planner Agent: decomposes user goal into high-level steps
    async def _planner_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(payload.get("user_goal") or "").strip()
        g = goal.lower()
        steps: List[Dict[str, str]] = []
        if not goal:
            steps = [{"step_name": "UnderstandGoal", "purpose": "Clarify the intended outcome."}]
        else:
            # Very lightweight heuristic decomposition
            if "http" in g or "url" in g or "https://" in g or "http://" in g:
                steps.append(
                    {
                        "step_name": "FetchWebpage",
                        "purpose": "Fetch the content from the referenced URL.",
                    }
                )
            elif "search" in g or "find" in g or "lookup" in g:
                steps.append(
                    {
                        "step_name": "WebSearch",
                        "purpose": "Search the web for relevant information.",
                    }
                )
            else:
                steps.append(
                    {
                        "step_name": "Echo Input",
                        "purpose": "Safely echo or stringify the input as a baseline step.",
                    }
                )

            if "save" in g or "write" in g or "export" in g:
                steps.append(
                    {
                        "step_name": "SaveToFile",
                        "purpose": "Persist the result to a file if requested.",
                    }
                )

        plan_summary = (
            f"Plan derived from goal: {goal[:80]}" if goal else "Plan derived from unspecified goal"
        )
        return {"plan_summary": plan_summary, "steps": steps}

    if reg.get("flujo.architect.planner") is None:
        reg.register(
            "flujo.architect.planner",
            lambda: _planner_agent,
            description="Agentic planner: decomposes goal into high-level steps.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_goal": {"type": "string"},
                    "available_skills": {"type": "array"},
                    "project_summary": {"type": "string"},
                    "flujo_schema": {"type": "object"},
                },
                "required": ["user_goal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "plan_summary": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_name": {"type": "string"},
                                "purpose": {"type": "string"},
                            },
                            "required": ["step_name", "purpose"],
                        },
                    },
                },
                "required": ["plan_summary", "steps"],
            },
            side_effects=False,
        )

    # Tool Matcher Agent: select a skill for each planned step
    async def _tool_matcher_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
        step_name = payload.get("step_name") or "Step"
        purpose = (payload.get("purpose") or "").lower()
        available = payload.get("available_skills") or []

        # Helper to check availability
        def _is_avail(sid: str) -> bool:
            try:
                if any(isinstance(x, dict) and x.get("id") == sid for x in available):
                    return True
                entry = get_skill_registry().get(sid)
                # Treat empty dicts/None as unavailable
                return bool(entry) if isinstance(entry, dict) else (entry is not None)
            except Exception:
                return False

        # Simple heuristics to choose a skill
        if any(k in purpose for k in ["http", "url", "fetch", "webpage", "download"]):
            sid = (
                "flujo.builtins.http_get"
                if _is_avail("flujo.builtins.http_get")
                else "flujo.builtins.stringify"
            )
            params: Dict[str, Any] = {}
        elif any(k in purpose for k in ["search", "find", "lookup", "discover"]):
            sid = (
                "flujo.builtins.web_search"
                if _is_avail("flujo.builtins.web_search")
                else "flujo.builtins.stringify"
            )
            params = {"query": purpose[:80]} if sid.endswith("web_search") else {}
        elif any(k in purpose for k in ["save", "write", "persist", "export", "file"]):
            sid = (
                "flujo.builtins.fs_write_file"
                if _is_avail("flujo.builtins.fs_write_file")
                else "flujo.builtins.stringify"
            )
            params = {"path": "output.txt"} if sid.endswith("fs_write_file") else {}
        else:
            sid = "flujo.builtins.stringify"
            params = {}

        return {"step_name": step_name, "chosen_agent_id": sid, "agent_params": params}

    if reg.get("flujo.architect.tool_matcher") is None:
        reg.register(
            "flujo.architect.tool_matcher",
            lambda: _tool_matcher_agent,
            description="Agentic tool matcher: selects best skill for a step.",
            input_schema={
                "type": "object",
                "properties": {
                    "step_name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "available_skills": {"type": "array"},
                },
                "required": ["step_name", "purpose"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "step_name": {"type": "string"},
                    "chosen_agent_id": {"type": "string"},
                    "agent_params": {"type": "object"},
                },
                "required": ["step_name", "chosen_agent_id", "agent_params"],
            },
            side_effects=False,
        )

    # YAML Writer Agent: assemble final pipeline.yaml
    async def _yaml_writer_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
        goal = payload.get("user_goal")
        selections = payload.get("tool_selections") or []
        # schema = payload.get("flujo_schema") or {}  # Unused variable removed
        name = goal or "generated_pipeline"
        try:
            if isinstance(name, str):
                import re as _re

                norm = _re.sub(r"[^A-Za-z0-9\s]+", "", name)[:40].strip().lower()
                if norm:
                    name = ("_".join(norm.split()) or name)[:40]
        except Exception:
            name = "generated_pipeline"

        import yaml as _yaml

        steps_yaml: List[str] = []
        # Decide whether to construct a parallel block when goal hints parallelism
        goal_text = (goal or "").lower() if isinstance(goal, str) else ""
        wants_parallel = ("parallel" in goal_text or "concurrent" in goal_text) and len(
            selections
        ) > 1

        if wants_parallel:
            # Build a single ParallelStep with each selection as its own branch
            branches: Dict[str, List[Dict[str, Any]]] = {}
            for idx, sel in enumerate(selections, start=1):
                if not isinstance(sel, dict):
                    continue
                sid = sel.get("chosen_agent_id") or "flujo.builtins.stringify"
                params = sel.get("agent_params") or {}
                sname = sel.get("step_name") or f"Step {idx}"
                step_dict = {"kind": "step", "name": sname, "agent": {"id": sid, "params": params}}
                branches[f"branch_{idx}"] = [step_dict]

            parallel_dict: Dict[str, Any] = {
                "kind": "parallel",
                "name": "DoInParallel",
                "branches": branches,
            }
            steps_yaml.append(_yaml.safe_dump(parallel_dict, sort_keys=False).strip())
        else:
            # Linear steps
            for sel in selections:
                if not isinstance(sel, dict):
                    continue
                sid = sel.get("chosen_agent_id") or "flujo.builtins.stringify"
                params = sel.get("agent_params") or {}
                sname = sel.get("step_name") or "Step"
                step_dict = {"kind": "step", "name": sname, "agent": {"id": sid, "params": params}}
                steps_yaml.append(_yaml.safe_dump(step_dict, sort_keys=False).strip())

        if not steps_yaml:
            # Minimal scaffold
            yaml_text = f'version: "0.1"\nname: {name}\nsteps: []\n'
        else:
            steps_block = "\n".join(
                [
                    "- " + line if i == 0 else "  " + line
                    for block in steps_yaml
                    for i, line in enumerate(block.splitlines())
                ]
            )
            yaml_text = f'\nversion: "0.1"\nname: {name}\nsteps:\n{steps_block}\n'
        return {"generated_yaml": yaml_text}

    if reg.get("flujo.architect.yaml_writer") is None:
        reg.register(
            "flujo.architect.yaml_writer",
            lambda: _yaml_writer_agent,
            description="Agentic YAML writer: assembles pipeline.yaml from selections.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_goal": {"type": "string"},
                    "tool_selections": {"type": "array"},
                    "flujo_schema": {"type": "object"},
                },
                "required": ["tool_selections"],
            },
            output_schema={
                "type": "object",
                "properties": {"generated_yaml": {"type": "string"}},
                "required": ["generated_yaml"],
            },
            side_effects=False,
        )


try:
    _register_architect_agents()
except Exception:
    # Never fail module import due to registration
    pass


# Top-level utility: decide whether YAML exists in context for branch precheck
def has_yaml_key(_out: Any = None, ctx: DomainBaseModel | None = None, **_kwargs: Any) -> str:
    try:
        yt = getattr(ctx, "yaml_text", None)
    except Exception:
        yt = None
    present = isinstance(yt, str) and yt.strip() != ""
    return "present" if present else "absent"


class DiscoverSkillsAgent(AsyncAgentProtocol[Any, Dict[str, Any]]):
    """Builtin agent that discovers available skills and exposes them to context.

    - Loads skills from a local catalog (skills.yaml/skills.json) and Python entry points.
    - Returns a structure suitable for LLM tool matching steps.
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = directory or "."

    async def run(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        # Best-effort: load catalog + packaged entry points
        try:
            load_skills_catalog(self.directory)
            load_skills_entry_points()
        except Exception:
            # Non-fatal; continue with whatever is registered
            pass

        # Collect a public view of registered skills
        skills: List[Dict[str, Any]] = []
        try:
            reg = get_skill_registry()
            entries = getattr(reg, "_entries", {})  # Access internal map read-only
            for sid, meta in entries.items():
                skills.append(
                    {
                        "id": sid,
                        "description": meta.get("description"),
                        "input_schema": meta.get("input_schema"),
                    }
                )
        except Exception:
            # If registry access fails, return empty list
            skills = []

        return {"available_skills": skills}


# --- Adapter: extract decomposed steps into a flat context key ---
async def extract_decomposed_steps(
    decomposition: Any, *, output_key: str = "prepared_steps_for_mapping"
) -> Dict[str, Any]:
    """Adapter to extract a list of step dicts from the decomposer output.

    Returns a dict so that `updates_context: true` can merge it into the pipeline context.
    """
    steps: List[Dict[str, Any]] = []
    try:
        # Handle pydantic models with .model_dump()
        if isinstance(decomposition, PydanticBaseModel):
            try:
                raw = decomposition.model_dump()
            except Exception:
                raw = {}
            if isinstance(raw, dict):
                cand = raw.get("steps")
                if isinstance(cand, list):
                    steps = [x for x in cand if isinstance(x, dict)]
        # Handle plain dict
        elif isinstance(decomposition, dict):
            cand = decomposition.get("steps")
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
        # Handle object attribute access
        else:
            cand = getattr(decomposition, "steps", None)
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
    except Exception:
        steps = []

    return {output_key: steps}


# --- Adapter: extract YAML text from writer output ---
async def extract_yaml_text(writer_output: Any) -> Dict[str, str]:
    """
    Robustly extracts YAML text from various agent output formats,
    stores it in the context, and returns it as a dictionary.
    This function is the definitive bridge from YAML generation to the rest of the pipeline.
    """
    text: str | None = None
    try:
        # --- DEBUGGING: See exactly what we are receiving ---
        print(f"DEBUG [extract_yaml_text]: Received type: {type(writer_output)}")
        print(
            f"DEBUG [extract_yaml_text]: Received value (first 200 chars): {str(writer_output)[:200]}"
        )

        # --- EXTRACTION LOGIC ---
        # 1. Highest priority: Pydantic model-like object with attributes
        if hasattr(writer_output, "generated_yaml"):
            val = getattr(writer_output, "generated_yaml")
            if isinstance(val, (str, bytes)):
                text = val.decode() if isinstance(val, bytes) else val
        if text is None and hasattr(writer_output, "yaml_text"):
            val = getattr(writer_output, "yaml_text")
            if isinstance(val, (str, bytes)):
                text = val.decode() if isinstance(val, bytes) else val

        # 2. Fallback: Dictionary
        if text is None and isinstance(writer_output, dict):
            val = writer_output.get("generated_yaml") or writer_output.get("yaml_text")
            if isinstance(val, (str, bytes)):
                text = val.decode() if isinstance(val, bytes) else str(val)

        # 3. Fallback: Raw string or bytes - check if it's JSON first
        if text is None and isinstance(writer_output, (str, bytes)):
            raw_str = writer_output.decode() if isinstance(writer_output, bytes) else writer_output
            # Check if this looks like JSON
            if raw_str.strip().startswith("{") and raw_str.strip().endswith("}"):
                try:
                    import json

                    parsed = json.loads(raw_str)
                    if isinstance(parsed, dict):
                        val = parsed.get("generated_yaml") or parsed.get("yaml_text")
                        if isinstance(val, str):
                            text = val
                            print(
                                "DEBUG [extract_yaml_text]: Successfully parsed JSON and extracted YAML"
                            )
                except json.JSONDecodeError:
                    # Not valid JSON, treat as raw string
                    text = raw_str
            else:
                text = raw_str

        # 4. Last resort: Stringify the object and try to parse the YAML out of it
        if text is None:
            str_repr = str(writer_output)
            # Look for the YAML content inside a string like "YamlWriter(generated_yaml='...')""
            if "generated_yaml='" in str_repr:
                start = str_repr.find("generated_yaml='") + len("generated_yaml='")
                end = str_repr.rfind("'")
                if start < end:
                    text = str_repr[start:end]
            elif 'generated_yaml:"' in str_repr:  # Handle double quotes
                start = str_repr.find('generated_yaml:"') + len('generated_yaml:"')
                end = str_repr.rfind('"')
                if start < end:
                    text = str_repr[start:end]
    except Exception as e:
        print(f"DEBUG [extract_yaml_text]: Exception during extraction: {e}")
        text = str(writer_output)  # Fallback to string representation on error

    # --- CLEANUP and RETURN ---
    final_text = text or ""

    # Strip markdown fences just in case the LLM added them
    if "```" in final_text:
        import re

        match = re.search(r"```(?:yaml|yml)?\n(.*)\n```", final_text, re.DOTALL)
        if match:
            final_text = match.group(1).strip()

    # Final check to ensure we have something that looks like YAML
    if not ("version:" in final_text or "steps:" in final_text):
        print(
            "DEBUG [extract_yaml_text]: WARNING - Extracted text does not look like a valid Flujo YAML."
        )

    print(
        f"DEBUG [extract_yaml_text]: Successfully extracted YAML (first 100 chars): {final_text[:100]}"
    )

    return {"yaml_text": final_text, "generated_yaml": final_text}


# --- Adapter: capture ValidationReport for later error extraction ---
async def capture_validation_report(report: Any) -> Dict[str, Any]:
    """Capture the full ValidationReport in the context for later error extraction."""
    try:
        if hasattr(report, "model_dump"):
            report_dict = report.model_dump()
        elif isinstance(report, dict):
            report_dict = report
        else:
            report_dict = {}

        return {
            "validation_report": report_dict,
            "yaml_is_valid": bool(report_dict.get("is_valid", False)),
        }
    except Exception:
        return {"validation_report": {}, "yaml_is_valid": False}


# --- Adapter: turn ValidationReport into a boolean flag on context ---
async def validation_report_to_flag(report: Any) -> Dict[str, Any]:
    """Return a dict with yaml_is_valid based on a ValidationReport-like input."""
    try:
        if isinstance(report, dict):
            val = bool(report.get("is_valid", False) or report.get("yaml_is_valid", False))
        else:
            val = bool(getattr(report, "is_valid", False))
    except Exception:
        val = False
    # Return the flag in output so the immediate next conditional can read it
    return {"yaml_is_valid": val}


def exit_when_yaml_valid(_out: Any, context: Any | None) -> bool:
    """Exit when validation flag is present.

    Checks immediate output first (supports body steps returning the flag),
    then falls back to pipeline context.
    """
    try:
        if isinstance(_out, dict) and "yaml_is_valid" in _out:
            return bool(_out.get("yaml_is_valid", False))
    except Exception:
        pass
    try:
        return bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        try:
            if isinstance(context, dict):
                return bool(context.get("yaml_is_valid", False))
        except Exception:
            pass
        return False


# --- Adapter: extract validation errors for repair loop ---
async def extract_validation_errors(
    report: Any, *, context: DomainBaseModel | None = None
) -> Dict[str, Any]:
    """Extract error messages from a ValidationReport-like input for repair loops.

    Also returns the current yaml_is_valid flag (when available) so that the
    subsequent ValidityBranch can read a decisive signal from the immediate
    previous output.
    """
    errors: List[str] = []
    try:
        # Prioritize the full validation report from context, as the `report` arg might be a summarized flag.
        report_source = None
        if hasattr(context, "validation_report"):
            report_source = getattr(context, "validation_report")
        elif hasattr(context, "errors"):
            report_source = context
        else:
            report_source = report

        report_dict: Dict[str, Any]
        if isinstance(report_source, dict):
            report_dict = report_source
        elif (
            report_source is not None
            and hasattr(report_source, "model_dump")
            and callable(getattr(report_source, "model_dump"))
        ):
            report_dict = report_source.model_dump()
        else:
            report_dict = {}
        for finding in report_dict.get("errors", []) or []:
            msg = finding.get("message") if isinstance(finding, dict) else None
            if msg:
                errors.append(str(msg))
    except Exception:
        errors = []
    import json as _json

    # Respect the explicit validity flag from the context as the source of truth
    is_valid = False
    try:
        is_valid = bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        is_valid = False

    result: Dict[str, Any] = {
        "validation_errors": _json.dumps(errors),
        "yaml_is_valid": is_valid,
    }
    return result


# --- HITL helper: interpret user confirmation into branch key ---
async def check_user_confirmation(
    _out: Any = None,
    ctx: DomainBaseModel | None = None,
    *,
    user_input: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Map free-form user input to a conditional branch key.

    Accepts flexible calling conventions used by conditionals:
    - First positional `_out` as the previous step output
    - Keyword `user_input`
    - Ignores extra positional/context args

    Returns:
        "approved" for affirmative ("y", "yes", empty/whitespace), otherwise "denied".
    """
    # Resolve input text from explicit kwarg, previous output, or default
    text_val: Any = user_input if user_input is not None else _out
    try:
        text = "" if text_val is None else str(text_val)
    except Exception:
        text = ""
    norm = text.strip().lower()
    if norm == "":
        return "approved"
    if norm in {"y", "yes"}:
        return "approved"
    return "denied"


# Synchronous wrapper for conditional branching contexts (YAML 'condition')
def check_user_confirmation_sync(
    _out: Any = None,
    ctx: DomainBaseModel | None = None,
    *,
    user_input: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    try:
        text_val: Any = user_input if user_input is not None else _out
        text = "" if text_val is None else str(text_val)
    except Exception:
        text = ""
    norm = text.strip().lower()
    if norm == "":
        return "approved"
    if norm in {"y", "yes"}:
        return "approved"
    return "denied"


# --- Conditional key selector: 'valid' or 'invalid' based on context ---
def select_validity_branch(
    _out: Any = None,
    ctx: DomainBaseModel | None = None,
    **kwargs: Any,
) -> str:
    """Return 'valid' or 'invalid' using safe shape guard first, then explicit flags.

    Order:
    1) If YAML shape shows unmatched inline list on previous output or context, return 'invalid'
    2) If previous output dict carries 'yaml_is_valid', respect it
    3) Else if context.yaml_is_valid is present, respect it
    4) Else default to 'valid'
    """
    context = kwargs.get("context", ctx)

    def _shape_invalid(text: Any) -> bool:
        if not isinstance(text, str) or "steps:" not in text:
            return False
        try:
            line = text.split("steps:", 1)[1].splitlines()[0]
        except Exception:
            line = ""
        return ("[" in line and "]" not in line) and ("[]" not in line)

    try:
        val = None
        if isinstance(_out, dict) and "yaml_is_valid" in _out:
            try:
                val = bool(_out.get("yaml_is_valid"))
            except Exception:
                val = None
        print(
            "[SVB] out_type=",
            type(_out).__name__,
            " out_keys=",
            (list(_out.keys()) if isinstance(_out, dict) else None),
            " out_valid=",
            val,
            " ctx_flag=",
            (getattr(context, "yaml_is_valid", None) if context is not None else None),
            " ctx_has=",
            (hasattr(context, "yaml_is_valid") if context is not None else False),
            " ctx_type=",
            (type(context).__name__ if context is not None else None),
            " ctx_validation_report=",
            (hasattr(context, "validation_report") if context is not None else False),
        )
    except Exception:
        pass

    # 1) Early shape guard from previous output and context
    # Note: We don't discard _out here anymore, as the output flag should take priority
    try:
        if isinstance(_out, dict):
            yt0 = _out.get("yaml_text") or _out.get("generated_yaml")
            if _shape_invalid(yt0):
                return "invalid"
        elif isinstance(_out, str) and _shape_invalid(_out):
            return "invalid"
    except Exception:
        pass
    try:
        yt_ctx = getattr(context, "yaml_text", None)
        if _shape_invalid(yt_ctx):
            return "invalid"
    except Exception:
        pass

    # 2) Previous output signal
    try:
        if isinstance(_out, dict) and "yaml_is_valid" in _out:
            return "valid" if bool(_out.get("yaml_is_valid")) else "invalid"
    except Exception:
        pass

    # 3) Context flag
    try:
        if hasattr(context, "yaml_is_valid"):
            return "valid" if bool(getattr(context, "yaml_is_valid")) else "invalid"
    except Exception:
        pass
    if isinstance(context, dict) and "yaml_is_valid" in context:
        return "valid" if bool(context.get("yaml_is_valid")) else "invalid"

    # 4) Default to valid
    return "valid"


# --- Compute branch key from context validity (top-level importable) ---
async def compute_validity_key(_x: Any = None, *, context: DomainBaseModel | None = None) -> str:
    try:
        val = bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        try:
            val = bool(context.get("yaml_is_valid", False)) if isinstance(context, dict) else False
        except Exception:
            val = False
    return "valid" if val else "invalid"


def select_by_yaml_shape(
    _out: Any = None,
    ctx: DomainBaseModel | None = None,
    **kwargs: Any,
) -> str:
    """Return 'invalid' only for unmatched inline list on steps:

    Accepts both positional (output, context) and kw-only 'context'.
    """
    context = kwargs.get("context", ctx)
    """Heuristic selector to catch a very specific malformed YAML shape.

    - Returns 'invalid' only when the line after 'steps:' contains an opening '['
      without a matching closing ']' (e.g., "steps: ["), which is a common
      transient error pattern in early drafts.
    - Treats "steps: []" and other balanced inline lists as valid; also treats
    - normal block lists as valid.
    - Falls back to checking context.yaml_is_valid when available.
    """

    def _eval(text: str) -> str:
        parts = text.split("steps:", 1)
        if len(parts) == 2:
            line = parts[1].splitlines()[0]
            if "[" in line and "]" not in line and "[]" not in line:
                return "invalid"
        return "valid"

    try:
        prev = None
        if isinstance(_out, dict):
            prev = _out.get("yaml_text") or _out.get("generated_yaml")
        elif isinstance(_out, str):
            prev = _out
        ctx_text = None
        try:
            ctx_text = getattr(context, "yaml_text", None)
        except Exception:
            ctx_text = None
        print(
            "[SBYS] prev_is_dict=",
            isinstance(_out, dict),
            "ctx_flag=",
            (getattr(context, "yaml_is_valid", None) if context is not None else None),
            "prev_head=",
            (str(prev)[:30] if isinstance(prev, str) else None),
            "ctx_head=",
            (str(ctx_text)[:30] if isinstance(ctx_text, str) else None),
        )
    except Exception:
        pass

    try:
        if isinstance(_out, dict):
            val = _out.get("yaml_text") or _out.get("generated_yaml")
            if isinstance(val, str):
                res = _eval(val)
                if res == "invalid":
                    return res
        elif isinstance(_out, str):
            res = _eval(_out)
            if res == "invalid":
                return res
    except Exception:
        pass

    # Evaluate context.yaml_text shape next; this is authoritative for YAML shape
    try:
        yt_ctx = getattr(context, "yaml_text", None)
        if isinstance(yt_ctx, str):
            res = _eval(yt_ctx)
            if res == "invalid":
                return res
            # If shape looks valid, prefer 'valid' without relying on flags
            return "valid"
    except Exception:
        pass

    # Respect explicit context validity when provided
    try:
        if hasattr(context, "yaml_is_valid"):
            return "valid" if bool(getattr(context, "yaml_is_valid")) else "invalid"
    except Exception:
        pass
    if isinstance(context, dict) and "yaml_is_valid" in context:
        return "valid" if bool(context.get("yaml_is_valid")) else "invalid"

    # Fallback to context.yaml_text heuristic
    try:
        yt = getattr(context, "yaml_text", None)
        if isinstance(yt, str):
            res = _eval(yt)
            if res == "invalid":
                return res
    except Exception:
        pass
    return "valid"


async def shape_to_validity_flag(*, context: DomainBaseModel | None = None) -> Dict[str, Any]:
    """Return {'yaml_is_valid': bool} based on a quick YAML shape heuristic.

    - False only when the 'steps:' line contains an opening '[' without a closing ']'.
    - True otherwise. This does not replace the validator; it just seeds a sensible default
      for the immediate conditional branch when validator behavior is mocked.
    """
    try:
        yt = getattr(context, "yaml_text", None)
    except Exception:
        yt = None
    if isinstance(yt, str) and "steps:" in yt:
        try:
            line = yt.split("steps:", 1)[1].splitlines()[0]
        except Exception:
            line = ""
        if "[" in line and "]" not in line:
            _out: Dict[str, Any] = {"yaml_is_valid": False, "yaml_text": yt}
            try:
                gy = getattr(context, "generated_yaml", None)
                if isinstance(gy, str):
                    _out["generated_yaml"] = gy
            except Exception:
                pass
            return _out
    # If we have any YAML-like structure with balanced inline list or block lists, treat as valid
    if isinstance(yt, str):
        try:
            after = yt.split("steps:", 1)[1]
            first = after.splitlines()[0]
            if (
                "[]" in first
                or ("[" in first and "]" in first)
                or not ("[" in first and "]" not in first)
            ):
                _out2: Dict[str, Any] = {"yaml_is_valid": True, "yaml_text": yt}
                try:
                    gy = getattr(context, "generated_yaml", None)
                    if isinstance(gy, str):
                        _out2["generated_yaml"] = gy
                except Exception:
                    pass
                return _out2
        except Exception:
            pass
    _out3: Dict[str, Any] = {"yaml_is_valid": True}
    if isinstance(yt, str):
        _out3["yaml_text"] = yt
    try:
        gy = getattr(context, "generated_yaml", None)
        if isinstance(gy, str):
            _out3["generated_yaml"] = gy
    except Exception:
        pass
    return _out3


def always_valid_key(_out: Any = None, ctx: DomainBaseModel | None = None) -> str:
    """Return 'valid' unconditionally (used after successful repair)."""
    return "valid"


# --- In-memory YAML validation skill ---
async def validate_yaml(yaml_text: str, base_dir: Optional[str] = None) -> Any:
    """Validate a YAML blueprint string and return a ValidationReport.

    Never raises for invalid YAML; returns a report with an error finding instead.
    """
    try:
        from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir)
        # Let the pipeline perform its deeper graph validation
        if pipeline is not None:
            return pipeline.validate_graph()
        else:
            # Handle case where pipeline loading returns None
            from flujo.domain.pipeline_validation import (
                ValidationReport as _VR,
                ValidationFinding as _VF,
            )

            return _VR(
                errors=[
                    _VF(
                        rule_id="YAML-LOAD",
                        severity="error",
                        message="Pipeline loading returned None",
                    ),
                ],
                warnings=[],
            )
    except Exception as e:
        try:
            # Construct a report capturing the parse/compile error
            from flujo.domain.pipeline_validation import (
                ValidationReport as _VR,
                ValidationFinding as _VF,
            )

            return _VR(
                errors=[
                    _VF(rule_id="YAML-PARSE", severity="error", message=str(e)),
                ],
                warnings=[],
            )
        except Exception:
            # Absolute fallback: minimal dict compatible with adapters/predicates
            return {"is_valid": False, "errors": [str(e)], "warnings": []}


# --- Passthrough adapter (identity) ---
async def passthrough(x: Any) -> Any:
    """Return the input unchanged (identity)."""
    return x


async def repair_yaml_ruamel(yaml_text: str) -> Dict[str, Any]:
    """Conservatively attempt to repair malformed pipeline YAML text.

    Strategy:
    - Heuristic fix: if the line after 'steps:' contains an unmatched '[', rewrite as 'steps: []'.
    - If ruamel.yaml is available, perform a round-trip load/dump to normalize formatting while
      preserving quotes and structure. Ensure top-level keys like version exist.
    - Always return a mapping containing both 'generated_yaml' and 'yaml_text'.
    """
    # Normalize input
    text: str = yaml_text or ""

    # Heuristic patch for common malformed inline list after 'steps:'
    try:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("steps:"):
                tail = line.split("steps:", 1)[1]
                if "[" in tail and "]" not in tail:
                    # Preserve indentation when replacing the line
                    indent = line[: len(line) - len(line.lstrip(" "))]
                    lines[i] = f"{indent}steps: []"
                    text = "\n".join(lines)
                    break
    except Exception:
        # Best-effort heuristic; ignore failures and continue with original text
        pass

    # Round-trip parse and dump if ruamel is available to normalize YAML
    if _ruamel_yaml is not None:
        try:
            yaml = _ruamel_yaml.YAML()
            yaml.preserve_quotes = True
            data = yaml.load(text)
            # Ensure minimal keys and sane defaults
            if isinstance(data, dict):
                if "version" not in data:
                    data["version"] = "0.1"
                # Add a default pipeline name when missing to satisfy validators
                if "name" not in data:
                    data["name"] = "generated_pipeline"
                if "steps" in data and data["steps"] is None:
                    data["steps"] = []
            from io import StringIO

            buf = StringIO()
            yaml.dump(data, buf)
            fixed = buf.getvalue()
            return {"generated_yaml": fixed, "yaml_text": fixed}
        except Exception:
            # Fall through to return (possibly heuristically patched) text
            pass

    # Fallback: return text as-is (possibly after heuristic correction)
    return {"generated_yaml": text, "yaml_text": text}


# --- Adapter: return YAML in CLI-expected format ---
async def return_yaml_for_cli(yaml_text: Any) -> Dict[str, str]:
    """Return YAML in the format that the CLI expects to find, extracting it if necessary.

    Defensive against LLMs that occasionally prepend prose before the YAML. If any
    non-YAML text precedes the first YAML key, trim everything before the first line
    starting with 'version:' or 'name:'.
    """
    import re

    # Handle dictionary input (from extract_yaml_text)
    if isinstance(yaml_text, dict):
        yaml_string = (
            yaml_text.get("yaml_text") or yaml_text.get("generated_yaml") or str(yaml_text)
        )
    else:
        yaml_string = str(yaml_text)

    # Defensive extraction: find the start of the YAML content
    try:
        match = re.search(r"^(version:|name:)", yaml_string, re.MULTILINE)
        if match:
            yaml_string = yaml_string[match.start() :]
    except Exception:
        # Best-effort; keep original string on regex failure
        pass

    return {"generated_yaml": yaml_string, "yaml_text": yaml_string}


# --- Welcome agent for new users ---
async def welcome_agent(name: str = "Developer") -> str:
    """
    Return a welcome message, optionally with ASCII art when pyfiglet is available.

    Gracefully degrades to plain text when pyfiglet (optional) is not installed
    or if the configured font is unavailable.
    """
    welcome_header = f"Welcome, {name}!"

    flujo_art = ""
    if _pyfiglet:
        try:
            fig = _pyfiglet.Figlet(font="slant")
            flujo_art = fig.renderText("Flujo")
        except Exception:
            # Fallback to plain text if font is missing or another error occurs
            flujo_art = "F L U J O\n"

    welcome_body = (
        "\nYou have successfully run your first pipeline!\n\n"
        "This is a simple workflow defined in `pipeline.yaml`.\n"
        "You can edit it or create a new one from scratch by running:\n\n"
        '  flujo create --goal "Your new workflow goal"\n\n'
        "Happy building!\n"
    )

    return f"{welcome_header}\n\n{flujo_art}{welcome_body}"


def _register_builtins() -> None:
    """Register builtin skills with the global registry."""
    try:
        reg = get_skill_registry()

        # Welcome experience for new users
        reg.register(
            "flujo.builtins.welcome_agent",
            lambda **_params: welcome_agent,
            description="Returns a fun welcome message for new users.",
            arg_schema={
                "type": "object",
                "properties": {"name": {"type": "string", "default": "Developer"}},
            },
            side_effects=False,
        )
        # Factory accepts params to match YAML 'agent: { id: ..., params: {...} }'
        reg.register(
            "flujo.builtins.discover_skills",
            lambda directory=".": DiscoverSkillsAgent(directory=directory),
            description="Discover local and packaged skills; returns available_skills list.",
        )
        # Adapter function: return the async callable without invoking it
        # Loader will call this factory with params (none by default) and expect an agent object
        reg.register(
            "flujo.builtins.extract_decomposed_steps",
            # Factory returns the coroutine function itself so Step.from_callable can wrap it
            lambda **_params: extract_decomposed_steps,
            description=(
                "Extract list of step dicts from decomposer output into 'prepared_steps_for_mapping'"
            ),
        )
        # Adapter extractor for YAML string
        reg.register(
            "flujo.builtins.extract_yaml_text",
            lambda **_params: extract_yaml_text,
            description="Extract YAML string from YamlWriter output object or dict.",
        )

        # --- FSD: Built-in data transforms (M1)
        # to_csv: list[dict] -> CSV string
        async def to_csv(rows: Any, *, headers: Optional[List[str]] = None) -> str:
            import io
            import csv

            # Normalize input to list[dict[str, Any]] where possible
            norm: List[Dict[str, Any]]
            if isinstance(rows, dict):
                norm = [rows]
            elif isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
                norm = rows
            else:
                # Best-effort: coerce to list of dicts using a single 'value' column
                if isinstance(rows, list):
                    norm = [x if isinstance(x, dict) else {"value": x} for x in rows]
                else:
                    norm = [rows if isinstance(rows, dict) else {"value": rows}]

            # Determine headers deterministically
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
                    # Skip malformed rows defensively
                    continue
            return buf.getvalue()

        reg.register(
            "flujo.builtins.to_csv",
            lambda **_params: to_csv,
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

        # aggregate: sum/avg/count over numeric field in list[dict]
        async def aggregate(
            data: Any,
            *,
            operation: str,
            field: Optional[str] = None,
        ) -> float | int:
            op = (operation or "").strip().lower()
            items: List[Dict[str, Any]]
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                # Not a supported structure
                items = []

            def _nums() -> List[float]:
                out: List[float] = []
                if not field:
                    return out
                for obj in items:
                    try:
                        # At this point, field is guaranteed non-None by the guard above
                        fld = cast(str, field)
                        val = obj.get(fld)
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

            # Unknown operation -> 0
            return 0

        reg.register(
            "flujo.builtins.aggregate",
            lambda **_params: aggregate,
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

        # select_fields: projection/rename over dict or list[dict]
        async def select_fields(
            data: Any,
            *,
            include: Optional[List[str]] = None,
            rename: Optional[Dict[str, str]] = None,
        ) -> Any:
            includes = list(include) if include else None
            ren = dict(rename) if rename else {}

            def _project(obj: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    keys = list(obj.keys()) if includes is None else [k for k in includes]
                    out: Dict[str, Any] = {}
                    for k in keys:
                        if k in obj:
                            out[ren.get(k, k)] = obj.get(k)
                    # If only rename provided without include, also consider renamed-only keys present
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
            # Unsupported structure: return as-is
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
                    "rename": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["data"],
            },
            side_effects=False,
        )

        # flatten: list[list[T]] -> list[T]
        async def flatten(items: Any) -> List[Any]:
            if not isinstance(items, list):
                return []
            out: List[Any] = []
            for sub in items:
                if isinstance(sub, list):
                    out.extend(sub)
                elif isinstance(sub, tuple):
                    out.extend(list(sub))
                else:
                    # Be permissive: include non-list items as-is
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

        # Return YAML in CLI-expected format
        reg.register(
            "flujo.builtins.return_yaml_for_cli",
            lambda **_params: return_yaml_for_cli,
            description="Return YAML in the format that the CLI expects to find (with generated_yaml and yaml_text keys).",
        )

        # --- FSD-024: analyze_project (safe filesystem scan)
        async def analyze_project(
            _data: Any = None, *, directory: str = ".", max_files: int = 200
        ) -> Dict[str, Any]:
            import os

            try:
                files: list[str] = []
                for root, dirs, fnames in os.walk(directory):
                    # Limit depth to top 2 levels
                    depth = os.path.relpath(root, directory).count(os.sep)
                    if depth > 1:
                        dirs[:] = []
                    for f in fnames:
                        if len(files) >= int(max_files):
                            break
                        files.append(os.path.relpath(os.path.join(root, f), directory))
                detected: list[str] = []
                s = set(files)
                for mark in ("requirements.txt", "pyproject.toml", "flujo.toml", "pipeline.yaml"):
                    if any(p.endswith(mark) for p in s):
                        detected.append(mark)
                return {
                    "project_summary": f"Found {len(files)} files. Detected: "
                    + (", ".join(detected) if detected else "none")
                }
            except Exception:
                return {"project_summary": "Error analyzing project"}

        def _make_analyze_runner(directory: str = ".", max_files: int = 200) -> Any:
            async def _runner(_data: Any = None, **_k: Any) -> Dict[str, Any]:
                return await analyze_project(_data, directory=directory, max_files=max_files)

            return _runner

        reg.register(
            "flujo.builtins.analyze_project",
            _make_analyze_runner,
            description="Scan project tree to produce a short summary (no network).",
            arg_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "max_files": {"type": "integer", "default": 200},
                },
            },
            side_effects=False,
        )

        # --- FSD-024: visualize_plan -> Mermaid
        async def visualize_plan(plan: Any) -> Dict[str, str]:
            try:
                lines: list[str] = ["graph TD"]
                if isinstance(plan, list):
                    for i, step in enumerate(plan, start=1):
                        label = None
                        if isinstance(step, dict):
                            label = step.get("name") or step.get("id") or f"Step {i}"
                        else:
                            label = getattr(step, "name", None) or f"Step {i}"
                        lines.append(f'  S{i}["{str(label)}"]')
                        if i > 1:
                            lines.append(f"  S{i - 1} --> S{i}")
                return {"plan_mermaid_graph": "\n".join(lines)}
            except Exception:
                return {"plan_mermaid_graph": 'graph TD\n  S1["Plan unavailable"]'}

        reg.register(
            "flujo.builtins.visualize_plan",
            lambda **_params: visualize_plan,
            description="Render a simple Mermaid graph for a linear plan.",
            side_effects=False,
        )

        # --- FSD-024: estimate_plan_cost (sum registry est_cost)
        async def estimate_plan_cost(plan: Any) -> Dict[str, float]:
            total = 0.0
            try:
                registry = get_skill_registry()
                if isinstance(plan, list):
                    for step in plan:
                        sid = None
                        if isinstance(step, dict):
                            agent = step.get("agent")
                            if isinstance(agent, dict):
                                sid = agent.get("id")
                        if isinstance(sid, str):
                            entry = registry.get(sid) or {}
                            try:
                                total += float(entry.get("est_cost", 0.0))
                            except Exception:
                                pass
            except Exception:
                total = 0.0
            return {"plan_estimated_cost_usd": round(float(total), 4)}

        reg.register(
            "flujo.builtins.estimate_plan_cost",
            lambda **_params: estimate_plan_cost,
            description="Estimate cost by summing est_cost metadata for referenced skills.",
            side_effects=False,
        )

        # --- FSD-024: run_pipeline_in_memory (safe, mocks side effects)
        async def run_pipeline_in_memory(
            yaml_text: str,
            input_text: str = "",
            sandbox: bool = True,
            base_dir: Optional[str] = None,
        ) -> Dict[str, Any]:
            from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
            from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling
            from flujo.infra.skill_registry import get_skill_registry as _get
            from typing import Any as _Any
            import os as _os
            import asyncio as _asyncio

            reg_local = _get()
            restore: dict[str, dict[str, Any]] = {}
            mutated = False
            # Identify and (optionally) mock side-effect skills referenced in YAML
            if sandbox:
                try:
                    from flujo.cli.helpers import find_side_effect_skills_in_yaml as _find

                    side_ids = _find(yaml_text)
                except Exception:
                    side_ids = []
            else:
                side_ids = []
            try:
                if sandbox and side_ids:
                    for sid in side_ids:
                        entry = reg_local.get(sid)
                        if not entry:
                            continue
                        restore[sid] = dict(entry)

                        def _make_factory(
                            _sid: str,
                        ) -> Callable[..., Callable[..., Awaitable[Dict[str, _Any]]]]:
                            async def _mock(*_a: _Any, **_k: _Any) -> Dict[str, _Any]:
                                return {"mocked": True, "skill": _sid}

                            return lambda **_p: _mock

                        entry["factory"] = _make_factory(sid)
                        entry["side_effects"] = False
                        mutated = True

                # Compile blueprint with base_dir for correct relative resolution
                _base = base_dir or _os.getcwd()
                pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=_base)
                runner = create_flujo_runner(pipeline, None, {"initial_prompt": input_text})

                # Execute synchronously via a worker thread to avoid blocking the event loop
                def _run_sync() -> Any:
                    return execute_pipeline_with_output_handling(runner, input_text, None, False)

                result = await _asyncio.to_thread(_run_sync)
                return {"dry_run_result": result}
            finally:
                if mutated:
                    try:
                        for sid, entry in restore.items():
                            reg_local._entries[sid] = entry
                    except Exception:
                        pass

        reg.register(
            "flujo.builtins.run_pipeline_in_memory",
            lambda **_params: run_pipeline_in_memory,
            description="Compile and run a YAML pipeline in-memory, mocking side-effect skills.",
            arg_schema={
                "type": "object",
                "properties": {
                    "yaml_text": {"type": "string"},
                    "input_text": {"type": "string", "default": ""},
                    "sandbox": {"type": "boolean", "default": True},
                    "base_dir": {"type": "string"},
                },
                "required": ["yaml_text"],
            },
            side_effects=False,
        )

        reg.register(
            "flujo.builtins.capture_validation_report",
            lambda **_params: capture_validation_report,
            description="Capture full validation report in context for later error extraction.",
        )
        reg.register(
            "flujo.builtins.validation_report_to_flag",
            lambda **_params: validation_report_to_flag,
            description="Map validation report to {'yaml_is_valid': bool} and update context.",
        )
        reg.register(
            "flujo.builtins.select_validity_branch",
            lambda **_params: select_validity_branch,
            description="Return 'valid' if context.yaml_is_valid else 'invalid'.",
            side_effects=False,
        )
        reg.register(
            "flujo.builtins.select_by_yaml_shape",
            lambda **_params: select_by_yaml_shape,
            description="Return 'invalid' when context.yaml_text uses inline list for steps, else 'valid'.",
            side_effects=False,
        )
        reg.register(
            "flujo.builtins.extract_validation_errors",
            lambda **_params: extract_validation_errors,
            description="Extract error messages from a validation report into context.validation_errors.",
            side_effects=False,
        )
        # Heuristic flagger to seed validity for the subsequent conditional
        reg.register(
            "flujo.builtins.shape_to_validity_flag",
            lambda **_params: shape_to_validity_flag,
            description=(
                "Return {'yaml_is_valid': bool} based on simple inline-list shape after 'steps:'"
            ),
            side_effects=False,
        )

        # HITL: prompt user for approval
        async def ask_user(question: Optional[str] = None) -> str:
            """Ask the user for input, with non-interactive fallback.

            Behavior:
            - If stdin is non-interactive (e.g., piped input), return the provided
              value directly without prompting. This enables CLI usage like:
                  echo "goal" | flujo run pipeline.yaml
            - Otherwise, prompt interactively using the question (or a default).
            """
            try:
                import sys as _sys

                # Non-interactive: treat provided value as the answer and do not prompt
                if not _sys.stdin.isatty():
                    return str(question or "").strip()

                import typer as _typer

                q = question or "Does this plan look correct? (Y/n)"
                resp = _typer.prompt(q, default="Y")
                return str(resp)
            except Exception:
                # Conservative fallback to an affirmative response to avoid breaking flows
                return "Y"

        reg.register(
            "flujo.builtins.ask_user",
            lambda **_params: ask_user,
            description=("Prompt user and return raw response string."),
            arg_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
            },
            side_effects=False,
        )
        reg.register(
            "flujo.builtins.always_valid_key",
            lambda **_params: always_valid_key,
            description="Return 'valid' unconditionally for post-repair branch logging.",
            side_effects=False,
        )

        # --- Optional utility: render_jinja_template ---
        async def render_jinja_template(
            template: str, variables: Dict[str, Any] | None = None
        ) -> str:
            """Render a Jinja2 template string with provided variables.

            - Uses StrictUndefined to surface missing variables during development.
            - Falls back gracefully when Jinja2 is missing by returning the input unchanged.
            """
            if _jinja2 is None:
                return template
            try:
                env = _jinja2.Environment(undefined=_jinja2.StrictUndefined, autoescape=False)
                tmpl = env.from_string(template)
                result = tmpl.render(**(variables or {}))
                return str(result)  # Ensure we return a string
            except Exception:
                # Do not raise in CLI flows; return original to avoid breaking pipelines
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

        # Simple adapter: stringify any object (useful to bridge model outputs to HITL)
        async def stringify(x: Any) -> str:
            try:
                return str(x)
            except Exception:
                return ""

        reg.register(
            "flujo.builtins.stringify",
            lambda **_params: stringify,
            description="Convert any input value to a string via str(x).",
            side_effects=False,
        )

        # Introspect registered framework step primitives and produce JSON Schemas
        async def get_framework_schema() -> Dict[str, Any]:
            try:
                import flujo.framework as _fw  # noqa: F401
                from flujo.framework.registry import (
                    get_registered_step_kinds,
                    register_step_type as _reg_step,
                    register_policy as _reg_policy,
                )

                mapping = get_registered_step_kinds()
                if not mapping:
                    # Try explicit registration call if exposed
                    try:
                        import flujo.framework as _framework_mod

                        if hasattr(_framework_mod, "_register_core_primitives"):
                            _framework_mod._register_core_primitives()
                            mapping = get_registered_step_kinds()
                    except Exception:
                        pass
                if not mapping:
                    # Fallback: force-register StateMachine explicitly
                    try:
                        from flujo.domain.dsl.state_machine import StateMachineStep as _SM
                        from flujo.application.core.step_policies import (
                            StateMachinePolicyExecutor as _SMPol,
                        )

                        _reg_step(_SM)
                        _reg_policy(_SM, _SMPol())
                        mapping = get_registered_step_kinds()
                    except Exception:
                        pass
            except Exception:
                mapping = {}
            schemas: Dict[str, Any] = {}
            for kind, cls in mapping.items():
                try:
                    if hasattr(cls, "model_json_schema") and callable(
                        getattr(cls, "model_json_schema")
                    ):
                        # Create a JSON schema-compatible version by excluding non-serializable fields
                        # The issue is that Step classes have fields like ValidationPlugin, Callable, etc.
                        # that cannot be converted to JSON schema
                        try:
                            # First try the standard approach
                            schemas[kind] = cls.model_json_schema()
                        except Exception:
                            # If that fails, create a simplified schema with only the essential fields
                            # This is a fallback for complex step types that have non-serializable fields
                            if "StateMachine" in kind:
                                # For StateMachine, create a simplified schema with only the essential fields
                                simplified_schema = {
                                    "type": "object",
                                    "title": f"{kind}",
                                    "properties": {
                                        "name": {"type": "string", "title": "Name"},
                                        "kind": {"type": "string", "const": kind, "title": "Kind"},
                                        "states": {
                                            "type": "object",
                                            "title": "States",
                                            "description": "Map of state name to Pipeline configuration",
                                        },
                                        "start_state": {"type": "string", "title": "Start State"},
                                        "end_states": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "title": "End States",
                                        },
                                    },
                                    "required": ["name", "start_state"],
                                }
                                schemas[kind] = simplified_schema
                            else:
                                # For other step types, create a basic schema
                                schemas[kind] = {
                                    "type": "object",
                                    "title": f"{kind}",
                                    "properties": {
                                        "name": {"type": "string", "title": "Name"},
                                        "kind": {"type": "string", "const": kind, "title": "Kind"},
                                    },
                                    "required": ["name"],
                                }
                except Exception as e:
                    # Log the error for debugging but continue with other step types
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to generate schema for {kind}: {e}"
                    )
                    continue
            return {"steps": schemas}

        reg.register(
            "flujo.builtins.get_framework_schema",
            lambda **_params: get_framework_schema,
            description=("Return JSON Schemas for registered framework steps (by kind)."),
            side_effects=False,
        )

        reg.register(
            "flujo.builtins.repair_yaml_ruamel",
            lambda **_params: repair_yaml_ruamel,
            description="Conservatively repair YAML text via ruamel.yaml round-trip load/dump.",
            arg_schema={
                "type": "object",
                "properties": {"yaml_text": {"type": "string"}},
                "required": ["yaml_text"],
            },
            side_effects=False,
        )

        # Emit current YAML validity flag from context for loop exit to read
        async def get_yaml_validity(*, context: DomainBaseModel | None = None) -> Dict[str, Any]:
            try:
                val = bool(getattr(context, "yaml_is_valid", False))
            except Exception:
                try:
                    val = (
                        bool(context.get("yaml_is_valid", False))
                        if isinstance(context, dict)
                        else False
                    )
                except Exception:
                    val = False
            return {"yaml_is_valid": val}

        reg.register(
            "flujo.builtins.get_yaml_validity",
            lambda **_params: get_yaml_validity,
            description="Return {'yaml_is_valid': <bool>} from the current context.",
            side_effects=False,
        )

        reg.register(
            "flujo.builtins.compute_validity_key",
            lambda **_params: compute_validity_key,
            description="Return 'valid' or 'invalid' based on context.yaml_is_valid.",
            side_effects=False,
        )

        # Decide whether YAML already exists in context
        async def has_yaml_key(*, context: DomainBaseModel | None = None) -> str:
            try:
                yt = getattr(context, "yaml_text", None)
            except Exception:
                yt = None
            present = isinstance(yt, str) and yt.strip() != ""
            return "present" if present else "absent"

        reg.register(
            "flujo.builtins.has_yaml_key",
            lambda **_params: has_yaml_key,
            description="Return 'present' if context.yaml_text is a non-empty string, else 'absent'.",
            side_effects=False,
        )
        # Human-in-the-loop confirmation interpreter
        reg.register(
            "flujo.builtins.check_user_confirmation",
            lambda **_params: check_user_confirmation,
            description=(
                "Interpret user input as 'approved' when affirmative (y/yes/empty), else 'denied'."
            ),
            arg_schema={
                "type": "object",
                "properties": {"user_input": {"type": "string"}},
                "required": ["user_input"],
            },
            side_effects=False,
        )
        # Identity adapter useful in conditional valid branches
        reg.register(
            "flujo.builtins.passthrough",
            lambda **_params: passthrough,
            description="Identity adapter that returns input unchanged.",
            side_effects=False,
        )

        # In-memory YAML validation that returns a ValidationReport and never raises on invalid YAML
        def _resolve_validate_yaml(**_params: Any) -> Any:
            # Dynamic resolution so test monkeypatches to flujo.builtins.validate_yaml take effect
            try:
                import importlib as _importlib

                mod = _importlib.import_module("flujo.builtins")
                return getattr(mod, "validate_yaml")
            except Exception:
                return validate_yaml

        reg.register(
            "flujo.builtins.validate_yaml",
            _resolve_validate_yaml,
            description=(
                "Validate YAML blueprint text in-memory; returns ValidationReport without raising."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "yaml_text": {"type": "string"},
                    "base_dir": {"type": "string"},
                },
                "required": ["yaml_text"],
            },
            side_effects=False,
        )

        # Aggregator: combine mapped results with goal and (optional) skills
        async def aggregate_plan(
            mapped_step_results: Any, *, context: DomainBaseModel | None = None
        ) -> Dict[str, Any]:
            try:
                user_goal = getattr(context, "user_goal", None) or getattr(
                    context, "initial_prompt", None
                )
            except Exception:
                user_goal = None
            # Normalize list of results into list of dicts
            plans: List[Dict[str, Any]] = []
            try:
                if isinstance(mapped_step_results, list):
                    for item in mapped_step_results:
                        if isinstance(item, dict):
                            plans.append(item)
                        elif hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
                            try:
                                plans.append(item.model_dump())
                            except Exception:
                                pass
                        else:
                            try:
                                plans.append(dict(item))
                            except Exception:
                                pass
            except Exception:
                plans = []

            skills: List[Dict[str, Any]] = []
            try:
                maybe = getattr(context, "available_skills", None)
                if isinstance(maybe, list):
                    skills = [x for x in maybe if isinstance(x, dict)]
            except Exception:
                pass

            return {
                "user_goal": user_goal or "",
                "step_plans": plans,
                "available_skills": skills,
            }

        reg.register(
            "flujo.builtins.aggregate_plan",
            lambda **_params: aggregate_plan,
            description="Aggregate mapped tool decisions and goal for YAML writer.",
        )

        # Adapter: build input for tool matcher from a step item and context skills
        async def build_tool_match_input(
            item: Any, *, context: DomainBaseModel | None = None
        ) -> Dict[str, Any]:
            name = None
            purpose = None
            try:
                if isinstance(item, dict):
                    name = item.get("step_name") or item.get("name") or item.get("title")
                    purpose = item.get("purpose") or item.get("description")
                else:
                    name = getattr(item, "step_name", None) or getattr(item, "name", None)
                    purpose = getattr(item, "purpose", None) or getattr(item, "description", None)
            except Exception:
                name = None
                purpose = None

            try:
                maybe_skills = getattr(context, "available_skills", None)
                skills = (
                    [x for x in maybe_skills if isinstance(x, dict)]
                    if isinstance(maybe_skills, list)
                    else []
                )
            except Exception:
                skills = []

            return {
                "step_name": str(name or ""),
                "purpose": str(purpose or ""),
                "available_skills": skills,
            }

        reg.register(
            "flujo.builtins.build_tool_match_input",
            lambda **_params: build_tool_match_input,
            description=(
                "Construct {step_name, purpose, available_skills} for the tool matcher from a step item."
            ),
        )

        # --- Killer Demo: web_search ---
        async def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
            """Perform a DuckDuckGo web search (top N simplified results).

            Returns a list of {title, link, snippet} dicts.
            """
            if _DDGSAsync is None and _DDGS_CLASS is None:
                # Graceful degrade if optional dependency not installed
                return []

            results: List[Dict[str, Any]] = []
            try:
                if _DDGSAsync is not None:
                    # Use async client when available
                    async with _DDGSAsync() as ddgs:
                        agen = None
                        try:
                            agen = ddgs.text(query, max_results=max_results)  # duckduckgo_search
                        except Exception:
                            try:
                                agen = ddgs.atext(query, max_results=max_results)  # ddgs
                            except Exception:
                                agen = None
                        if agen is not None:
                            async for r in agen:
                                if isinstance(r, dict):
                                    results.append(r)
                else:
                    # Use DDGS in a thread pool since sync
                    import asyncio
                    from concurrent.futures import ThreadPoolExecutor

                    def _search_sync() -> List[Dict[str, Any]]:
                        assert _DDGS_CLASS is not None
                        ddgs = _DDGS_CLASS()
                        search_results: List[Dict[str, Any]] = []
                        try:
                            iterable = ddgs.text(query, max_results=max_results)
                        except Exception:
                            # Some versions expect 'max_results' or 'max_results' under different name (e.g., 'max_results')
                            iterable = ddgs.text(query, max_results=max_results)
                        for r in iterable:
                            if isinstance(r, dict):
                                search_results.append(r)
                        return search_results

                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        results = await loop.run_in_executor(executor, _search_sync)
            except Exception:
                # Non-fatal: return empty results on any search error
                return []

            simplified: List[Dict[str, Any]] = []
            for item in results:
                try:
                    title = item.get("title") if isinstance(item, dict) else None
                    link = None
                    snippet = None
                    if isinstance(item, dict):
                        # Support both ddgs and duckduckgo_search field names
                        link = item.get("href") or item.get("link")
                        snippet = item.get("body") or item.get("snippet")
                    simplified.append({"title": title, "link": link, "snippet": snippet})
                except Exception:
                    continue
            return simplified

        reg.register(
            "flujo.builtins.web_search",
            lambda **_params: web_search,
            description=(
                "Performs a web search and returns the top results (titles, links, snippets)."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
            side_effects=False,
        )

        # --- Killer Demo: extract_from_text ---
        async def extract_from_text(
            text: str,
            schema: Dict[str, Any],
            *,
            model: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Extract structured data from unstructured text using an LLM.

            The JSON schema is used as instruction; output is a dict.
            """
            # Default lightweight model consistent with examples
            chosen_model = model or "openai:gpt-5-mini"

            system_prompt = (
                "You extract structured data from text.\n"
                "Return only valid JSON matching the provided JSON Schema.\n"
                "Do not include prose, backticks, or explanations.\n"
            )

            # Compose a single input string; the wrapper handles retries/repair
            input_payload = (
                "JSON_SCHEMA:\n"
                f"{schema}\n\n"
                "TEXT:\n"
                f"{text}\n\n"
                "Respond with JSON that validates against JSON_SCHEMA."
            )

            agent = make_agent_async(
                model=chosen_model,
                system_prompt=system_prompt,
                output_type=Dict[str, Any],
                max_retries=2,
                auto_repair=True,
            )

            result = await agent.run(input_payload)
            # The wrapper returns processed content; ensure it's a dict
            return result if isinstance(result, dict) else {"result": result}

        reg.register(
            "flujo.builtins.extract_from_text",
            lambda **_params: extract_from_text,
            description=(
                "Extracts structured data from text based on a provided JSON schema using an LLM."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "schema": {"type": "object"},
                    "model": {"type": "string"},
                },
                "required": ["text", "schema"],
            },
            side_effects=False,
        )

        # --- Killer Demo: http_get ---
        async def http_get(url: str, timeout: int = 30) -> Dict[str, Any]:
            """Fetch content from a URL and return status, headers, and body."""
            if _httpx is None:
                return {
                    "status_code": 500,
                    "headers": {},
                    "body": "httpx not installed; install optional dependency 'httpx'",
                }
            try:
                async with _httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=timeout, follow_redirects=True)
                    return {
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "body": resp.text,
                    }
            except Exception as e:  # pragma: no cover - network errors
                return {"status_code": 500, "headers": {}, "body": f"HTTP GET failed: {e}"}

        reg.register(
            "flujo.builtins.http_get",
            lambda **_params: http_get,
            description="Fetch content from a URL.",
            arg_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["url"],
            },
            side_effects=False,
        )

        # --- Killer Demo: fs_write_file ---
        async def fs_write_file(path: str, content: str) -> Dict[str, Any]:
            """Write content to a local file asynchronously.

            Prefers true async I/O via aiofiles when available. Falls back to
            thread offload to avoid blocking the event loop when aiofiles is not installed.
            """
            try:
                # Prefer true async I/O with aiofiles if installed
                import aiofiles

                async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
                    await f.write(content)
                return {"success": True, "path": path}
            except ImportError:
                # Fallback to thread offload if aiofiles is not available
                import asyncio as _asyncio

                def _write_sync() -> Dict[str, Any]:
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(content)
                        return {"success": True, "path": path}
                    except Exception as e:  # pragma: no cover - filesystem errors
                        return {"success": False, "error": str(e)}

                loop = _asyncio.get_event_loop()
                return await loop.run_in_executor(None, _write_sync)
            except Exception as e:
                return {"success": False, "error": str(e)}

        reg.register(
            "flujo.builtins.fs_write_file",
            lambda **_params: fs_write_file,
            description="Write content to a local file (side-effect).",
            arg_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            side_effects=True,
        )

        # Convenience: write pipeline YAML where the step input is the content
        async def write_pipeline_yaml(content: str, path: str = "pipeline.yaml") -> Dict[str, Any]:
            """Write YAML content to disk; treats step input as content.

            This adapter mirrors fs_write_file but accepts content as the first
            parameter so it works naturally with blueprint 'input' templating.
            """
            try:
                import aiofiles

                async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
                    await f.write(content)
                return {"success": True, "path": path}
            except ImportError:
                import asyncio as _asyncio

                def _write_sync() -> Dict[str, Any]:
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(content)
                        return {"success": True, "path": path}
                    except Exception as e:
                        return {"success": False, "error": str(e)}

                loop = _asyncio.get_event_loop()
                return await loop.run_in_executor(None, _write_sync)
            except Exception as e:  # pragma: no cover - filesystem errors
                return {"success": False, "error": str(e)}

        reg.register(
            "flujo.builtins.write_pipeline_yaml",
            lambda **_params: write_pipeline_yaml,
            description=(
                "Write YAML content to disk where the step input is the content (defaults to pipeline.yaml)."
            ),
            arg_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            side_effects=True,
        )
    except Exception:
        # Registration failures should not break import
        pass


# Register on import so CLI/YAML resolution can find it
_register_builtins()

from __future__ import annotations

from typing import Any, List, Optional

from flujo.type_definitions.common import JSONObject

from pydantic import BaseModel as PydanticBaseModel
from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.infra.skill_registry import get_skill_registry
from flujo.infra import telemetry
from flujo.domain.agent_protocol import AsyncAgentProtocol

# --- Architect agent stubs (Planner, ToolMatcher, YAML Writer) ---


def _register_architect_agents() -> None:
    """Register stub implementations for architect agents.

    These stubs enable local iteration without external LLMs.
    They follow the contracts defined in flujo/architect/models.py.
    """
    reg = get_skill_registry()

    # Planner Agent: decomposes user goal into high-level steps
    async def _planner_agent(payload: JSONObject) -> JSONObject:
        goal = str(payload.get("user_goal") or "").strip()
        g = goal.lower()
        steps: list[dict[str, str]] = []
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
    async def _tool_matcher_agent(payload: JSONObject) -> JSONObject:
        step_name = payload.get("step_name") or "Step"
        purpose = (payload.get("purpose") or "").lower()
        available = payload.get("available_skills") or []

        # Helper to check availability
        def _is_avail(sid: str) -> bool:
            try:
                # Check explicit available_skills list first
                if any(isinstance(x, dict) and x.get("id") == sid for x in available):
                    return True

                # Fallback to registry lookup
                entry = get_skill_registry().get(sid, scope=None)
                is_available = bool(entry) if isinstance(entry, dict) else (entry is not None)
                telemetry.logfire.debug(
                    f"[Architect] Checking availability of {sid}: {is_available}"
                )
                return is_available
            except Exception as e:
                telemetry.logfire.warning(f"[Architect] Error checking availability of {sid}: {e}")
                return False

        # Simple heuristics to choose a skill
        if any(k in purpose for k in ["http", "url", "fetch", "webpage", "download"]):
            avail = _is_avail("flujo.builtins.http_get")
            telemetry.logfire.debug(f"[Architect] http_get available: {avail}")
            sid = "flujo.builtins.http_get" if avail else "flujo.builtins.stringify"
            params: JSONObject = {}
        elif any(k in purpose for k in ["search", "find", "lookup", "discover"]):
            avail = _is_avail("flujo.builtins.web_search")
            telemetry.logfire.debug(f"[Architect] web_search available: {avail}")
            sid = "flujo.builtins.web_search" if avail else "flujo.builtins.stringify"
            params = {"query": purpose[:80]} if sid.endswith("web_search") else {}
        elif any(k in purpose for k in ["save", "write", "persist", "export", "file"]):
            avail = _is_avail("flujo.builtins.fs_write_file")
            telemetry.logfire.debug(f"[Architect] fs_write_file available: {avail}")
            sid = "flujo.builtins.fs_write_file" if avail else "flujo.builtins.stringify"
            params = {"path": "output.txt"} if sid.endswith("fs_write_file") else {}
        else:
            sid = "flujo.builtins.stringify"
            params = {}

        telemetry.logfire.debug(f"[Architect] Chosen agent for {step_name}: {sid}")
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
    async def _yaml_writer_agent(payload: JSONObject) -> JSONObject:
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
            branches: dict[str, list[JSONObject]] = {}
            for idx, sel in enumerate(selections, start=1):
                if not isinstance(sel, dict):
                    continue
                sid = sel.get("chosen_agent_id") or "flujo.builtins.stringify"
                params = sel.get("agent_params") or {}
                sname = sel.get("step_name") or f"Step {idx}"
                step_dict = {"kind": "step", "name": sname, "agent": {"id": sid, "params": params}}
                branches[f"branch_{idx}"] = [step_dict]

            parallel_dict: JSONObject = {
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
def has_yaml_key(
    _out: JSONObject | str | None = None,
    ctx: DomainBaseModel | None = None,
    **_kwargs: Any,
) -> str:
    try:
        yt = getattr(ctx, "yaml_text", None)
    except Exception:
        yt = None
    present = isinstance(yt, str) and yt.strip() != ""
    return "present" if present else "absent"


class DiscoverSkillsAgent(AsyncAgentProtocol[Any, JSONObject]):
    """Builtin agent that discovers available skills and exposes them to context.

    - Loads skills from a local catalog (skills.yaml/skills.json) and Python entry points.
    - Returns a structure suitable for LLM tool matching steps.
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = directory or "."

    async def run(self, data: JSONObject | str | None, **kwargs: Any) -> JSONObject:
        # Best-effort: load catalog + packaged entry points
        try:
            load_skills_catalog(self.directory)
            load_skills_entry_points()
        except Exception:
            # Non-fatal; continue with whatever is registered
            pass

        # Collect a public view of registered skills
        skills: list[JSONObject] = []
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
    decomposition: PydanticBaseModel | JSONObject,
    *,
    output_key: str = "prepared_steps_for_mapping",
) -> JSONObject:
    """Adapter to extract a list of step dicts from the decomposer output.

    Returns a dict so that `updates_context: true` can merge it into the pipeline context.
    """
    steps: list[JSONObject] = []
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
async def extract_yaml_text(
    writer_output: PydanticBaseModel | JSONObject | str | bytes,
) -> dict[str, str]:
    """
    Robustly extracts YAML text from various agent output formats,
    stores it in the context, and returns it as a dictionary.
    This function is the definitive bridge from YAML generation to the rest of the pipeline.
    """
    text: str | None = None
    try:
        # --- DEBUGGING: See exactly what we are receiving ---
        telemetry.logfire.debug(f"[extract_yaml_text] Received type: {type(writer_output)}")
        telemetry.logfire.debug(
            f"[extract_yaml_text] Received value (first 200 chars): {str(writer_output)[:200]}"
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
                            telemetry.logfire.debug(
                                "[extract_yaml_text] Successfully parsed JSON and extracted YAML"
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
        telemetry.logfire.warning(f"[extract_yaml_text] Exception during extraction: {e}")
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
        telemetry.logfire.warning(
            "[extract_yaml_text] Extracted text does not look like valid Flujo YAML."
        )

    telemetry.logfire.debug(
        f"[extract_yaml_text] Successfully extracted YAML (first 100 chars): {final_text[:100]}"
    )

    return {"yaml_text": final_text, "generated_yaml": final_text}


# --- Adapter: capture ValidationReport for later error extraction ---
async def capture_validation_report(
    report: PydanticBaseModel | JSONObject,
) -> JSONObject:
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
async def validation_report_to_flag(
    report: PydanticBaseModel | JSONObject,
) -> JSONObject:
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


def exit_when_yaml_valid(_out: JSONObject | str | None, context: DomainBaseModel | None) -> bool:
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
    report: PydanticBaseModel | JSONObject,
    *,
    context: DomainBaseModel | None = None,
) -> JSONObject:
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

        report_dict: JSONObject
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

    result: JSONObject = {
        "validation_errors": _json.dumps(errors),
        "yaml_is_valid": is_valid,
    }
    return result


# --- HITL helper: interpret user confirmation into branch key ---
async def check_user_confirmation(
    _out: str | JSONObject | None = None,
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
    _out: str | JSONObject | None = None,
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
    _out: JSONObject | str | None = None,
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
        telemetry.logfire.debug(
            "[SVB] out_type=%s out_keys=%s out_valid=%s ctx_flag=%s ctx_has=%s "
            "ctx_type=%s ctx_validation_report=%s"
            % (
                type(_out).__name__,
                (list(_out.keys()) if isinstance(_out, dict) else None),
                val,
                (getattr(context, "yaml_is_valid", None) if context is not None else None),
                (hasattr(context, "yaml_is_valid") if context is not None else False),
                (type(context).__name__ if context is not None else None),
                (hasattr(context, "validation_report") if context is not None else False),
            )
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
async def compute_validity_key(
    _x: JSONObject | None = None, *, context: DomainBaseModel | None = None
) -> str:
    try:
        val = bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        try:
            if isinstance(context, dict):
                val = bool(context.get("yaml_is_valid", False))
            else:
                val = False
        except Exception:
            val = False
    return "valid" if val else "invalid"


def select_by_yaml_shape(
    _out: JSONObject | str | None = None,
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
        telemetry.logfire.debug(
            "[SBYS] prev_is_dict=%s ctx_flag=%s prev_head=%s ctx_head=%s"
            % (
                isinstance(_out, dict),
                (getattr(context, "yaml_is_valid", None) if context is not None else None),
                (str(prev)[:30] if isinstance(prev, str) else None),
                (str(ctx_text)[:30] if isinstance(ctx_text, str) else None),
            )
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


async def shape_to_validity_flag(*, context: DomainBaseModel | None = None) -> JSONObject:
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
            _out: JSONObject = {"yaml_is_valid": False, "yaml_text": yt}
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
                _out2: JSONObject = {"yaml_is_valid": True, "yaml_text": yt}
                try:
                    gy = getattr(context, "generated_yaml", None)
                    if isinstance(gy, str):
                        _out2["generated_yaml"] = gy
                except Exception:
                    pass
                return _out2
        except Exception:
            pass
    _out3: JSONObject = {"yaml_is_valid": True}
    if isinstance(yt, str):
        _out3["yaml_text"] = yt
    try:
        gy = getattr(context, "generated_yaml", None)
        if isinstance(gy, str):
            _out3["generated_yaml"] = gy
    except Exception:
        pass
    return _out3


def always_valid_key(
    _out: JSONObject | str | None = None, ctx: DomainBaseModel | None = None
) -> str:
    """Return 'valid' unconditionally (used after successful repair)."""
    return "valid"


# --- In-memory YAML validation skill ---
async def validate_yaml(
    yaml_text: str, base_dir: Optional[str] = None
) -> PydanticBaseModel | JSONObject:
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

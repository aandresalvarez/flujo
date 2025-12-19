from __future__ import annotations
from typing import Any, Awaitable, Callable, Optional

from flujo.domain.models import BaseModel as DomainBaseModel
from pydantic import BaseModel as PydanticBaseModel
from flujo.infra.jinja_utils import create_jinja_environment
from flujo.infra.skill_models import SkillRegistration
from flujo.infra.skill_registry import get_skill_registry
from flujo.type_definitions.common import JSONObject
from flujo.domain.sandbox import SandboxExecution, SandboxProtocol
from flujo.infra.sandbox import NullSandbox
from ..agents.wrapper import make_agent_async
from .support import (
    context_get,
    context_merge,
    context_set,
    passthrough,
    repair_yaml_ruamel,
    welcome_agent,
)
from .architect import (
    DiscoverSkillsAgent,
    always_valid_key,
    capture_validation_report,
    check_user_confirmation,
    compute_validity_key,
    extract_decomposed_steps,
    extract_validation_errors,
    extract_yaml_text,
    select_by_yaml_shape,
    select_validity_branch,
    shape_to_validity_flag,
    validation_report_to_flag,
    validate_yaml,
)
from .extras_transforms import register_transform_skills

_jinja2: Any = None
try:
    import jinja2 as _jinja2
except ImportError:
    pass
_httpx: Any = None
try:  # pragma: no cover - optional dependency
    import httpx as _httpx
except Exception:
    pass
_DDGSAsync: Any = None
_DDGS_CLASS: Any = None
try:  # pragma: no cover - optional dependency
    from duckduckgo_search import AsyncDDGS as _DDGSAsync  # type: ignore[no-redef]
except Exception:
    try:
        from duckduckgo_search import DDGS as _DDGS_CLASS  # type: ignore[no-redef]
    except Exception:
        _DDGSAsync = None
        _DDGS_CLASS = None


def _resolve_sandbox(context: DomainBaseModel | None) -> SandboxProtocol:
    """Return sandbox from context when available, otherwise a NullSandbox."""
    if context is not None:
        for attr in ("sandbox", "_sandbox"):
            try:
                sandbox = getattr(context, attr)
                if isinstance(sandbox, SandboxProtocol):
                    return sandbox
            except Exception:
                continue
    return NullSandbox()


async def render_jinja_template(template: str, variables: JSONObject | None = None) -> str:
    """Render a Jinja2 template string with provided variables using sandboxing."""
    if _jinja2 is None:
        return template
    try:
        env = create_jinja_environment(_jinja2)
        tmpl = env.from_string(template)
        return str(tmpl.render(**(variables or {})))
    except Exception:
        # Do not raise in CLI flows; return original to avoid breaking pipelines
        return template


async def code_interpreter(
    code: str,
    *,
    language: str = "python",
    files: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    arguments: list[str] | None = None,
    timeout_s: float | None = None,
    context: DomainBaseModel | None = None,
) -> JSONObject:
    """Execute code within the configured sandbox and return structured output."""
    sandbox = _resolve_sandbox(context)
    request = SandboxExecution(
        code=code,
        language=language,
        files=files,
        environment=environment,
        arguments=tuple(arguments or ()),
        timeout_s=timeout_s,
    )
    try:
        result = await sandbox.exec_code(request)
    except Exception as exc:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 1,
            "timed_out": False,
            "error": str(exc),
            "sandbox_id": None,
            "succeeded": False,
            "artifacts": None,
        }
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "error": result.error,
        "sandbox_id": result.sandbox_id,
        "succeeded": bool(result.succeeded),
        "artifacts": result.artifacts,
    }


def _register_builtins() -> None:
    """Register builtin skills with the global registry."""
    try:
        reg = get_skill_registry()
        # NOTE: Prefer SkillRegistration for all new builtins to keep metadata consistent.
        # --- Context manipulation helpers (Task 2.3) ---
        if reg.get("flujo.builtins.context_set") is None:
            reg.register(
                **SkillRegistration(
                    id="flujo.builtins.context_set",
                    factory=lambda **_params: context_set,
                    description="Set a context field at a dot-separated path (e.g., 'call_count' or 'import_artifacts.counter')",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "value": {},  # Any type
                        },
                        "required": ["path", "value"],
                    },
                    side_effects=True,
                ).__dict__
            )
        if reg.get("flujo.builtins.context_merge") is None:
            reg.register(
                **SkillRegistration(
                    id="flujo.builtins.context_merge",
                    factory=lambda **_params: context_merge,
                    description="Merge a dictionary into context at a path (e.g., 'hitl_data' or 'import_artifacts.extras')",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "value": {"type": "object"},
                        },
                        "required": ["path", "value"],
                    },
                    side_effects=True,
                ).__dict__
            )
        if reg.get("flujo.builtins.context_get") is None:
            reg.register(
                **SkillRegistration(
                    id="flujo.builtins.context_get",
                    factory=lambda **_params: context_get,
                    description="Get a value from context at a dot-separated path with optional default",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "default": {},  # Any type
                        },
                        "required": ["path"],
                    },
                    side_effects=False,
                ).__dict__
            )
        # Welcome experience for new users
        reg.register(
            **SkillRegistration(
                id="flujo.builtins.welcome_agent",
                factory=lambda **_params: welcome_agent,
                description="Returns a fun welcome message for new users.",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string", "default": "Developer"}},
                },
                side_effects=False,
            ).__dict__
        )
        # Factory accepts params to match YAML 'agent: { id: ..., params: {...} }'
        reg.register(
            **SkillRegistration(
                id="flujo.builtins.discover_skills",
                factory=lambda directory=".": DiscoverSkillsAgent(directory=directory),
                description="Discover local and packaged skills; returns available_skills list.",
            ).__dict__
        )
        # Adapter function: return the async callable without invoking it
        # Loader will call this factory with params (none by default) and expect an agent object
        reg.register(
            **SkillRegistration(
                id="flujo.builtins.extract_decomposed_steps",
                factory=lambda **_params: extract_decomposed_steps,
                description=(
                    "Extract list of step dicts from decomposer output into 'prepared_steps_for_mapping'"
                ),
            ).__dict__
        )
        # Adapter extractor for YAML string
        reg.register(
            **SkillRegistration(
                id="flujo.builtins.extract_yaml_text",
                factory=lambda **_params: extract_yaml_text,
                description="Extract YAML string from YamlWriter output object or dict.",
            ).__dict__
        )

        # --- FSD: Built-in data transforms (M1)
        register_transform_skills(reg)

        # --- FSD-024: analyze_project (safe filesystem scan)
        async def analyze_project(
            _data: JSONObject | None = None, *, directory: str = ".", max_files: int = 200
        ) -> JSONObject:
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

        def _make_analyze_runner(
            directory: str = ".", max_files: int = 200
        ) -> Callable[..., Awaitable[JSONObject]]:
            async def _runner(_data: JSONObject | None = None, **_k: Any) -> JSONObject:
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
        async def visualize_plan(
            plan: list[JSONObject] | JSONObject,
        ) -> dict[str, str]:
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
        async def estimate_plan_cost(
            plan: list[JSONObject],
        ) -> dict[str, float]:
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
        ) -> JSONObject:
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
            referenced_side_ids: list[str] = []
            if sandbox:
                side_ids: list[str] = []
                _find: Callable[[str], list[str]] | None = None
                try:
                    from flujo.cli.helpers import find_side_effect_skills_in_yaml as _find
                except ImportError:
                    _find = None
                if _find is not None:
                    side_ids = _find(yaml_text)
                    referenced_side_ids = list(side_ids)
                # If no side-effect skills were detected, fall back to all registered
                # side-effect skills to ensure mocking in dry-run.
                if not side_ids:
                    entries = getattr(reg_local, "_entries", {})
                    if isinstance(entries, dict):
                        for scoped in entries.values():
                            if not isinstance(scoped, dict):
                                continue
                            for sid, versions in scoped.items():
                                if not isinstance(versions, dict):
                                    continue
                                entry_latest = versions.get("latest")
                                if isinstance(entry_latest, dict) and entry_latest.get(
                                    "side_effects"
                                ):
                                    side_ids.append(sid)
                # Heuristic: if YAML references fs_write_file explicitly, always mock it
                if "flujo.builtins.fs_write_file" in yaml_text:
                    if "flujo.builtins.fs_write_file" not in side_ids:
                        side_ids.append("flujo.builtins.fs_write_file")
                    if "flujo.builtins.fs_write_file" not in referenced_side_ids:
                        referenced_side_ids.append("flujo.builtins.fs_write_file")
            else:
                side_ids = []
            try:
                if sandbox and side_ids:
                    for sid in side_ids:
                        entry = reg_local.get(sid)
                        if not entry:
                            continue
                        # Capture scoped/latest entry for proper restore
                        versions = (
                            getattr(reg_local, "_entries", {}).get("default", {}).get(sid, {})
                        )
                        original_latest = (
                            versions.get("latest") if isinstance(versions, dict) else None
                        )
                        restore[sid] = {"scope": "default", "latest": original_latest}

                        def _make_factory(
                            _sid: str,
                        ) -> Callable[..., Callable[..., Awaitable[dict[str, _Any]]]]:
                            async def _mock(*_a: _Any, **_k: _Any) -> dict[str, _Any]:
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
                # Do not clobber outputs; optional decoration could be added per-step if needed.
                return {"dry_run_result": result}
            finally:
                if mutated:
                    for sid, snapshot in restore.items():
                        versions = (
                            getattr(reg_local, "_entries", {})
                            .get(snapshot.get("scope", "default"), {})
                            .get(sid)
                        )
                        if isinstance(versions, dict):
                            versions["latest"] = snapshot.get("latest")

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
        async def ask_user(
            question: Optional[str] = None, *, context: DomainBaseModel | None = None
        ) -> str:
            """Ask the user for input, with non-interactive fallback.
            Behavior:
            1) If context.initial_prompt is set (from piped input or FLUJO_INPUT), use it
            2) If stdin is non-interactive (e.g., piped input), return the provided
               value directly without prompting. This enables CLI usage like:
                   echo "goal" | flujo run pipeline.yaml
            3) Otherwise, prompt interactively using the question (or a default).
            Args:
                question: The question to ask the user (optional)
                context: Pipeline context (injected automatically by Flujo)
            Returns:
                User input string
            """
            try:
                # CRITICAL FIX: Check context.initial_prompt first (from piped input)
                # This handles the case where stdin was already consumed by resolve_initial_input()
                if context is not None:
                    try:
                        initial_prompt = getattr(context, "initial_prompt", None)
                        if initial_prompt and str(initial_prompt).strip():
                            return str(initial_prompt).strip()
                    except Exception:
                        pass
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
            **SkillRegistration(
                id="flujo.builtins.code_interpreter",
                factory=lambda **_params: code_interpreter,
                description="Execute code inside the configured sandbox and return stdout/stderr.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "default": "python"},
                        "files": {"type": "object"},
                        "environment": {"type": "object"},
                        "arguments": {"type": "array", "items": {"type": "string"}},
                        "timeout_s": {"type": "number"},
                    },
                    "required": ["code"],
                },
                side_effects=True,
            ).__dict__
        )
        reg.register(
            "flujo.builtins.always_valid_key",
            lambda **_params: always_valid_key,
            description="Return 'valid' unconditionally for post-repair branch logging.",
            side_effects=False,
        )
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
            lambda **_params: stringify,  # Factory pattern: call with YAML params, returns function
            description="Convert any input value to a string via str(x).",
            side_effects=False,
        )

        # Introspect registered framework step primitives and produce JSON Schemas
        async def get_framework_schema() -> JSONObject:
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
            schemas: JSONObject = {}
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
        async def get_yaml_validity(*, context: DomainBaseModel | None = None) -> JSONObject:
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
        def _resolve_validate_yaml(
            **_params: Any,
        ) -> Callable[..., Any]:
            # Dynamic resolution so test monkeypatches to flujo.builtins.validate_yaml take effect
            try:
                import importlib as _importlib
                from types import ModuleType as _ModuleType

                mod: _ModuleType = _importlib.import_module("flujo.builtins")
                resolved: object = getattr(mod, "validate_yaml", None)
                if callable(resolved):
                    return resolved
                return validate_yaml
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
            mapped_step_results: list[JSONObject] | JSONObject,
            *,
            context: DomainBaseModel | None = None,
        ) -> JSONObject:
            try:
                user_goal = getattr(context, "user_goal", None) or getattr(
                    context, "initial_prompt", None
                )
            except Exception:
                user_goal = None
            # Normalize list of results into list of dicts
            plans: list[JSONObject] = []
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
            skills: list[JSONObject] = []
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
            item: JSONObject | PydanticBaseModel,
            *,
            context: DomainBaseModel | None = None,
        ) -> JSONObject:
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
        async def web_search(query: str, max_results: int = 3) -> list[JSONObject]:
            """Perform a DuckDuckGo web search (top N simplified results).
            Returns a list of {title, link, snippet} dicts.
            """
            if _DDGSAsync is None and _DDGS_CLASS is None:
                # Graceful degrade if optional dependency not installed
                return []
            results: list[JSONObject] = []
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

                    def _search_sync() -> list[JSONObject]:
                        assert _DDGS_CLASS is not None
                        ddgs = _DDGS_CLASS()
                        search_results: list[JSONObject] = []
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
            simplified: list[JSONObject] = []
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

        if reg.get("flujo.builtins.web_search") is None:
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
            schema: JSONObject,
            *,
            model: Optional[str] = None,
        ) -> JSONObject:
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
            try:
                import importlib

                _mod = importlib.import_module("flujo.builtins")
                _make_agent_async = getattr(_mod, "make_agent_async", make_agent_async)
            except Exception:
                _make_agent_async = make_agent_async
            agent = _make_agent_async(
                model=chosen_model,
                system_prompt=system_prompt,
                output_type=JSONObject,
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
        async def http_get(url: str, timeout: int = 30) -> JSONObject:
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
            **SkillRegistration(
                id="flujo.builtins.http_get",
                factory=lambda **_params: http_get,
                description="Fetch content from a URL.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30},
                    },
                    "required": ["url"],
                },
                side_effects=False,
            ).__dict__
        )

        # --- Killer Demo: fs_write_file ---
        async def fs_write_file(path: str, content: str) -> JSONObject:
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

                def _write_sync() -> JSONObject:
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
        async def write_pipeline_yaml(content: str, path: str = "pipeline.yaml") -> JSONObject:
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

                def _write_sync() -> JSONObject:
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

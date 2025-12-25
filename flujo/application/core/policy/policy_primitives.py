from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from dataclasses import dataclass

from flujo.exceptions import ConfigurationError
from flujo.infra import telemetry
from .policy_registry import PolicyRegistry, StepPolicy

__all__ = [
    "LoopResumeState",
    "PolicyRegistry",
    "StepPolicy",
    "_unpack_agent_result",
    "_load_template_config",
    "_check_hitl_nesting_safety",
    "_normalize_plugin_feedback",
    "_normalize_builtin_params",
]


# Structured helpers ---------------------------------------------------------


@dataclass
class LoopResumeState:
    """Structured representation of the data needed to resume a paused loop iteration."""

    iteration: int = 1
    step_index: int = 0
    requires_hitl_payload: bool = False
    last_output: object | None = None
    paused_step_name: str | None = None

    STORAGE_KEY = "loop_resume_state"

    @classmethod
    def from_context(cls, context: object) -> LoopResumeState | None:
        if context is None:
            return None
        maybe_iteration = getattr(context, "loop_iteration_index", None)
        maybe_index = getattr(context, "loop_step_index", None)
        if isinstance(maybe_iteration, int) and isinstance(maybe_index, int):
            return cls(
                iteration=max(1, maybe_iteration),
                step_index=max(0, maybe_index),
                requires_hitl_payload=bool(
                    getattr(context, "loop_resume_requires_hitl_output", False)
                ),
                last_output=getattr(context, "loop_last_output", None),
                paused_step_name=getattr(context, "loop_paused_step_name", None),
            )
        return None

    def persist(self, context: object, body_length: int) -> None:
        bounded_index = max(0, min(self.step_index, body_length))
        if context is None:
            return
        try:
            if hasattr(context, "loop_iteration_index"):
                context.loop_iteration_index = max(1, self.iteration)
            if hasattr(context, "loop_step_index"):
                context.loop_step_index = bounded_index
            if hasattr(context, "loop_last_output"):
                context.loop_last_output = self.last_output
            if hasattr(context, "loop_paused_step_name"):
                context.loop_paused_step_name = self.paused_step_name
            if hasattr(context, "loop_resume_requires_hitl_output"):
                context.loop_resume_requires_hitl_output = bool(self.requires_hitl_payload)
        except Exception:
            pass

    @classmethod
    def clear(cls, context: object) -> None:
        if context is None:
            return
        try:
            if hasattr(context, "loop_iteration_index"):
                context.loop_iteration_index = None
            if hasattr(context, "loop_step_index"):
                context.loop_step_index = None
            if hasattr(context, "loop_last_output"):
                context.loop_last_output = None
            if hasattr(context, "loop_paused_step_name"):
                context.loop_paused_step_name = None
            if hasattr(context, "loop_resume_requires_hitl_output"):
                context.loop_resume_requires_hitl_output = False
        except Exception:
            pass


# Shared utilities -----------------------------------------------------------


def _unpack_agent_result(output: object) -> object:
    """Best-effort unpacking of common agent result wrappers."""

    try:
        from pydantic import BaseModel as _BM  # local import to avoid startup costs

        if isinstance(output, _BM):
            return output
    except Exception:
        pass
    for attr in ("output", "content", "result", "data", "text", "message", "value"):
        try:
            if hasattr(output, attr):
                return getattr(output, attr)
        except Exception:
            pass
    return output


def _load_template_config() -> tuple[bool, bool]:
    """Load template configuration from flujo.toml with fallback to defaults."""

    from flujo.infra.config_manager import TemplateConfig, get_config_manager
    import flujo.infra.telemetry as template_telemetry

    strict = False
    log_resolution = False
    try:
        config_mgr = get_config_manager()
        config = config_mgr.load_config()
        template_config = config.template or TemplateConfig()
        strict = template_config.undefined_variables == "strict"
        log_resolution = template_config.log_resolution
    except Exception as exc:
        template_telemetry.logfire.debug(f"Failed to load template config: {exc}")

    return strict, log_resolution


def _check_hitl_nesting_safety(step: object, core: object) -> None:
    """Runtime safety check for HITL steps in nested contexts."""

    try:
        execution_stack = getattr(core, "_execution_stack", None)
        if execution_stack is None:
            return

        has_loop = False
        has_conditional = False
        context_chain: list[str] = []

        for frame in execution_stack:
            frame_type = getattr(frame, "step_kind", None) or getattr(frame, "kind", None)
            frame_name = getattr(frame, "name", "unnamed")

            if frame_type in ("loop", "LoopStep"):
                has_loop = True
                context_chain.append(f"loop:{frame_name}")
            elif frame_type in ("conditional", "ConditionalStep"):
                has_conditional = True
                context_chain.append(f"conditional:{frame_name}")

        if has_loop and has_conditional:
            context_desc = " > ".join(context_chain)
            error_msg = (
                f"HITL step '{getattr(step, 'name', 'unnamed')}' cannot run inside a "
                f"conditional that is nested within a loop (context: {context_desc}). "
                "This structure is unsupported and will skip HITL execution, causing data loss. "
                "Move the HITL outside the loop or remove the conditional wrapper."
            )
            telemetry.logfire.error(f"HITL nesting safety check failed: {error_msg}")
            raise ConfigurationError(
                error_msg,
                suggestion=(
                    "Place HITL steps at the top level of the loop body or remove the "
                    "conditional wrapper to avoid being skipped."
                ),
                code="HITL-NESTED-001",
            )

    except (RuntimeError, ConfigurationError):
        raise
    except Exception as exc:
        telemetry.logfire.debug(f"HITL nesting safety check skipped due to error: {exc}")
        return


def _normalize_plugin_feedback(msg: str) -> str:
    """Shared message normalizer for consistent feedback formatting across policies."""

    try:
        prefixes = (
            "Plugin execution failed after max retries: ",
            "Plugin validation failed: ",
            "Agent execution failed: ",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if msg.startswith(prefix):
                    msg = msg[len(prefix) :]
                    changed = True
        return msg.strip()
    except Exception:
        return msg


def _normalize_builtin_params(step: object, data: object) -> object:
    """Normalize builtin skill parameters to support both 'params' and 'input'."""

    agent_spec = getattr(step, "agent", None)
    if agent_spec is None:
        return data

    if hasattr(agent_spec, "_step_callable"):
        func = agent_spec._step_callable
        if hasattr(func, "__module__") and func.__module__ == "flujo.builtins":
            step_input = getattr(step, "input", None)
            if isinstance(step_input, dict):
                return step_input

    agent_id = None
    if isinstance(agent_spec, str):
        agent_id = agent_spec
    elif isinstance(agent_spec, dict):
        agent_id = agent_spec.get("id")
    elif hasattr(agent_spec, "id"):
        agent_id = getattr(agent_spec, "id", None)

    if not isinstance(agent_id, str) or not agent_id.startswith("flujo.builtins."):
        return data

    params: JSONObject = {}

    if isinstance(agent_spec, dict) and "params" in agent_spec:
        agent_params = agent_spec["params"]
        if isinstance(agent_params, dict):
            params.update(agent_params)
    elif hasattr(agent_spec, "params"):
        agent_params = getattr(agent_spec, "params", None)
        if isinstance(agent_params, dict):
            params.update(agent_params)

    if not params:
        step_input = getattr(step, "input", None)
        if isinstance(step_input, dict):
            params.update(step_input)

    if not params:
        return data

    return params

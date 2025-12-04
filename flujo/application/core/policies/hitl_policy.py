from __future__ import annotations
# mypy: ignore-errors

from typing import Type

from ._shared import (  # noqa: F401
    Any,
    Callable,
    ConversationHistoryPromptProcessor,
    Dict,
    HistoryManager,
    HistoryStrategyConfig,
    HumanInTheLoopStep,
    Optional,
    Paused,
    PausedException,
    PipelineResult,
    Protocol,
    StepOutcome,
    StepResult,
    Success,
    UsageLimits,
    telemetry,
    to_outcome,
    _check_hitl_nesting_safety,
    _load_template_config,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame

# --- Human-In-The-Loop Step Executor policy ---


class HitlStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: HumanInTheLoopStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> StepOutcome[StepResult]: ...


class DefaultHitlStepExecutor(StepPolicy[HumanInTheLoopStep]):
    @property
    def handles_type(self) -> Type[HumanInTheLoopStep]:
        return HumanInTheLoopStep

    async def execute(
        self,
        core: Any,
        step: HumanInTheLoopStep,
        data: Any | None = None,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    ) -> StepOutcome[StepResult]:
        """Handle Human-In-The-Loop step execution."""
        if isinstance(step, ExecutionFrame):
            frame = step
            step = frame.step
            data = frame.data
            context = frame.context
        import time

        from flujo.exceptions import TemplateResolutionError

        try:
            if context is not None and hasattr(context, "scratchpad"):
                sp = context.scratchpad
                if isinstance(sp, dict):
                    if not sp.get("loop_resume_requires_hitl_output"):
                        sp.pop("hitl_data", None)
                        sp.pop("user_input", None)
                        sp.pop("paused_step_input", None)
                    last_step = sp.get("last_hitl_step")
                    # When we enter a new HITL step, clear stale HITL markers so we don't auto-consume
                    # previous input and accidentally skip pausing, except when we are resuming and
                    # intentionally feeding the pending user input to this step.
                    if last_step is None or last_step != step.name:
                        sp.pop("hitl_data", None)
                        sp.pop("user_input", None)
                        sp.pop("pause_message", None)
                        sp.pop("hitl_message", None)
                        sp.pop("paused_step_input", None)
                        if not sp.get("loop_resume_requires_hitl_output"):
                            sp.pop("loop_resume_requires_hitl_output", None)
                        sp.pop("loop_last_output", None)
                    sp["last_hitl_step"] = step.name
                    if data is not None and sp.get("status") == "paused":
                        # Hydrate user_input when resuming into the same HITL or when the resume flag is set.
                        if sp.get("loop_resume_requires_hitl_output") and "user_input" in sp:
                            # Preserve human resume payload; do not overwrite with loop data.
                            pass
                        elif last_step == step.name or sp.get("loop_resume_requires_hitl_output"):
                            sp["user_input"] = data
                        else:
                            sp.pop("user_input", None)
                    # Hard fast-path: if we already have a resume payload, consume it immediately.
                    if sp.get("loop_resume_requires_hitl_output") and "user_input" in sp:
                        resume_payload = sp.get("user_input")
                        sp["status"] = "running"
                        sp.pop("loop_resume_requires_hitl_output", None)
                        # Apply sink_to before returning (mirrors logic at lines 248-260)
                        if step.sink_to and context is not None:
                            try:
                                from flujo.utils.context import set_nested_context_field

                                set_nested_context_field(context, step.sink_to, resume_payload)
                            except Exception:
                                pass
                        return Success(
                            step_result=StepResult(
                                name=getattr(step, "name", "hitl"),
                                output=resume_payload,
                                success=True,
                                attempts=1,
                                latency_s=0.0,
                                token_counts=0,
                                cost_usd=0.0,
                            )
                        )
        except Exception:
            pass

        telemetry.logfire.debug("=== HANDLE HITL STEP ===")
        telemetry.logfire.debug(f"HITL step name: {step.name}")

        # Runtime safety check: Detect HITL in nested contexts
        # This is a fallback in case validation was bypassed or disabled
        _check_hitl_nesting_safety(step, core)

        # If resuming, auto-consume the most recent human response that matches
        # this HITL step's rendered message. This enables proper resume behavior
        # for complex steps (e.g., LoopStep) without forcing the user to answer again.
        #
        # Policy-Driven Execution: this logic belongs here (not in the runner).
        def _render_message(raw: Optional[str]) -> str:
            try:
                msg = raw if raw is not None else str(data)
            except Exception:
                msg = None
            if not isinstance(msg, str):
                return "Data conversion failed"
            text = msg
            if "{{" in text and "}}" in text:
                try:
                    from flujo.utils.prompting import AdvancedPromptFormatter
                    from flujo.utils.template_vars import (
                        get_steps_map_from_context,
                        TemplateContextProxy,
                        StepValueProxy,
                    )
                    from flujo.exceptions import TemplateResolutionError

                    # Get template configuration
                    strict, log_resolution = _load_template_config()

                    steps_map = get_steps_map_from_context(context)
                    steps_wrapped = {
                        k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                        for k, v in steps_map.items()
                    }
                    fmt_ctx = {
                        "context": TemplateContextProxy(context, steps=steps_wrapped),
                        "previous_step": data,
                        "steps": steps_wrapped,
                    }

                    # Use configured formatter with strict mode and logging
                    formatter = AdvancedPromptFormatter(
                        text, strict=strict, log_resolution=log_resolution
                    )
                    return formatter.format(**fmt_ctx)
                except TemplateResolutionError as e:
                    # In strict mode, log the error with step context and re-raise
                    telemetry.logfire.error(
                        f"[HITL] Template resolution failed in step '{step.name}': {e}"
                    )
                    raise
                except Exception:
                    return text
            return text

        try:
            rendered_message = _render_message(step.message_for_user)
        except TemplateResolutionError:
            # In strict mode, template failures must propagate
            raise
        except Exception:
            rendered_message = "Paused"

        # If no prior pause/input exists, initiate a pause immediately.
        try:
            sp = getattr(context, "scratchpad", {}) if context is not None else {}
            if isinstance(sp, dict):
                has_input = sp.get("user_input") is not None or sp.get("hitl_data") is not None
                status_is_paused = sp.get("status") == "paused"
                if not status_is_paused and not has_input:
                    sp["status"] = "paused"
                    sp["hitl_data"] = data
                    sp["hitl_message"] = rendered_message
                    sp["pause_message"] = rendered_message
                    sp["paused_step_input"] = data
                    setattr(context, "scratchpad", sp)
                    raise PausedException(rendered_message)
        except PausedException:
            raise
        except Exception:
            pass

        try:
            # When resuming, runner records the human response in ctx.hitl_history.
            # Complete this exact paused instance only when the current step input
            # matches the previously paused input (scratchpad['hitl_data']).
            hitl_hist = getattr(context, "hitl_history", None) if context is not None else None
            last = hitl_hist[-1] if isinstance(hitl_hist, list) and hitl_hist else None
            sp = getattr(context, "scratchpad", None) if context is not None else None
            prev_data = sp.get("hitl_data") if isinstance(sp, dict) else None
            status_is_paused = isinstance(sp, dict) and sp.get("status") == "paused"
            same_input = False
            try:
                same_input = (prev_data == data) or (str(prev_data) == str(data))
            except Exception:
                same_input = False
            if last is not None and same_input and status_is_paused:
                msg = getattr(last, "message_to_human", None)
                resp = getattr(last, "human_response", None)
                if isinstance(msg, str) and msg == rendered_message and resp is not None:
                    # Record a user turn in conversation history if enabled
                    try:
                        from flujo.domain.models import ConversationTurn, ConversationRole

                        hist = getattr(context, "conversation_history", None)
                        if isinstance(hist, list):
                            if not hist or getattr(hist[-1], "content", None) != str(resp):
                                hist.append(
                                    ConversationTurn(role=ConversationRole.user, content=str(resp))
                                )
                    except Exception:
                        pass
                    # Update steps map snapshot for templates
                    try:
                        if hasattr(context, "scratchpad") and isinstance(context.scratchpad, dict):
                            steps_map = context.scratchpad.setdefault("steps", {})
                            if isinstance(steps_map, dict):
                                steps_map[getattr(step, "name", "")] = str(resp)
                            # Clear the correlation data to avoid accidental reuse
                            context.scratchpad.pop("hitl_data", None)
                    except Exception:
                        pass
                    # If sink_to is specified, automatically store the response to context
                    if step.sink_to and context is not None:
                        try:
                            from flujo.utils.context import set_nested_context_field

                            telemetry.logfire.debug(
                                f"HITL sink_to: storing response to '{step.sink_to}'"
                            )
                            set_nested_context_field(context, step.sink_to, resp)
                            telemetry.logfire.info(f"HITL response stored to {step.sink_to}")
                        except Exception as e:
                            telemetry.logfire.warning(f"Failed to sink HITL to {step.sink_to}: {e}")
                    # Always mirror the response onto scratchpad.user_input for downstream steps
                    try:
                        if hasattr(context, "scratchpad") and isinstance(context.scratchpad, dict):
                            context.scratchpad["user_input"] = resp
                    except Exception:
                        pass
                    else:
                        if not step.sink_to:
                            telemetry.logfire.debug(
                                f"HITL step has no sink_to (sink_to={step.sink_to})"
                            )
                        if context is None:
                            telemetry.logfire.debug("HITL context is None")

                    # Produce a successful step result using the recorded response
                    return Success(
                        step_result=StepResult(
                            name=getattr(step, "name", "hitl"),
                            output=resp,
                            success=True,
                            attempts=1,
                            latency_s=0.0,
                            token_counts=0,
                            cost_usd=0.0,
                        )
                    )
        except Exception:
            # Fall through to pause behavior
            pass

        # Note: If not resuming, HITL step raises PausedException immediately.
        # Do not auto-consume human responses otherwise. Resumption should pause again
        # for subsequent HITL steps (e.g., Map over multiple items).

        if context is not None:
            try:
                if hasattr(context, "scratchpad") and isinstance(context.scratchpad, dict):
                    context.scratchpad["status"] = "paused"
                    context.scratchpad["last_state_update"] = time.monotonic()
                else:
                    core._update_context_state(context, "paused")
            except Exception as e:
                telemetry.logfire.error(f"Failed to update context state: {e}")

        # Fast path: if resuming from paused state and user input is provided, treat it
        # as the HITL response and return success without re-pausing.
        try:
            if (
                context is not None
                and hasattr(context, "scratchpad")
                and isinstance(context.scratchpad, dict)
                and context.scratchpad.get("status") == "paused"
                and context.scratchpad.get("loop_resume_requires_hitl_output")
                and data is not None
            ):
                resp = data
                sp = context.scratchpad
                sp["status"] = "running"
                sp["user_input"] = resp
                if step.sink_to:
                    try:
                        from flujo.utils.context import set_nested_context_field

                        set_nested_context_field(context, step.sink_to, resp)
                    except Exception:
                        pass
                # Consume the resume marker so subsequent HITLs will pause again.
                sp.pop("loop_resume_requires_hitl_output", None)
                sp.pop("hitl_data", None)
                return Success(
                    step_result=StepResult(
                        name=getattr(step, "name", "hitl"),
                        output=resp,
                        success=True,
                        attempts=1,
                        latency_s=0.0,
                        token_counts=0,
                        cost_usd=0.0,
                    )
                )
        except Exception:
            pass

        if context is not None and hasattr(context, "scratchpad"):
            try:
                # Use the rendered message computed earlier
                hitl_message = rendered_message
                context.scratchpad["hitl_message"] = hitl_message
                context.scratchpad["hitl_data"] = data
                # Persist user-provided input for downstream sinks/validators
                if data is not None:
                    context.scratchpad["user_input"] = data
                # Append assistant turn to conversation history so loops in conversation:true
                # capture the question even when the iteration pauses here.
                try:
                    # Ensure conversation_history container exists
                    if not hasattr(context, "conversation_history") or not isinstance(
                        getattr(context, "conversation_history", None), list
                    ):
                        setattr(context, "conversation_history", [])
                    # Append assistant question turn if not duplicated
                    from flujo.domain.models import ConversationTurn, ConversationRole

                    hist_list = getattr(context, "conversation_history", [])
                    last = hist_list[-1] if hist_list else None
                    if not last or getattr(last, "content", None) != hitl_message:
                        hist_list.append(
                            ConversationTurn(role=ConversationRole.assistant, content=hitl_message)
                        )
                        setattr(context, "conversation_history", hist_list)
                except Exception:
                    pass
                # Preserve pending AskHuman command for resumption logging
                try:
                    from flujo.domain.commands import AskHumanCommand as _AskHuman

                    context.scratchpad.setdefault(
                        "paused_step_input", _AskHuman(question=hitl_message)
                    )
                except Exception:
                    pass
            except Exception as e:
                telemetry.logfire.error(f"Failed to update context scratchpad: {e}")

        try:
            # Reuse same rendering path for the outgoing pause message
            if context is not None and hasattr(context, "scratchpad"):
                tmp_msg = context.scratchpad.get("hitl_message", "Paused")
                message = tmp_msg if isinstance(tmp_msg, str) else str(tmp_msg)
            else:
                # Render directly when context is not available
                def _render_direct(raw: Optional[str]) -> str:
                    try:
                        return raw if raw is not None else str(data)
                    except Exception:
                        return "Data conversion failed"

                message = _render_direct(step.message_for_user)
        except Exception:
            message = "Data conversion failed"
        # Return a Paused outcome when no context is present (unit tests), otherwise
        # raise to let orchestration capture the pause state.
        if context is None:
            return Paused(message=message)
        raise PausedException(message)


# --- End Human-In-The-Loop Step Executor policy ---

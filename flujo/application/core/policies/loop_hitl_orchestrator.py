from __future__ import annotations

from dataclasses import dataclass

from ._shared import Any, ContextManager, LoopResumeState, telemetry


@dataclass
class LoopResumeConfig:
    saved_iteration: int
    saved_step_index: int
    is_resuming: bool
    saved_last_output: Any
    resume_requires_hitl_output: bool
    resume_payload: Any
    paused_step_name: str | None
    iteration_count: int
    current_step_index: int


def prepare_resume_config(loop_step: Any, current_context: Any, data: Any) -> LoopResumeConfig:
    saved_iteration = 1
    saved_step_index = 0
    is_resuming = False
    saved_last_output = None
    resume_requires_hitl_output = False
    resume_payload = data
    paused_step_name: str | None = None
    resume_state = LoopResumeState.from_context(current_context) if current_context else None
    if resume_state is not None:
        saved_iteration = resume_state.iteration
        saved_step_index = resume_state.step_index
        is_resuming = True
        saved_last_output = resume_state.last_output
        resume_requires_hitl_output = resume_state.requires_hitl_payload
        paused_step_name = resume_state.paused_step_name
        LoopResumeState.clear(current_context)
        if hasattr(current_context, "status"):
            current_context.status = "paused" if resume_requires_hitl_output else "running"
        if hasattr(current_context, "loop_resume_requires_hitl_output"):
            current_context.loop_resume_requires_hitl_output = bool(resume_requires_hitl_output)
        telemetry.logfire.info(
            f"LoopStep '{loop_step.name}' RESUMING from iteration {saved_iteration}, step {saved_step_index}"
        )
    else:
        maybe_iteration = getattr(current_context, "loop_iteration_index", None)
        maybe_index = getattr(current_context, "loop_step_index", None)
        maybe_status = getattr(current_context, "status", None)
        maybe_last_output = getattr(current_context, "loop_last_output", None)
        resume_flag = getattr(current_context, "loop_resume_requires_hitl_output", None)
        paused_step_name_raw = getattr(current_context, "loop_paused_step_name", None)
        if isinstance(paused_step_name_raw, str) and paused_step_name_raw:
            paused_step_name = paused_step_name_raw
        if (
            isinstance(maybe_iteration, int)
            and maybe_iteration >= 1
            and isinstance(maybe_index, int)
            and maybe_index >= 0
            and (maybe_status in {"paused", "running", None})
        ):
            saved_iteration = maybe_iteration
            saved_step_index = maybe_index
            is_resuming = True
            saved_last_output = maybe_last_output
            resume_requires_hitl_output = bool(resume_flag)
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}' RESUMING from iteration {saved_iteration}, step {saved_step_index} (status={maybe_status})"
            )
            # Use typed field for status
            if hasattr(current_context, "status"):
                current_context.status = "paused" if resume_requires_hitl_output else "running"
    resume_payload = _resolve_resume_payload(
        current_context=current_context,
        resume_requires_hitl_output=resume_requires_hitl_output,
        paused_step_name=paused_step_name,
        resume_payload=resume_payload,
    )
    iteration_count = saved_iteration if saved_iteration >= 1 else 1
    try:
        if current_context is not None:
            if is_resuming and resume_requires_hitl_output:
                if hasattr(current_context, "status"):
                    current_context.status = "paused"
            if hasattr(current_context, "loop_iteration_index"):
                current_context.loop_iteration_index = iteration_count - 1
    except Exception:
        pass
    current_step_index = saved_step_index
    return LoopResumeConfig(
        saved_iteration=saved_iteration,
        saved_step_index=saved_step_index,
        is_resuming=is_resuming,
        saved_last_output=saved_last_output,
        resume_requires_hitl_output=resume_requires_hitl_output,
        resume_payload=resume_payload,
        paused_step_name=paused_step_name,
        iteration_count=iteration_count,
        current_step_index=current_step_index,
    )


def clear_hitl_markers(ctx: Any) -> None:
    try:
        if hasattr(ctx, "hitl_data"):
            ctx.hitl_data = {}
        if hasattr(ctx, "paused_step_input"):
            ctx.paused_step_input = None
        if hasattr(ctx, "user_input"):
            ctx.user_input = None
    except Exception:
        pass


def propagate_pause_state(
    *,
    iteration_context: Any,
    current_context: Any,
    iteration_count: int,
    current_step_index: int,
    current_data: Any,
    paused_step_name: str | None,
    hitl_output: Any = None,
) -> None:
    # Typed-only pause propagation (scratchpad is deprecated; no writes).
    if hasattr(iteration_context, "status"):
        iteration_context.status = "paused"
    if hasattr(iteration_context, "loop_iteration_index"):
        iteration_context.loop_iteration_index = iteration_count
    if hasattr(iteration_context, "loop_step_index"):
        iteration_context.loop_step_index = current_step_index + 1
    if hasattr(iteration_context, "loop_last_output"):
        iteration_context.loop_last_output = current_data
    if hasattr(iteration_context, "loop_resume_requires_hitl_output"):
        iteration_context.loop_resume_requires_hitl_output = True
    if hasattr(iteration_context, "loop_paused_step_name"):
        iteration_context.loop_paused_step_name = paused_step_name
    if hitl_output is not None:
        if hasattr(iteration_context, "paused_step_input"):
            iteration_context.paused_step_input = hitl_output
        val = getattr(hitl_output, "human_response", None)
        if hasattr(iteration_context, "user_input"):
            iteration_context.user_input = val
        if hasattr(iteration_context, "hitl_data") and val is not None:
            iteration_context.hitl_data = {"human_response": val}
    _append_pause_message(
        target_context=iteration_context,
        pause_message=getattr(iteration_context, "pause_message", None) or "",
    )
    if current_context is not None:
        try:
            # Keep the main context aligned for resume.
            for attr in (
                "status",
                "loop_iteration_index",
                "loop_step_index",
                "loop_last_output",
                "loop_resume_requires_hitl_output",
                "loop_paused_step_name",
                "paused_step_input",
                "user_input",
                "hitl_data",
                "pause_message",
            ):
                if hasattr(iteration_context, attr) and hasattr(current_context, attr):
                    setattr(current_context, attr, getattr(iteration_context, attr))
        except Exception:
            pass
        _append_pause_message(
            target_context=current_context,
            pause_message=getattr(current_context, "pause_message", None) or "",
        )
    try:
        ContextManager.merge(current_context, iteration_context)
    except Exception:
        pass


def _append_pause_message(target_context: Any, pause_message: str) -> None:
    if not pause_message:
        return
    try:
        from flujo.domain.models import ConversationRole, ConversationTurn

        hist = getattr(target_context, "conversation_history", None)
        if isinstance(hist, list):
            if not hist or getattr(hist[-1], "content", None) != pause_message:
                hist.append(
                    ConversationTurn(role=ConversationRole.assistant, content=str(pause_message))
                )
    except Exception:
        pass


def _resolve_resume_payload(
    *,
    current_context: Any,
    resume_requires_hitl_output: bool,
    paused_step_name: str | None,
    resume_payload: Any,
) -> Any:
    if not resume_requires_hitl_output:
        return resume_payload
    try:
        hitl_history = getattr(current_context, "hitl_history", None)
        if isinstance(hitl_history, list) and hitl_history:
            latest_resp = getattr(hitl_history[-1], "human_response", None)
            if latest_resp is not None:
                return latest_resp
    except Exception:
        pass
    steps_snap = getattr(current_context, "step_outputs", None)
    if resume_payload is None and isinstance(steps_snap, dict):
        try:
            if isinstance(paused_step_name, str) and paused_step_name in steps_snap:
                return steps_snap.get(paused_step_name)
        except Exception:
            return resume_payload
    return resume_payload

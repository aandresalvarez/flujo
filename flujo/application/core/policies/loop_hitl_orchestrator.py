from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ._shared import BaseModel, ContextManager, LoopResumeState, telemetry


class _NamedStep(Protocol):
    name: str


@dataclass
class LoopResumeConfig:
    saved_iteration: int
    saved_step_index: int
    is_resuming: bool
    saved_last_output: object | None
    resume_requires_hitl_output: bool
    resume_payload: object
    paused_step_name: str | None
    iteration_count: int
    current_step_index: int


def prepare_resume_config(
    loop_step: _NamedStep, current_context: object | None, data: object
) -> LoopResumeConfig:
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
        if current_context is not None and hasattr(current_context, "status"):
            try:
                setattr(
                    current_context,
                    "status",
                    "paused" if resume_requires_hitl_output else "running",
                )
            except Exception:
                pass
        if current_context is not None and hasattr(
            current_context, "loop_resume_requires_hitl_output"
        ):
            try:
                setattr(
                    current_context,
                    "loop_resume_requires_hitl_output",
                    bool(resume_requires_hitl_output),
                )
            except Exception:
                pass
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
            if current_context is not None and hasattr(current_context, "status"):
                try:
                    setattr(
                        current_context,
                        "status",
                        "paused" if resume_requires_hitl_output else "running",
                    )
                except Exception:
                    pass
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


def clear_hitl_markers(ctx: BaseModel | None) -> None:
    if ctx is None:
        return
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
    iteration_context: object | None,
    current_context: object | None,
    iteration_count: int,
    current_step_index: int,
    current_data: object,
    paused_step_name: str | None,
    hitl_output: object | None = None,
) -> None:
    # Typed-only pause propagation (scratchpad is deprecated; no writes).
    if iteration_context is not None and hasattr(iteration_context, "status"):
        try:
            setattr(iteration_context, "status", "paused")
        except Exception:
            pass
    if iteration_context is not None and hasattr(iteration_context, "loop_iteration_index"):
        try:
            setattr(iteration_context, "loop_iteration_index", iteration_count)
        except Exception:
            pass
    if iteration_context is not None and hasattr(iteration_context, "loop_step_index"):
        try:
            setattr(iteration_context, "loop_step_index", current_step_index + 1)
        except Exception:
            pass
    if iteration_context is not None and hasattr(iteration_context, "loop_last_output"):
        try:
            setattr(iteration_context, "loop_last_output", current_data)
        except Exception:
            pass
    if iteration_context is not None and hasattr(
        iteration_context, "loop_resume_requires_hitl_output"
    ):
        try:
            setattr(iteration_context, "loop_resume_requires_hitl_output", True)
        except Exception:
            pass
    if iteration_context is not None and hasattr(iteration_context, "loop_paused_step_name"):
        try:
            setattr(iteration_context, "loop_paused_step_name", paused_step_name)
        except Exception:
            pass
    if hitl_output is not None:
        if iteration_context is not None and hasattr(iteration_context, "paused_step_input"):
            try:
                setattr(iteration_context, "paused_step_input", hitl_output)
            except Exception:
                pass
        val = getattr(hitl_output, "human_response", None)
        if iteration_context is not None and hasattr(iteration_context, "user_input"):
            try:
                setattr(iteration_context, "user_input", val)
            except Exception:
                pass
        if (
            iteration_context is not None
            and hasattr(iteration_context, "hitl_data")
            and val is not None
        ):
            try:
                setattr(iteration_context, "hitl_data", {"human_response": val})
            except Exception:
                pass
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
    if isinstance(current_context, BaseModel) and isinstance(iteration_context, BaseModel):
        try:
            ContextManager.merge(current_context, iteration_context)
        except Exception:
            pass


def _append_pause_message(target_context: object | None, pause_message: str) -> None:
    if not pause_message:
        return
    if target_context is None:
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
    current_context: object | None,
    resume_requires_hitl_output: bool,
    paused_step_name: str | None,
    resume_payload: object,
) -> object:
    if not resume_requires_hitl_output:
        return resume_payload
    if current_context is None:
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

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
    scratchpad_ref = getattr(current_context, "scratchpad", None)
    resume_state = LoopResumeState.from_context(current_context) if current_context else None
    if resume_state is not None:
        saved_iteration = resume_state.iteration
        saved_step_index = resume_state.step_index
        is_resuming = True
        saved_last_output = resume_state.last_output
        resume_requires_hitl_output = resume_state.requires_hitl_payload
        paused_step_name = resume_state.paused_step_name
        LoopResumeState.clear(current_context)
        if isinstance(scratchpad_ref, dict):
            scratchpad_ref["status"] = "paused" if resume_requires_hitl_output else "running"
        telemetry.logfire.info(
            f"LoopStep '{loop_step.name}' RESUMING from iteration {saved_iteration}, step {saved_step_index}"
        )
    elif isinstance(scratchpad_ref, dict):
        maybe_iteration = scratchpad_ref.get("loop_iteration")
        maybe_index = scratchpad_ref.get("loop_step_index")
        maybe_status = scratchpad_ref.get("status")
        maybe_last_output = scratchpad_ref.get("loop_last_output")
        resume_flag = scratchpad_ref.get("loop_resume_requires_hitl_output")
        paused_step_name_raw = scratchpad_ref.get("loop_paused_step_name")
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
            scratchpad_ref["status"] = "paused" if resume_requires_hitl_output else "running"
    resume_payload = _resolve_resume_payload(
        current_context=current_context,
        resume_requires_hitl_output=resume_requires_hitl_output,
        paused_step_name=paused_step_name,
        resume_payload=resume_payload,
    )
    iteration_count = saved_iteration if saved_iteration >= 1 else 1
    try:
        if current_context is not None and hasattr(current_context, "scratchpad"):
            sp_main = getattr(current_context, "scratchpad")
            if isinstance(sp_main, dict):
                if is_resuming and resume_requires_hitl_output:
                    sp_main["status"] = "paused"
                sp_main["loop_iteration"] = iteration_count - 1
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
    scratch = getattr(ctx, "scratchpad", None)
    if isinstance(scratch, dict):
        scratch.pop("hitl_data", None)
        scratch.pop("paused_step_input", None)


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
    scratchpad_iter = getattr(iteration_context, "scratchpad", None)
    if isinstance(scratchpad_iter, dict):
        scratchpad_iter["status"] = "paused"
        scratchpad_iter["loop_iteration"] = iteration_count
        scratchpad_iter["loop_step_index"] = current_step_index
        scratchpad_iter["loop_last_output"] = current_data
        scratchpad_iter["loop_resume_requires_hitl_output"] = True
        scratchpad_iter["loop_paused_step_name"] = paused_step_name
        if hitl_output is not None:
            scratchpad_iter["paused_step_input"] = hitl_output
            scratchpad_iter["hitl_data"] = getattr(hitl_output, "human_response", None)
        _append_pause_message(
            target_context=iteration_context, pause_message=scratchpad_iter.get("pause_message", "")
        )
    if current_context is not None and isinstance(
        getattr(current_context, "scratchpad", None), dict
    ):
        current_context.scratchpad.update(
            scratchpad_iter if isinstance(scratchpad_iter, dict) else {}
        )
        _append_pause_message(
            target_context=current_context,
            pause_message=current_context.scratchpad.get("pause_message", ""),
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
    scratchpad_ref = getattr(current_context, "scratchpad", None)
    if resume_payload is None and isinstance(scratchpad_ref, dict):
        try:
            steps_snap = scratchpad_ref.get("steps")
            if (
                isinstance(steps_snap, dict)
                and isinstance(paused_step_name, str)
                and paused_step_name in steps_snap
            ):
                return steps_snap.get(paused_step_name)
        except Exception:
            return resume_payload
    return resume_payload

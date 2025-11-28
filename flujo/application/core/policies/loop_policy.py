from __future__ import annotations
from typing import Type
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from flujo.domain.dsl.loop import LoopStep
from ._shared import (  # noqa: F401
    Any,
    Awaitable,
    Callable,
    Dict,
    ConversationHistoryPromptProcessor,
    HistoryManager,
    HistoryStrategyConfig,
    HumanInTheLoopStep,
    List,
    LoopResumeState,
    Optional,
    Pipeline,
    PipelineContext,
    PipelineResult,
    Protocol,
    Step,
    StepOutcome,
    StepResult,
    UsageLimitExceededError,
    UsageLimits,
    ContextManager,
    Paused,
    PausedException,
    telemetry,
    time,
    to_outcome,
)
from .loop_iteration_runner import run_loop_iterations


class LoopStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        loop_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]: ...


class DefaultLoopStepExecutor(StepPolicy[LoopStep[Any]]):
    @property
    def handles_type(self) -> Type[LoopStep[Any]]:
        return LoopStep

    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]:
        if isinstance(step, ExecutionFrame):
            frame = step
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            cache_key = cache_key or (
                core._cache_key(frame) if getattr(core, "_enable_cache", False) else None
            )
            try:
                _fallback_depth = int(getattr(frame, "_fallback_depth", _fallback_depth) or 0)
            except Exception:
                _fallback_depth = _fallback_depth
        loop_step = step
        context_setter = getattr(core, "_context_setter", None)
        telemetry.logfire.info(
            f"[POLICY] DefaultLoopStepExecutor executing '{getattr(loop_step, 'name', '<unnamed>')}'"
        )
        telemetry.logfire.debug(f"Handling LoopStep '{getattr(loop_step, 'name', '<unnamed>')}'")
        _bp = (
            loop_step.get_loop_body_pipeline()
            if hasattr(loop_step, "get_loop_body_pipeline")
            else getattr(loop_step, "loop_body_pipeline", None)
        )
        telemetry.logfire.info(f"[POLICY] Loop body pipeline: {_bp}")
        telemetry.logfire.info(
            f"[POLICY] Core has _execute_pipeline: {hasattr(core, '_execute_pipeline')}"
        )
        start_time = time.monotonic()
        current_data = data
        current_context = context or PipelineContext(initial_prompt=str(data))
        try:
            if hasattr(loop_step, "_results_var") and hasattr(loop_step, "_items_var"):
                getattr(loop_step, "_results_var").set([])
                getattr(loop_step, "_items_var").set([])
            if hasattr(loop_step, "_body_var") and hasattr(loop_step, "_original_body_pipeline"):
                getattr(loop_step, "_body_var").set(getattr(loop_step, "_original_body_pipeline"))
            if hasattr(loop_step, "_max_loops_var"):
                # Preserve the configured max_loops instead of forcing a single iteration.
                configured_max_loops = getattr(loop_step, "max_loops", 1)
                try:
                    configured_max_loops = int(configured_max_loops)
                except Exception:
                    configured_max_loops = 1
                getattr(loop_step, "_max_loops_var").set(configured_max_loops)
        except Exception:
            pass
        initial_mapper = (
            loop_step.get_initial_input_to_loop_body_mapper()
            if hasattr(loop_step, "get_initial_input_to_loop_body_mapper")
            else getattr(loop_step, "initial_input_to_loop_body_mapper", None)
        )
        if initial_mapper:
            try:
                current_data = initial_mapper(current_data, current_context)
            except Exception as e:
                sr = StepResult(
                    name=loop_step.name,
                    success=False,
                    output=None,
                    attempts=0,
                    latency_s=time.monotonic() - start_time,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Error in initial_input_to_loop_body_mapper for LoopStep '{loop_step.name}': {e}",
                    branch_context=current_context,
                    metadata_={"iterations": 0, "exit_reason": "initial_input_mapper_error"},
                    step_history=[],
                )
                return to_outcome(sr)
        body_pipeline = (
            loop_step.get_loop_body_pipeline()
            if hasattr(loop_step, "get_loop_body_pipeline")
            else getattr(loop_step, "loop_body_pipeline", None)
        )
        try:
            if body_pipeline is not None and (
                not hasattr(body_pipeline, "steps") or not isinstance(body_pipeline.steps, list)
            ):
                alt = getattr(loop_step, "loop_body_pipeline", None)
                if alt is not None and hasattr(alt, "steps") and isinstance(alt.steps, list):
                    body_pipeline = alt
        except Exception:
            pass
        if body_pipeline is None or not getattr(body_pipeline, "steps", []):
            sr = StepResult(
                name=loop_step.name,
                success=False,
                output=data,
                attempts=0,
                latency_s=time.monotonic() - start_time,
                token_counts=0,
                cost_usd=0.0,
                feedback="LoopStep has empty pipeline",
                branch_context=current_context,
                metadata_={"iterations": 0, "exit_reason": "empty_pipeline"},
                step_history=[],
            )
            return to_outcome(sr)
        conv_enabled = False
        history_cfg: HistoryStrategyConfig | None = None
        history_template: str | None = None
        ai_src: str = "last"
        user_src: list[str] = ["hitl"]
        named_steps_set: set[str] = set()
        try:
            meta = getattr(loop_step, "meta", None)
            if isinstance(meta, dict):
                conv_enabled = bool(meta.get("conversation") is True)
                hm = meta.get("history_management")
                if isinstance(hm, dict):
                    history_cfg = HistoryStrategyConfig(
                        strategy=str(hm.get("strategy") or "truncate_tokens"),
                        max_tokens=int(hm.get("max_tokens") or 4096),
                        max_turns=int(hm.get("max_turns") or 20),
                        summarizer_agent=None,
                        summarize_ratio=float(hm.get("summarize_ratio") or 0.5),
                    )
                if isinstance(meta.get("history_template"), str):
                    history_template = str(meta.get("history_template"))
                ai_turn_source = str(meta.get("ai_turn_source") or "last").strip().lower()
                user_turn_sources = meta.get("user_turn_sources")
                named_steps = meta.get("named_steps")
                try:
                    _ai_turn_source = ai_turn_source
                except Exception:
                    _ai_turn_source = "last"
                try:
                    _user_turn_sources = (
                        list(user_turn_sources)
                        if isinstance(user_turn_sources, (list, tuple))
                        else ([user_turn_sources] if user_turn_sources else ["hitl"])
                    )
                except Exception:
                    _user_turn_sources = ["hitl"]
                try:
                    _named_steps = set(str(s) for s in (named_steps or []))
                except Exception:
                    _named_steps = set()
                ai_src = _ai_turn_source
                user_src = list(_user_turn_sources)
                named_steps_set = set(_named_steps)
        except Exception:
            conv_enabled = False
        max_loops = (
            loop_step.get_max_loops()
            if hasattr(loop_step, "get_max_loops")
            else getattr(loop_step, "max_loops", 1)
        )
        if not isinstance(max_loops, int):
            ml = getattr(loop_step, "max_loops", None)
            if isinstance(ml, int):
                max_loops = ml
            else:
                mr = getattr(loop_step, "max_retries", None)
                max_loops = mr if isinstance(mr, int) else 1
        try:
            items_len = (
                len(getattr(loop_step, "_items_var").get())
                if hasattr(loop_step, "_items_var")
                else -1
            )
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}': configured max_loops={max_loops}, items_len={items_len}"
            )
        except Exception:
            pass
        try:
            with telemetry.logfire.span(loop_step.name):
                telemetry.logfire.debug(f"[POLICY] Opened overall loop span for '{loop_step.name}'")
        except Exception:
            pass
        saved_iteration = 1
        saved_step_index = 0
        is_resuming = False  # Track if this is a resume call
        saved_last_output = None  # Track last output before pause
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
        if resume_requires_hitl_output:
            try:
                hitl_history = getattr(current_context, "hitl_history", None)
                if isinstance(hitl_history, list) and hitl_history:
                    latest_resp = getattr(hitl_history[-1], "human_response", None)
                    if latest_resp is not None:
                        resume_payload = latest_resp
            except Exception:
                pass
            if resume_payload is None and isinstance(scratchpad_ref, dict):
                try:
                    steps_snap = scratchpad_ref.get("steps")
                    if (
                        isinstance(steps_snap, dict)
                        and isinstance(paused_step_name, str)
                        and paused_step_name in steps_snap
                    ):
                        resume_payload = steps_snap.get(paused_step_name)
                except Exception:
                    pass
        else:
            resume_payload = data
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
        return await run_loop_iterations(
            core=core,
            loop_step=loop_step,
            body_pipeline=body_pipeline,
            current_data=current_data,
            current_context=current_context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            conv_enabled=conv_enabled,
            history_cfg=history_cfg,
            history_template=history_template,
            ai_src=ai_src,
            user_src=user_src,
            named_steps_set=named_steps_set,
            max_loops=max_loops,
            saved_iteration=saved_iteration,
            saved_step_index=saved_step_index,
            is_resuming=is_resuming,
            resume_requires_hitl_output=resume_requires_hitl_output,
            resume_payload=resume_payload,
            saved_last_output=saved_last_output,
            paused_step_name=paused_step_name,
            iteration_count=iteration_count,
            current_step_index=current_step_index,
            start_time=start_time,
            context_setter=context_setter,
            _fallback_depth=_fallback_depth,
        )

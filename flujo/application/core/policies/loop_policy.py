from __future__ import annotations
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import BaseModel as DomainBaseModel
from ._shared import (  # noqa: F401
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
from .loop_hitl_orchestrator import LoopResumeConfig, prepare_resume_config
from .loop_iteration_runner import run_loop_iterations
from .loop_mapper import map_initial_input


class LoopStepExecutor(Protocol):
    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[DomainBaseModel],
    ) -> StepOutcome[StepResult]: ...


class DefaultLoopStepExecutor(StepPolicy[LoopStep[DomainBaseModel]]):
    @property
    def handles_type(self) -> type[LoopStep[DomainBaseModel]]:
        return LoopStep

    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]:
        loop_step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        stream = frame.stream
        on_chunk = frame.on_chunk
        cache_key = None
        if getattr(core, "_enable_cache", False):
            try:
                cache_key_fn = getattr(core, "_cache_key", None)
                cache_key = cache_key_fn(frame) if callable(cache_key_fn) else None
            except Exception:
                cache_key = None
        try:
            _fallback_depth = int(getattr(frame, "_fallback_depth", 0) or 0)
        except Exception:
            _fallback_depth = 0
        context_setter = getattr(frame, "context_setter", getattr(core, "_context_setter", None))
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
        # Note: keep an isolation call for compliance but preserve legacy semantics for HITL pause/resume.
        _ = (
            ContextManager.isolate(
                context,
                purpose=f"loop_base:{getattr(loop_step, 'name', '<unnamed>')}",
            )
            if context is not None
            else None
        )
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
        current_data, initial_mapper_outcome = map_initial_input(
            loop_step=loop_step,
            current_data=current_data,
            current_context=current_context,
            start_time=start_time,
        )
        if initial_mapper_outcome:
            return initial_mapper_outcome
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
        resume_config: LoopResumeConfig = prepare_resume_config(
            loop_step=loop_step,
            current_context=current_context,
            data=data,
        )
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
            saved_iteration=resume_config.saved_iteration,
            saved_step_index=resume_config.saved_step_index,
            is_resuming=resume_config.is_resuming,
            resume_requires_hitl_output=resume_config.resume_requires_hitl_output,
            resume_payload=resume_config.resume_payload,
            saved_last_output=resume_config.saved_last_output,
            paused_step_name=resume_config.paused_step_name,
            iteration_count=resume_config.iteration_count,
            current_step_index=resume_config.current_step_index,
            start_time=start_time,
            context_setter=context_setter,
            _fallback_depth=_fallback_depth,
        )

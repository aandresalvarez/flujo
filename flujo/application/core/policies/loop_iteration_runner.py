from __future__ import annotations

from typing import Protocol

from ._shared import (
    Awaitable,
    BaseModel,
    Callable,
    ConversationHistoryPromptProcessor,
    ContextManager,
    HistoryManager,
    HistoryStrategyConfig,
    Optional,
    Paused,
    PausedException,
    Pipeline,
    PipelineResult,
    Step,
    StepOutcome,
    StepResult,
    UsageLimitExceededError,
    UsageLimits,
    telemetry,
    time,
    to_outcome,
)
from ..execution.executor_helpers import make_execution_frame
from ..context.context_vars import _CACHE_OVERRIDE
from flujo.exceptions import PipelineAbortSignal
from .loop_hitl_orchestrator import clear_hitl_markers, propagate_pause_state
from .loop_history import (
    collect_step_name_sources,
    seed_conversation_history,
    sync_conversation_history,
)
from .loop_mapper import apply_iteration_input_mapper, finalize_loop_output
from ._shared import HumanInTheLoopStep  # noqa: F401  # used via isinstance checks
from ..step_result_pool import build_pooled_step_result


class _NamedStep(Protocol):
    name: str


async def run_loop_iterations(
    *,
    core: object,
    loop_step: _NamedStep,
    body_pipeline: Pipeline[object, object] | object,
    current_data: object,
    current_context: BaseModel | None,
    resources: object,
    limits: UsageLimits | None,
    stream: bool,
    on_chunk: Optional[Callable[[object], Awaitable[None]]],
    cache_key: Optional[str],
    conv_enabled: bool,
    history_cfg: HistoryStrategyConfig | None,
    history_template: str | None,
    ai_src: str,
    user_src: list[str],
    named_steps_set: set[str],
    max_loops: int,
    saved_iteration: int,
    saved_step_index: int,
    is_resuming: bool,
    resume_requires_hitl_output: bool,
    resume_payload: object,
    saved_last_output: object,
    paused_step_name: str | None,
    iteration_count: int,
    current_step_index: int,
    start_time: float,
    context_setter: Optional[Callable[[PipelineResult[BaseModel], BaseModel | None], None]],
    _fallback_depth: int,
) -> StepOutcome[StepResult]:
    loop_body_steps: list[object] = []
    steps_attr = getattr(body_pipeline, "steps", None) if hasattr(body_pipeline, "steps") else None
    steps_len = 0
    try:
        steps_len = len(steps_attr) if steps_attr is not None else 0
    except Exception:
        steps_len = 0
    if steps_attr and steps_len > 1:
        loop_body_steps = list(steps_attr) if not isinstance(steps_attr, list) else steps_attr
        telemetry.logfire.info(
            f"LoopStep '{loop_step.name}' using step-by-step execution with {len(loop_body_steps)} steps"
        )
    else:
        loop_body_steps = []
        telemetry.logfire.info(
            f"LoopStep '{loop_step.name}' using regular execution (single/no-step pipeline)"
        )
    stashed_exec_lists: list[list[object]] = []
    if is_resuming:
        total_steps = len(loop_body_steps)
        if total_steps == 0:
            current_step_index = 0
        elif current_step_index > total_steps:
            telemetry.logfire.warning(
                f"LoopStep '{loop_step.name}' resume step index {current_step_index} exceeds body length {total_steps}; clamping"
            )
            current_step_index = total_steps
        if saved_last_output is not None:
            current_data = saved_last_output
        # When resuming to feed HITL, prefer the resume payload (human input) as the current data.
        if resume_requires_hitl_output and resume_payload is not None:
            current_data = resume_payload
    prev_cumulative_cost = 0.0
    prev_cumulative_tokens = 0
    cumulative_cost = 0.0
    cumulative_tokens = 0
    iteration_results: list[StepResult] = []
    iteration_results_all: list[StepResult] = []
    while iteration_count <= max_loops:
        prev_cumulative_cost = cumulative_cost
        prev_cumulative_tokens = cumulative_tokens
        iteration_results = []
        with telemetry.logfire.span(f"Loop '{loop_step.name}' - Iteration {iteration_count}"):
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}': Starting Iteration {iteration_count}/{max_loops}"
            )
        iteration_context = (
            ContextManager.isolate(
                current_context,
                purpose=f"loop_iteration:{loop_step.name}:{iteration_count}",
            )
            if current_context is not None
            else None
        )
        if current_context is not None and iteration_context is not None:
            ContextManager.verify_isolation(current_context, iteration_context)
        try:
            if iteration_count == 1:
                try:
                    init_fn = None
                    meta = getattr(loop_step, "meta", None)
                    if isinstance(meta, dict):
                        init_fn = meta.get("compiled_init_ops")
                except Exception:
                    init_fn = None
                if callable(init_fn):
                    init_fn(current_data, iteration_context)
            if conv_enabled and iteration_context is not None:
                seed_conversation_history(iteration_context, current_data)
        except PausedException as e:
            raise e
        except Exception as e:
            return to_outcome(
                build_pooled_step_result(
                    name=loop_step.name,
                    success=False,
                    output=None,
                    attempts=0,
                    latency_s=time.monotonic() - start_time,
                    token_counts=0,
                    cost_usd=cumulative_cost,
                    feedback=f"Error in loop.init for LoopStep '{loop_step.name}': {e}",
                    branch_context=iteration_context,
                    metadata={
                        "iterations": 0,
                        "exit_reason": "initial_input_mapper_error",
                    },
                    step_history=iteration_results_all + iteration_results,
                )
            )
        hitl_step_names, agent_step_names = collect_step_name_sources(body_pipeline)
        try:
            if conv_enabled and hasattr(body_pipeline, "steps"):
                new_steps = []
                for st in list(getattr(body_pipeline, "steps", [])):
                    use_hist = True
                    try:
                        if isinstance(getattr(st, "meta", None), dict):
                            uh = st.meta.get("use_history")
                            if uh is not None:
                                use_hist = bool(uh)
                    except Exception:
                        use_hist = True
                    if (
                        conv_enabled
                        and use_hist
                        and isinstance(st, Step)
                        and not getattr(st, "is_complex", False)
                    ):
                        try:
                            st_copy = st.model_copy(deep=True)
                            hm = HistoryManager(history_cfg) if history_cfg else HistoryManager()
                            proc = ConversationHistoryPromptProcessor(
                                history_manager=hm,
                                history_template=history_template,
                                model_id=None,
                            )
                            pp = list(getattr(st_copy.processors, "prompt_processors", []) or [])
                            st_copy.processors.prompt_processors = [proc] + pp
                            new_steps.append(st_copy)
                        except Exception:
                            new_steps.append(st)
                    else:
                        new_steps.append(st)
                # instrumented_pipeline = _PipelineDSL.model_construct(steps=new_steps)  # Not used in step-by-step execution
        except Exception:
            # instrumented_pipeline = body_pipeline  # Not used in step-by-step execution
            pass

        if not loop_body_steps:
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}' using standard pipeline execution (no step slicing)"
            )
            try:
                # Mark loop context for downstream orchestrators
                try:
                    object.__setattr__(iteration_context, "_last_loop_iterations", iteration_count)
                except Exception:
                    try:
                        setattr(iteration_context, "_last_loop_iterations", iteration_count)
                    except Exception:
                        pass
                prev_loop_flag = getattr(core, "_inside_loop_iteration", False)
                try:
                    object.__setattr__(core, "_inside_loop_iteration", True)
                except Exception:
                    try:
                        setattr(core, "_inside_loop_iteration", True)
                    except Exception:
                        pass
                step_force_prev: dict[object, object] = {}
                try:
                    for st in getattr(body_pipeline, "steps", []) or []:
                        try:
                            step_force_prev[st] = getattr(st, "_force_loop_fallback", None)
                            object.__setattr__(st, "_force_loop_fallback", True)
                        except Exception:
                            try:
                                setattr(st, "_force_loop_fallback", True)
                            except Exception:
                                pass
                except Exception:
                    pass
                token = _CACHE_OVERRIDE.set(False)
                try:
                    if not hasattr(core, "_execute_pipeline"):
                        raise TypeError("ExecutorCore missing _execute_pipeline")
                    pipeline_result: PipelineResult[BaseModel] = await core._execute_pipeline(
                        body_pipeline, current_data, iteration_context, resources, limits, stream
                    )
                finally:
                    _CACHE_OVERRIDE.reset(token)
                try:
                    for st, prev_val in step_force_prev.items():
                        try:
                            if prev_val is None:
                                object.__delattr__(st, "_force_loop_fallback")
                            else:
                                object.__setattr__(st, "_force_loop_fallback", prev_val)
                        except Exception:
                            try:
                                if prev_val is None:
                                    delattr(st, "_force_loop_fallback")
                                else:
                                    setattr(st, "_force_loop_fallback", prev_val)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    object.__setattr__(core, "_inside_loop_iteration", prev_loop_flag)
                except Exception:
                    try:
                        setattr(core, "_inside_loop_iteration", prev_loop_flag)
                    except Exception:
                        pass
            except (PausedException, PipelineAbortSignal):
                clear_hitl_markers(iteration_context)
                propagate_pause_state(
                    iteration_context=iteration_context,
                    current_context=current_context,
                    iteration_count=iteration_count,
                    current_step_index=current_step_index,
                    current_data=current_data,
                    paused_step_name=paused_step_name,
                )
                raise
            except UsageLimitExceededError as e:
                # Wrap quota breaches so the top-level history contains the LoopStep itself.
                # This preserves the invariant that complex steps appear as a single StepResult
                # (with attempts/iteration metadata) even when the body is denied pre-execution.
                completed_iterations = iteration_count - 1
                try:
                    if iteration_results:
                        completed_iterations = iteration_count
                except Exception:
                    completed_iterations = iteration_count - 1
                try:
                    loop_sr = build_pooled_step_result(
                        name=loop_step.name,
                        success=False,
                        output=None,
                        attempts=max(0, int(completed_iterations)),
                        latency_s=time.monotonic() - start_time,
                        token_counts=int(prev_cumulative_tokens),
                        cost_usd=float(prev_cumulative_cost),
                        feedback=str(e),
                        branch_context=iteration_context,
                        metadata={
                            "iterations": max(0, int(completed_iterations)),
                            "exit_reason": "quota_limit_exceeded",
                        },
                        step_history=iteration_results_all + iteration_results,
                    )
                except Exception:
                    loop_sr = StepResult(
                        name=loop_step.name,
                        success=False,
                        output=None,
                        attempts=max(0, int(completed_iterations)),
                        latency_s=time.monotonic() - start_time,
                        token_counts=int(prev_cumulative_tokens),
                        cost_usd=float(prev_cumulative_cost),
                        feedback=str(e),
                        branch_context=iteration_context,
                        metadata_={
                            "iterations": max(0, int(completed_iterations)),
                            "exit_reason": "quota_limit_exceeded",
                        },
                        step_history=iteration_results_all + iteration_results,
                    )
                pr = PipelineResult[BaseModel](
                    step_history=[loop_sr],
                    total_cost_usd=float(prev_cumulative_cost),
                    total_tokens=int(prev_cumulative_tokens),
                    final_pipeline_context=current_context,
                )
                raise UsageLimitExceededError(str(e), pr)
            except Exception as e:
                telemetry.logfire.error(
                    f"LoopStep '{loop_step.name}' pipeline execution error: {e}"
                )
                return to_outcome(
                    StepResult(
                        name=loop_step.name,
                        success=False,
                        output=None,
                        attempts=iteration_count,
                        latency_s=time.monotonic() - start_time,
                        token_counts=cumulative_tokens,
                        cost_usd=cumulative_cost,
                        feedback=str(e),
                        branch_context=iteration_context,
                        metadata_={
                            "iterations": iteration_count,
                            "exit_reason": "pipeline_execution_error",
                        },
                        step_history=iteration_results_all + iteration_results,
                    )
                )
            try:
                iteration_results.extend(pipeline_result.step_history)
                current_data = pipeline_result.step_history[-1].output
            except Exception:
                pass
            try:
                final_ctx = getattr(pipeline_result, "final_pipeline_context", None)
                if final_ctx is not None:
                    try:
                        merged_ctx = ContextManager.merge(current_context, final_ctx)
                        if merged_ctx is not None:
                            current_context = merged_ctx
                    except Exception:
                        current_context = final_ctx
            except Exception:
                pass
        else:
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}': executing loop body step-by-step with {len(loop_body_steps)} steps"
            )
            pipeline_result = PipelineResult[BaseModel](
                step_history=[],
                total_cost_usd=0.0,
                total_tokens=0,
                final_pipeline_context=iteration_context,
            )
            current_step_list = (
                loop_body_steps if not stashed_exec_lists else stashed_exec_lists[-1]
            )
            # If resuming with a pending HITL payload, resume from the saved step index using
            # that payload as current data and clear the resume markers.
            if resume_requires_hitl_output:
                current_data = resume_payload
                resume_requires_hitl_output = False
                try:
                    if iteration_context is not None:
                        if hasattr(iteration_context, "loop_resume_requires_hitl_output"):
                            iteration_context.loop_resume_requires_hitl_output = False
                        if hasattr(iteration_context, "paused_step_input"):
                            iteration_context.paused_step_input = None
                        if hasattr(iteration_context, "hitl_data"):
                            iteration_context.hitl_data = {}
                        if hasattr(iteration_context, "status"):
                            iteration_context.status = "running"
                except Exception:
                    pass
                try:
                    if current_context is not None:
                        if hasattr(current_context, "loop_resume_requires_hitl_output"):
                            current_context.loop_resume_requires_hitl_output = False
                        if hasattr(current_context, "paused_step_input"):
                            current_context.paused_step_input = None
                        if hasattr(current_context, "hitl_data"):
                            current_context.hitl_data = {}
                        if hasattr(current_context, "status"):
                            current_context.status = "running"
                except Exception:
                    pass
            while current_step_index < len(current_step_list):
                body_step = current_step_list[current_step_index]
                step_name = getattr(body_step, "name", f"loop_step_{current_step_index}")
                sr: StepResult | None = None
                prev_loop_flag = getattr(core, "_inside_loop_iteration", False)
                prev_force_flag = getattr(body_step, "_force_loop_fallback", None)
                try:
                    try:
                        object.__setattr__(core, "_inside_loop_iteration", True)
                    except Exception:
                        setattr(core, "_inside_loop_iteration", True)
                except Exception:
                    pass
                try:
                    try:
                        object.__setattr__(body_step, "_force_loop_fallback", True)
                    except Exception:
                        setattr(body_step, "_force_loop_fallback", True)
                except Exception:
                    pass
                try:
                    telemetry.logfire.info(
                        f"LoopStep '{loop_step.name}' executing step {current_step_index + 1}/{len(current_step_list)}: '{step_name}'"
                    )
                    try:
                        object.__setattr__(iteration_context, "_loop_iteration_active", True)
                        object.__setattr__(
                            iteration_context, "_loop_iteration_index", iteration_count
                        )
                        object.__setattr__(
                            iteration_context, "_last_loop_iterations", iteration_count
                        )
                    except Exception:
                        try:
                            setattr(iteration_context, "_loop_iteration_active", True)
                            setattr(iteration_context, "_loop_iteration_index", iteration_count)
                            setattr(iteration_context, "_last_loop_iterations", iteration_count)
                        except Exception:
                            pass
                    try:
                        # scratchpad writes removed (deprecated); use context attrs only.
                        pass
                    except Exception:
                        pass
                    frame = make_execution_frame(
                        core,
                        body_step,
                        current_data,
                        iteration_context,
                        resources,
                        limits,
                        context_setter=context_setter,
                        stream=stream,
                        on_chunk=on_chunk,
                        fallback_depth=_fallback_depth,
                        result=None,
                        quota=core._get_current_quota()
                        if hasattr(core, "_get_current_quota")
                        else None,
                    )
                    if not hasattr(core, "execute"):
                        raise TypeError("ExecutorCore missing execute")
                    token = _CACHE_OVERRIDE.set(False)
                    try:
                        outcome = await core.execute(frame)
                    finally:
                        _CACHE_OVERRIDE.reset(token)
                    if isinstance(outcome, StepResult):
                        sr = outcome
                    elif hasattr(core, "_unwrap_outcome_to_step_result"):
                        sr = core._unwrap_outcome_to_step_result(
                            outcome, getattr(body_step, "name", "<step>")
                        )
                    else:
                        raise TypeError("ExecutorCore missing _unwrap_outcome_to_step_result")
                    if not isinstance(sr, StepResult):
                        raise TypeError(
                            f"Expected StepResult for loop body step '{step_name}', got {type(sr).__name__}"
                        )
                    iteration_results.append(sr)
                    pipeline_result.step_history.append(sr)
                    try:
                        clear_hitl_markers(iteration_context)
                    except Exception:
                        pass
                    if not sr.success:
                        if sr.feedback == "hitl_pause" or isinstance(sr.output, Paused):
                            try:
                                if iteration_context is not None:
                                    if hasattr(iteration_context, "status"):
                                        iteration_context.status = "paused"
                                    if hasattr(iteration_context, "loop_iteration_index"):
                                        iteration_context.loop_iteration_index = iteration_count
                                    if hasattr(iteration_context, "loop_step_index"):
                                        iteration_context.loop_step_index = current_step_index + 1
                                    if hasattr(iteration_context, "loop_last_output"):
                                        iteration_context.loop_last_output = current_data
                                    if hasattr(
                                        iteration_context, "loop_resume_requires_hitl_output"
                                    ):
                                        iteration_context.loop_resume_requires_hitl_output = True
                                    if hasattr(iteration_context, "loop_paused_step_name"):
                                        iteration_context.loop_paused_step_name = step_name
                                    if hasattr(iteration_context, "paused_step_input"):
                                        iteration_context.paused_step_input = sr.output

                                val = getattr(sr.output, "human_response", None)
                                if iteration_context is not None:
                                    if hasattr(iteration_context, "user_input"):
                                        iteration_context.user_input = val
                                    if hasattr(iteration_context, "hitl_data") and val is not None:
                                        # Preserve HITL payload for auditing/resume without using scratchpad
                                        iteration_context.hitl_data = {"human_response": val}

                                if current_context is not None:
                                    if hasattr(current_context, "status"):
                                        current_context.status = "paused"
                                    if (
                                        iteration_context is not None
                                        and hasattr(iteration_context, "loop_iteration_index")
                                        and hasattr(current_context, "loop_iteration_index")
                                    ):
                                        current_context.loop_iteration_index = (
                                            iteration_context.loop_iteration_index
                                        )
                                    if (
                                        iteration_context is not None
                                        and hasattr(iteration_context, "loop_step_index")
                                        and hasattr(current_context, "loop_step_index")
                                    ):
                                        current_context.loop_step_index = (
                                            iteration_context.loop_step_index
                                        )
                                    if hasattr(current_context, "loop_last_output"):
                                        current_context.loop_last_output = current_data
                                    if hasattr(current_context, "loop_resume_requires_hitl_output"):
                                        current_context.loop_resume_requires_hitl_output = True
                                    if hasattr(current_context, "loop_paused_step_name"):
                                        current_context.loop_paused_step_name = step_name
                                    if hasattr(current_context, "paused_step_input"):
                                        current_context.paused_step_input = sr.output
                                    if hasattr(current_context, "user_input"):
                                        current_context.user_input = val
                                    if hasattr(current_context, "hitl_data") and val is not None:
                                        current_context.hitl_data = {"human_response": val}
                            except Exception:
                                pass
                            raise PausedException("Loop step paused via HITL")
                        return to_outcome(
                            build_pooled_step_result(
                                name=loop_step.name,
                                success=False,
                                output=None,
                                attempts=iteration_count,
                                latency_s=time.monotonic() - start_time,
                                token_counts=sr.token_counts,
                                cost_usd=sr.cost_usd,
                                feedback=f"Loop body failed: {sr.feedback}",
                                branch_context=iteration_context,
                                metadata={
                                    "iterations": iteration_count,
                                    "exit_reason": "body_step_error",
                                },
                                step_history=iteration_results_all + iteration_results,
                            )
                        )
                    current_data = sr.output
                    if hasattr(sr, "output") and isinstance(sr.output, dict):
                        if sr.output.get("_redirect_to_step"):
                            redirect_step = sr.output["_redirect_to_step"]
                            current_step_index = next(
                                (
                                    i
                                    for i, st in enumerate(current_step_list)
                                    if getattr(st, "name", None) == redirect_step
                                ),
                                len(current_step_list),
                            )
                            continue
                    try:
                        object.__setattr__(iteration_context, "_loop_iteration_active", False)
                        object.__setattr__(
                            iteration_context, "_loop_iteration_index", iteration_count
                        )
                    except Exception:
                        try:
                            setattr(iteration_context, "_loop_iteration_active", False)
                            setattr(iteration_context, "_loop_iteration_index", iteration_count)
                        except Exception:
                            pass
                    current_step_index += 1
                    if current_step_index >= len(current_step_list) and stashed_exec_lists:
                        current_step_list = stashed_exec_lists.pop()
                        continue
                except (PausedException, PipelineAbortSignal):
                    hitl_output = getattr(sr, "output", None) if sr is not None else None
                    propagate_pause_state(
                        iteration_context=iteration_context,
                        current_context=current_context,
                        iteration_count=iteration_count,
                        current_step_index=current_step_index,
                        current_data=current_data,
                        paused_step_name=step_name,
                        hitl_output=hitl_output,
                    )
                    raise
                except Exception as e:
                    telemetry.logfire.error(
                        f"LoopStep '{loop_step.name}' error executing step '{step_name}': {e}"
                    )
                    return to_outcome(
                        build_pooled_step_result(
                            name=loop_step.name,
                            success=False,
                            output=None,
                            attempts=iteration_count,
                            latency_s=time.monotonic() - start_time,
                            token_counts=0,
                            cost_usd=cumulative_cost,
                            feedback=str(e),
                            branch_context=iteration_context,
                            metadata={
                                "iterations": iteration_count,
                                "exit_reason": "body_step_execution_error",
                            },
                            step_history=iteration_results_all + iteration_results,
                        )
                    )
                finally:
                    try:
                        try:
                            object.__setattr__(core, "_inside_loop_iteration", prev_loop_flag)
                        except Exception:
                            setattr(core, "_inside_loop_iteration", prev_loop_flag)
                    except Exception:
                        pass
                    try:
                        if prev_force_flag is None:
                            try:
                                object.__delattr__(body_step, "_force_loop_fallback")
                            except Exception:
                                delattr(body_step, "_force_loop_fallback")
                        else:
                            try:
                                object.__setattr__(
                                    body_step, "_force_loop_fallback", prev_force_flag
                                )
                            except Exception:
                                setattr(body_step, "_force_loop_fallback", prev_force_flag)
                    except Exception:
                        pass
            pipeline_result.total_cost_usd = sum(
                getattr(sr, "cost_usd", 0.0) for sr in pipeline_result.step_history
            )
            pipeline_result.total_tokens = sum(
                getattr(sr, "token_counts", 0) for sr in pipeline_result.step_history
            )
            pipeline_result.final_pipeline_context = iteration_context
            try:
                if (
                    getattr(loop_step, "name", "") == "ValidateAndRepair"
                    and hasattr(loop_step, "fallback_step")
                    and getattr(loop_step, "fallback_step") is not None
                ):
                    if any(not sr.success for sr in pipeline_result.step_history):
                        telemetry.logfire.info(
                            "LoopStep 'ValidateAndRepair' detected failure; executing fallback."
                        )
                        fb_frame = make_execution_frame(
                            core,
                            loop_step.fallback_step,
                            current_data,
                            iteration_context,
                            resources,
                            limits,
                            context_setter=context_setter,
                            stream=stream,
                            on_chunk=on_chunk,
                            fallback_depth=_fallback_depth + 1,
                            result=None,
                            quota=(
                                core._get_current_quota()
                                if hasattr(core, "_get_current_quota")
                                else None
                            ),
                        )
                        if not hasattr(core, "execute"):
                            raise TypeError("ExecutorCore missing execute")
                        fb_outcome = await core.execute(fb_frame)
                        if hasattr(core, "_unwrap_outcome_to_step_result"):
                            fb_sr = core._unwrap_outcome_to_step_result(
                                fb_outcome,
                                getattr(loop_step.fallback_step, "name", "<fallback>"),
                            )
                        else:
                            raise TypeError("ExecutorCore missing _unwrap_outcome_to_step_result")
                        if not isinstance(fb_sr, StepResult):
                            raise TypeError(
                                f"Expected StepResult for loop fallback, got {type(fb_sr).__name__}"
                            )
                        pipeline_result.step_history.append(fb_sr)
                        iteration_results.append(fb_sr)
                        current_data = getattr(fb_sr, "output", current_data)
            except Exception:
                pass
        final_ctx = getattr(pipeline_result, "final_pipeline_context", None)
        if final_ctx is not None and getattr(loop_step, "name", "") != "ValidateAndRepair":
            iteration_context = final_ctx
            try:
                merged_context = ContextManager.merge(current_context, iteration_context)
                if merged_context is not None:
                    current_context = merged_context
            except Exception as e:
                telemetry.logfire.warning(
                    f"LoopStep '{loop_step.name}' ContextManager.merge failed: {e}"
                )
        elif final_ctx is not None:
            current_context = final_ctx
        elif iteration_context is not None and current_context is not None:
            # Some pipelines mutate the provided iteration_context in place without
            # returning it as final_pipeline_context. Merge it to avoid losing init/body updates.
            try:
                merged_context = ContextManager.merge(current_context, iteration_context)
                if merged_context is not None:
                    current_context = merged_context
            except Exception as e:
                telemetry.logfire.warning(
                    f"LoopStep '{loop_step.name}' merge of iteration_context failed: {e}"
                )

        # Ensure loop exit condition is re-evaluated with the latest context flags
        if (
            getattr(iteration_context, "is_complete", False)
            and current_context is not None
            and hasattr(current_context, "is_complete")
        ):
            current_context.is_complete = True
        sync_conversation_history(
            current_context=current_context,
            iteration_context=iteration_context,
            pipeline_result=pipeline_result,
            conv_enabled=conv_enabled,
            ai_src=ai_src,
            user_src=user_src,
            named_steps_set=named_steps_set,
            hitl_step_names=hitl_step_names,
        )
        # Once an iteration completes successfully, clear resume requirements so exit checks
        # use the fresh loop output instead of a stale resume payload.
        if resume_requires_hitl_output:
            resume_requires_hitl_output = False
            if current_context is not None and hasattr(
                current_context, "loop_resume_requires_hitl_output"
            ):
                try:
                    current_context.loop_resume_requires_hitl_output = False
                except Exception:
                    pass
        iteration_results_all.extend(iteration_results)
        cumulative_cost += pipeline_result.total_cost_usd
        cumulative_tokens += pipeline_result.total_tokens
        # No reactive post-iteration limit checks; quota reservation enforces limits.
        cond = (
            loop_step.get_exit_condition_callable()
            if hasattr(loop_step, "get_exit_condition_callable")
            else getattr(loop_step, "exit_condition_callable", None)
        )
        if cond:
            try:
                should_exit = False
                data_for_cond = current_data
                try:
                    if (
                        is_resuming
                        and resume_requires_hitl_output
                        and current_step_index >= len(loop_body_steps)
                    ):
                        if resume_payload is not None:
                            data_for_cond = resume_payload
                except Exception:
                    pass
                try:
                    should_exit = bool(cond(data_for_cond, current_context))
                except TypeError:
                    should_exit = bool(cond(pipeline_result, current_context))
                try:
                    expr = getattr(loop_step, "meta", {}).get("exit_expression")
                    if expr:
                        with telemetry.logfire.span(
                            f"Loop '{loop_step.name}' - ExitCheck {iteration_count}"
                        ) as _span:
                            try:
                                _span.set_attribute("evaluated_expression", str(expr))
                                _span.set_attribute("evaluated_value", bool(should_exit))
                            except Exception:
                                pass
                        try:
                            if current_context is not None:
                                setattr(current_context, "_last_exit_expression", str(expr))
                                setattr(current_context, "_last_exit_value", bool(should_exit))
                        except Exception:
                            pass
                except Exception:
                    pass
                if should_exit:
                    telemetry.logfire.info(
                        f"LoopStep '{loop_step.name}' exit condition met at iteration {iteration_count}."
                    )
                    exit_reason = "condition"
                    break
            except Exception as e:
                return to_outcome(
                    build_pooled_step_result(
                        name=loop_step.name,
                        success=False,
                        output=None,
                        attempts=iteration_count,
                        latency_s=time.monotonic() - start_time,
                        token_counts=cumulative_tokens,
                        cost_usd=cumulative_cost,
                        feedback=f"Exception in exit condition for LoopStep '{loop_step.name}': {e}",
                        branch_context=current_context,
                        metadata={
                            "iterations": iteration_count,
                            "exit_reason": "exit_condition_error",
                        },
                        step_history=iteration_results,
                    )
                )
        iteration_count += 1
        current_step_index = 0
        if iteration_count <= max_loops:
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}' completed iteration {iteration_count - 1}, starting iteration {iteration_count}"
            )
        current_data, iter_mapper_outcome = apply_iteration_input_mapper(
            loop_step=loop_step,
            current_data=current_data,
            current_context=current_context,
            iteration_count=iteration_count,
            max_loops=max_loops,
            start_time=start_time,
            cumulative_tokens=cumulative_tokens,
            cumulative_cost=cumulative_cost,
            iteration_results_all=iteration_results_all,
            iteration_results=iteration_results,
        )
        if iter_mapper_outcome:
            return iter_mapper_outcome
    final_output = current_data
    is_map_step = hasattr(loop_step, "iterable_input")
    final_output, output_mapper_outcome = finalize_loop_output(
        loop_step=loop_step,
        core=core,
        current_data=current_data,
        final_output=final_output,
        current_context=current_context,
        iteration_count=iteration_count,
        cumulative_tokens=cumulative_tokens,
        cumulative_cost=cumulative_cost,
        iteration_results_all=iteration_results_all + iteration_results,
        is_map_step=is_map_step,
        start_time=start_time,
    )
    if output_mapper_outcome:
        return output_mapper_outcome
    any_failure = any(not sr.success for sr in iteration_results_all)
    try:
        if current_context is not None:
            try:
                object.__setattr__(current_context, "_last_loop_iterations", iteration_count)
            except Exception:
                setattr(current_context, "_last_loop_iterations", iteration_count)
    except Exception:
        pass
    try:
        bp = (
            loop_step.get_loop_body_pipeline()
            if hasattr(loop_step, "get_loop_body_pipeline")
            else getattr(loop_step, "loop_body_pipeline", None)
        )
        if bp is not None and getattr(bp, "steps", None):
            any(getattr(s, "name", "") == "_capture_artifact" for s in bp.steps)
    except Exception:
        pass
    is_map_step = hasattr(loop_step, "iterable_input")
    last_failure_fb: str | None = None
    if any_failure:
        try:
            for _sr in reversed(iteration_results_all):
                if not getattr(_sr, "success", True):
                    fb_val = getattr(_sr, "feedback", None)
                    last_failure_fb = str(fb_val) if fb_val is not None else None
                    break
        except Exception:
            last_failure_fb = None
    if is_map_step:
        success_flag = True
        feedback_msg = None
        exit_reason = "condition"
    else:
        exit_reason = "condition" if iteration_count <= max_loops else "max_loops"
        if getattr(loop_step, "name", "") == "ValidateAndRepair":
            success_flag = exit_reason == "condition"
            feedback_msg = None if success_flag else "reached max_loops"
        else:
            if exit_reason == "max_loops":
                success_flag = False
                feedback_msg = "reached max_loops"
            elif exit_reason == "condition":
                success_flag = not any_failure
                if any_failure and last_failure_fb:
                    feedback_msg = f"Loop body failed: {last_failure_fb}"
                else:
                    feedback_msg = "loop exited by condition"
            else:
                success_flag = False
                feedback_msg = "reached max_loops"
    completed_iterations = iteration_count if exit_reason == "condition" else iteration_count - 1
    if current_context is not None:
        try:
            # Typed-only cleanup (scratchpad is deprecated; no writes).
            if hasattr(current_context, "loop_iteration_index"):
                current_context.loop_iteration_index = None
            if hasattr(current_context, "loop_step_index"):
                current_context.loop_step_index = None
            if hasattr(current_context, "loop_last_output"):
                current_context.loop_last_output = None
            if hasattr(current_context, "loop_resume_requires_hitl_output"):
                current_context.loop_resume_requires_hitl_output = False
            if hasattr(current_context, "loop_paused_step_name"):
                current_context.loop_paused_step_name = None
            if getattr(current_context, "status", None) == "paused":
                if hasattr(current_context, "status"):
                    current_context.status = "completed"
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}' cleaned up resume state on completion"
            )
        except Exception as cleanup_error:
            telemetry.logfire.warning(
                f"LoopStep '{loop_step.name}' failed to clean up resume state: {cleanup_error}"
            )
    result = build_pooled_step_result(
        name=loop_step.name,
        success=success_flag,
        output=final_output,
        attempts=completed_iterations,
        latency_s=time.monotonic() - start_time,
        token_counts=cumulative_tokens,
        cost_usd=cumulative_cost,
        feedback=feedback_msg,
        branch_context=current_context,
        metadata={
            "iterations": completed_iterations,
            "exit_reason": exit_reason or "max_loops",
            **(
                {
                    "evaluated_expression": getattr(current_context, "_last_exit_expression", None),
                    "evaluated_value": getattr(current_context, "_last_exit_value", None),
                }
                if current_context is not None
                else {}
            ),
        },
        step_history=iteration_results_all,
    )
    return to_outcome(result)

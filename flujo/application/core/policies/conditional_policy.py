from __future__ import annotations

import time
from typing import Optional, Protocol, Type

from flujo.application.core.context_manager import ContextManager
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import (
    BaseModel,
    Failure,
    Paused,
    PipelineContext,
    PipelineResult,
    StepOutcome,
    StepResult,
    Success,
)
from flujo.domain.outcomes import to_outcome
from flujo.exceptions import PausedException, UsageLimitExceededError
from flujo.infra import telemetry
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame


class ConditionalStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultConditionalStepExecutor(StepPolicy[ConditionalStep[PipelineContext]]):
    @property
    def handles_type(self) -> Type[ConditionalStep[PipelineContext]]:
        return ConditionalStep

    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[BaseModel],
    ) -> StepOutcome[StepResult]:
        """Handle ConditionalStep execution with proper context isolation and merging."""
        conditional_step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        context_setter = frame.context_setter
        try:
            _fallback_depth = int(getattr(frame, "_fallback_depth", 0) or 0)
        except Exception:
            _fallback_depth = 0

        telemetry.logfire.debug("=== HANDLE CONDITIONAL STEP ===")
        telemetry.logfire.debug(
            f"Handling ConditionalStep '{getattr(conditional_step, 'name', '<unnamed>')}'"
        )
        telemetry.logfire.debug(f"Conditional step name: {conditional_step.name}")

        # Defensive name helper to avoid attr errors on lightweight cores
        def _safe_name(obj: object) -> str:
            safe_fn = getattr(core, "_safe_step_name", None)
            if callable(safe_fn):
                try:
                    return str(safe_fn(obj))
                except Exception:
                    pass
            return str(getattr(obj, "name", "<unnamed>"))

        # Initialize result
        result = StepResult(
            name=conditional_step.name,
            output=None,
            success=False,
            attempts=1,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",
            branch_context=None,
            metadata_={},
        )
        start_time = time.monotonic()
        from flujo.exceptions import PipelineAbortSignal as _Abort, PausedException as _PausedExc

        with telemetry.logfire.span(conditional_step.name) as span:
            try:
                # Avoid noisy prints during benchmarks; retain only telemetry logs
                # Evaluate branch key using the immediate previous output and current context
                # Ensure the condition sees a meaningful payload even when the last output
                # is not a mapping by augmenting with context-derived signals.
                # Use original data and context for condition evaluation (contract)
                branch_key = conditional_step.condition_callable(data, context)
                # FSD-026: tolerant resolution for boolean expressions.
                # Prefer exact boolean keys (DSL usage), else fallback to 'true'/'false' strings (YAML usage).
                resolved_key = None
                if isinstance(branch_key, bool):
                    for cand in (branch_key, str(branch_key).lower()):
                        if cand in getattr(conditional_step, "branches", {}):
                            resolved_key = cand
                            break
                else:
                    if branch_key in getattr(conditional_step, "branches", {}):
                        resolved_key = branch_key
                try:
                    expr = getattr(conditional_step, "meta", {}).get("condition_expression")
                    if expr:
                        try:
                            span.set_attribute("evaluated_expression", str(expr))
                            span.set_attribute("evaluated_value", str(branch_key))
                        except Exception:
                            pass
                        try:
                            result.metadata_["evaluated_expression"] = str(expr)
                            result.metadata_["evaluated_value"] = branch_key
                        except Exception:
                            pass
                except Exception:
                    pass
                # Architect-specific safety: ensure ValidityBranch honors context validity/shape
                try:
                    if (
                        getattr(conditional_step, "name", "") == "ValidityBranch"
                        and branch_key != "valid"
                    ):
                        ctx_text = getattr(context, "yaml_text", None)

                        # Quick shape check: unmatched inline list is invalid; otherwise treat as valid
                        def _shape_invalid(text: object) -> bool:
                            if not isinstance(text, str) or "steps:" not in text:
                                return False
                            try:
                                line = text.split("steps:", 1)[1].splitlines()[0]
                            except Exception:
                                line = ""
                            return ("[" in line and "]" not in line) and ("[]" not in line)

                        yaml_flag = False
                        try:
                            yaml_flag = bool(getattr(context, "yaml_is_valid", False))
                        except Exception:
                            yaml_flag = False
                        if (
                            isinstance(ctx_text, str)
                            and ctx_text.strip()
                            and not _shape_invalid(ctx_text)
                        ) or yaml_flag:
                            branch_key = "valid"
                except Exception:
                    pass
                telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}'")
                try:
                    span.set_attribute("executed_branch_key", branch_key)
                    if resolved_key is not None and resolved_key is not branch_key:
                        span.set_attribute("resolved_branch_key", str(resolved_key))
                except Exception:
                    pass
                # Determine branch using resolved key when present; otherwise use evaluated branch_key
                branch_to_execute = None
                target_key = resolved_key if resolved_key is not None else branch_key
                if target_key in conditional_step.branches:
                    branch_to_execute = conditional_step.branches[target_key]
                elif conditional_step.default_branch_pipeline is not None:
                    branch_to_execute = conditional_step.default_branch_pipeline
                else:
                    # Attempt stringified key lookup for bool/int keys common in YAML
                    try:
                        key_str = str(branch_key).lower()
                        for k, v in (conditional_step.branches or {}).items():
                            if str(k).lower() == key_str:
                                branch_to_execute = v
                                resolved_key = k
                                break
                    except Exception:
                        pass
                if branch_to_execute is None:
                    telemetry.logfire.warn(
                        f"No branch found for key '{branch_key}' and no default branch provided"
                    )
                    result.success = False
                    result.metadata_["executed_branch_key"] = branch_key
                    result.feedback = (
                        f"No branch found for key '{branch_key}' and no default branch provided"
                    )
                    result.latency_s = time.monotonic() - start_time
                    return to_outcome(result)
                # Record executed branch key (always the evaluated key, even when default is used)
                result.metadata_["executed_branch_key"] = branch_key
                if resolved_key is not None and resolved_key is not branch_key:
                    result.metadata_["resolved_branch_key"] = resolved_key
                telemetry.logfire.info(f"Executing branch for key '{branch_key}'")
                branch_data = data
                # Detect HITL branches: pause only when no human input is available yet
                force_repause_after_branch = False
                resume_requires_hitl = False
                try:
                    from flujo.domain.dsl.step import HumanInTheLoopStep as _HITLStep

                    branch_steps = (
                        branch_to_execute.steps
                        if isinstance(branch_to_execute, Pipeline)
                        else [branch_to_execute]
                    )
                    has_hitl = any(isinstance(_s, _HITLStep) for _s in branch_steps)
                    if has_hitl:
                        has_input = False
                        if context is not None:
                            has_input = getattr(context, "user_input", None) is not None or bool(
                                getattr(context, "hitl_data", {}) or {}
                            )
                            resume_requires_hitl = bool(
                                getattr(context, "loop_resume_requires_hitl_output", False)
                            )
                        if not has_input:
                            msg = None
                            for _s in branch_steps:
                                if isinstance(_s, _HITLStep):
                                    msg = getattr(_s, "message", None) or getattr(
                                        _s, "message_for_user", None
                                    )
                                    break
                            if context is not None:
                                if hasattr(context, "status"):
                                    context.status = "paused"
                                if hasattr(context, "paused_step_input"):
                                    context.paused_step_input = data
                                if hasattr(context, "pause_message") and msg:
                                    context.pause_message = msg
                            raise PausedException(msg or "Awaiting human input")
                        else:
                            # If we have user_input from resume, feed it into branch_data for HITL step
                            try:
                                if context is not None:
                                    branch_data = getattr(context, "user_input", branch_data)
                                if resume_requires_hitl:
                                    # Preserve paused status and input so the HITL policy can auto-consume it.
                                    if context is not None:
                                        if hasattr(context, "status"):
                                            context.status = "paused"
                                        if (
                                            hasattr(context, "user_input")
                                            and getattr(context, "user_input", None) is None
                                        ):
                                            context.user_input = branch_data
                                        if (
                                            hasattr(context, "hitl_data")
                                            and branch_data is not None
                                        ):
                                            context.hitl_data = {"human_response": branch_data}
                                else:
                                    # Consume the user_input so subsequent HITL steps in later branches pause again.
                                    if context is not None:
                                        if hasattr(context, "user_input"):
                                            context.user_input = None
                                        if hasattr(context, "hitl_data"):
                                            context.hitl_data = {}
                                    try:
                                        # Use typed field for status
                                        if context is not None and hasattr(context, "status"):
                                            context.status = "running"
                                    except Exception:
                                        pass
                                # Apply sink_to for the first HITL in this branch
                                try:
                                    first_hitl = next(
                                        _s for _s in branch_steps if isinstance(_s, _HITLStep)
                                    )
                                    sink_target = getattr(first_hitl, "sink_to", None)
                                    if isinstance(sink_target, str) and context is not None:
                                        from flujo.utils.context import set_nested_context_field

                                        set_nested_context_field(context, sink_target, branch_data)
                                except Exception:
                                    pass
                                # Force a re-pause if more HITLs remain in this branch
                                remaining_hitl_count = sum(
                                    1 for _s in branch_steps if isinstance(_s, _HITLStep)
                                )
                                if remaining_hitl_count > 1:
                                    # Use typed field for status
                                    if context is not None and hasattr(context, "status"):
                                        context.status = "paused"
                                    if (
                                        context is not None
                                        and hasattr(context, "pause_message")
                                        and getattr(context, "pause_message", None) is None
                                    ):
                                        context.pause_message = getattr(
                                            first_hitl, "message", None
                                        ) or getattr(first_hitl, "message_for_user", "Paused")
                                    raise PausedException("Awaiting next HITL input")
                            except Exception:
                                pass
                except PausedException:
                    raise
                except Exception:
                    pass

                # Execute selected branch
                if branch_to_execute:
                    execute_fn = getattr(core, "execute", None)
                    if not callable(execute_fn):
                        raise TypeError("ExecutorCore missing execute()")
                    if conditional_step.branch_input_mapper:
                        branch_data = conditional_step.branch_input_mapper(branch_data, context)
                    # Use ContextManager for proper deep isolation
                    branch_context = (
                        ContextManager.isolate(context) if context is not None else None
                    )
                    # Execute pipeline
                    total_cost = 0.0
                    total_tokens = 0
                    total_latency = 0.0
                    step_history = []
                    step_result: Optional[StepResult] = None
                    res_any = None
                    for pipeline_step in (
                        branch_to_execute.steps
                        if isinstance(branch_to_execute, Pipeline)
                        else [branch_to_execute]
                    ):
                        # Span around the concrete branch step to expose its name for tests
                        with telemetry.logfire.span(
                            getattr(pipeline_step, "name", str(pipeline_step))
                        ):
                            try:
                                res_any = await execute_fn(
                                    pipeline_step,
                                    branch_data,
                                    context=branch_context,
                                    resources=resources,
                                    limits=limits,
                                    context_setter=context_setter,
                                    _fallback_depth=_fallback_depth,
                                )
                            except (_Abort, _PausedExc):
                                # Always propagate control-flow so the runner can pause/abort.
                                # Best-effort: merge branch context back to the parent before bubbling.
                                from flujo.domain.dsl.step import HumanInTheLoopStep as _HITL

                                is_hitl = isinstance(pipeline_step, _HITL)
                                if context is not None and branch_context is not None:
                                    try:
                                        from flujo.utils.context import (
                                            safe_merge_context_updates as _merge,
                                        )

                                        merged = _merge(context, branch_context)
                                        if merged is False:
                                            try:
                                                merged_ctx = ContextManager.merge(
                                                    context, branch_context
                                                )
                                                if merged_ctx is not None and isinstance(
                                                    merged_ctx, BaseModel
                                                ):
                                                    context = merged_ctx
                                            except Exception:
                                                pass
                                    except Exception:
                                        try:
                                            merged_ctx = ContextManager.merge(
                                                context, branch_context
                                            )
                                            if (
                                                merged_ctx is not None
                                                and is_hitl
                                                and isinstance(merged_ctx, BaseModel)
                                            ):
                                                context = merged_ctx
                                        except Exception:
                                            pass
                                raise
                        # Normalize StepOutcome to StepResult, and propagate Paused
                        if isinstance(res_any, StepOutcome):
                            if isinstance(res_any, Success):
                                step_result = res_any.step_result
                                if not isinstance(step_result, StepResult) or getattr(
                                    step_result, "name", None
                                ) in (None, "<unknown>", ""):
                                    step_result = StepResult(
                                        name=_safe_name(pipeline_step),
                                        output=None,
                                        success=False,
                                        feedback="Missing step_result",
                                    )
                            elif isinstance(res_any, Failure):
                                step_result = res_any.step_result or StepResult(
                                    name=_safe_name(pipeline_step),
                                    success=False,
                                    feedback=res_any.feedback,
                                )
                            elif isinstance(res_any, Paused):
                                if context is not None and branch_context is not None:
                                    try:
                                        from flujo.utils.context import safe_merge_context_updates

                                        merged = safe_merge_context_updates(context, branch_context)
                                        if merged is False:
                                            try:
                                                merged_ctx = ContextManager.merge(
                                                    context, branch_context
                                                )
                                                if merged_ctx is not None and isinstance(
                                                    merged_ctx, BaseModel
                                                ):
                                                    context = merged_ctx
                                            except Exception:
                                                pass
                                    except Exception:
                                        try:
                                            merged_ctx = ContextManager.merge(
                                                context, branch_context
                                            )
                                            if merged_ctx is not None and isinstance(
                                                merged_ctx, BaseModel
                                            ):
                                                context = merged_ctx
                                        except Exception:
                                            pass
                                return res_any
                            else:
                                step_result = StepResult(
                                    name=_safe_name(pipeline_step),
                                    success=False,
                                    feedback="Unsupported outcome",
                                )
                        else:
                            step_result = res_any
                        if step_result is None:
                            continue
                    if force_repause_after_branch and isinstance(context, PipelineContext):
                        try:
                            # Use typed field for status
                            if hasattr(context, "status"):
                                context.status = "paused"
                            if hasattr(context, "pause_message"):
                                context.pause_message = getattr(conditional_step, "name", "")
                        except Exception:
                            pass
                        raise PausedException("Awaiting next HITL input")
                    if step_result is None:
                        step_result = StepResult(
                            name=_safe_name(branch_to_execute),
                            output=branch_data,
                            success=True,
                            attempts=1,
                            latency_s=total_latency,
                            token_counts=total_tokens,
                            cost_usd=total_cost,
                            branch_context=branch_context,
                            metadata_={"executed_branch_key": branch_key},
                        )
                    try:
                        if getattr(step_result, "branch_context", None) is not None:
                            branch_context = step_result.branch_context
                    except Exception:
                        pass
                    total_cost += step_result.cost_usd
                    total_tokens += step_result.token_counts
                    total_latency += getattr(step_result, "latency_s", 0.0)
                    branch_data = step_result.output
                    if not step_result.success:
                        # Propagate branch failure details in feedback
                        msg = step_result.feedback or "Step execution failed"
                        result.feedback = f"Failure in branch '{branch_key}': {msg}"
                        result.success = False
                        result.latency_s = total_latency
                        result.token_counts = total_tokens
                        result.cost_usd = total_cost
                        return to_outcome(result)
                    step_history.append(step_result)
                    res_any = step_result
                    # Handle empty branch pipelines by short-circuiting success
                    if not step_history and (
                        isinstance(branch_to_execute, Pipeline)
                        and not getattr(branch_to_execute, "steps", None)
                    ):
                        step_result = StepResult(
                            name=_safe_name(branch_to_execute),
                            success=True,
                            output=branch_data,
                            attempts=1,
                            latency_s=total_latency,
                            token_counts=total_tokens,
                            cost_usd=total_cost,
                            branch_context=branch_context,
                            metadata_={"executed_branch_key": branch_key},
                        )
                        step_history.append(step_result)
                    # If branch had no executable steps, treat as no-op success
                    if (step_result is None or not step_history) and isinstance(
                        branch_to_execute, Pipeline
                    ):
                        result.success = True
                        result.output = branch_data
                        result.latency_s = total_latency
                        result.token_counts = total_tokens
                        result.cost_usd = total_cost
                        result.branch_context = branch_context
                        result.metadata_["executed_branch_key"] = branch_key
                        return to_outcome(result)

                    # Apply optional branch_output_mapper
                    final_output = branch_data
                    if getattr(conditional_step, "branch_output_mapper", None):
                        try:
                            final_output = conditional_step.branch_output_mapper(
                                final_output, branch_key, branch_context
                            )
                        except Exception as e:
                            result.success = False
                            result.feedback = f"Branch output mapper raised an exception: {e}"
                            result.latency_s = total_latency
                            result.token_counts = total_tokens
                            result.cost_usd = total_cost
                            return to_outcome(result)
                    result.success = True
                    result.output = final_output
                    result.latency_s = total_latency
                    result.token_counts = total_tokens
                    result.cost_usd = total_cost
                    if resume_requires_hitl and isinstance(context, PipelineContext):
                        try:
                            if hasattr(context, "loop_resume_requires_hitl_output"):
                                context.loop_resume_requires_hitl_output = False
                            if hasattr(context, "user_input"):
                                context.user_input = None
                            if hasattr(context, "hitl_data"):
                                context.hitl_data = {}
                        except Exception:
                            pass
                    # Update branch context using ContextManager and propagate into parent
                    merged_ctx = (
                        ContextManager.merge(context, branch_context)
                        if context is not None
                        else branch_context
                    )
                    result.branch_context = merged_ctx
                    if merged_ctx is not None and context is not None and merged_ctx is not context:
                        try:
                            # Ensure parent context reflects merged typed fields.
                            ContextManager.merge(context, merged_ctx)
                        except Exception:
                            pass
                    # Ensure HITL user input set on parent context when available in branch
                    try:
                        if (
                            context is not None
                            and branch_context is not None
                            and getattr(context, "user_input", None) is None
                        ):
                            val = getattr(branch_context, "user_input", None)
                            if val is not None and hasattr(context, "user_input"):
                                context.user_input = val
                    except Exception:
                        pass
                    # Invoke context setter on success when provided
                    if context_setter is not None:
                        try:
                            pipeline_result: PipelineResult[BaseModel] = PipelineResult(
                                step_history=step_history,
                                total_cost_usd=total_cost,
                                total_tokens=total_tokens,
                                final_pipeline_context=result.branch_context,
                            )
                            context_setter(pipeline_result, context)
                        except Exception:
                            pass
                    return to_outcome(result)
            except (_Abort, _PausedExc):
                # Bubble up pauses so the runner marks pipeline paused
                raise
            except UsageLimitExceededError:
                # Quota enforcement is orchestration-level; never convert it to a step failure.
                raise
            except Exception as e:
                # Log error for visibility in tests - include the original error message
                error_msg = str(e)
                try:
                    telemetry.logfire.error(error_msg)
                except Exception:
                    pass
                result.feedback = f"Error executing conditional logic or branch: {e}"
                result.success = False
        result.latency_s = time.monotonic() - start_time
        return to_outcome(result)


## Legacy adapter protocol removed: ConditionalStepExecutorOutcomes


## Legacy adapter removed: DefaultConditionalStepExecutorOutcomes (native outcomes supported)

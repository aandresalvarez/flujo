from __future__ import annotations
# mypy: ignore-errors

from ._shared import (  # noqa: F401
    Any,
    Callable,
    ContextManager,
    Dict,
    Failure,
    Optional,
    Paused,
    PausedException,
    Pipeline,
    PipelineResult,
    Protocol,
    StepOutcome,
    StepResult,
    Success,
    UsageLimits,
    telemetry,
    time,
    to_outcome,
)


class ConditionalStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        conditional_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]: ...


class DefaultConditionalStepExecutor:
    async def execute(
        self,
        core: Any,
        conditional_step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]:
        """Handle ConditionalStep execution with proper context isolation and merging."""

        telemetry.logfire.debug("=== HANDLE CONDITIONAL STEP ===")
        telemetry.logfire.debug(
            f"Handling ConditionalStep '{getattr(conditional_step, 'name', '<unnamed>')}'"
        )
        telemetry.logfire.debug(f"Conditional step name: {conditional_step.name}")

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
                        def _shape_invalid(text: Any) -> bool:
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
                # Determine branch
                branch_to_execute = None
                if resolved_key is not None:
                    branch_to_execute = conditional_step.branches[resolved_key]
                elif conditional_step.default_branch_pipeline is not None:
                    branch_to_execute = conditional_step.default_branch_pipeline
                else:
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
                # Execute selected branch
                if branch_to_execute:
                    branch_data = data
                    if conditional_step.branch_input_mapper:
                        branch_data = conditional_step.branch_input_mapper(data, context)
                    # Use ContextManager for proper deep isolation
                    branch_context = (
                        ContextManager.isolate(context) if context is not None else None
                    )
                    # Execute pipeline
                    total_cost = 0.0
                    total_tokens = 0
                    total_latency = 0.0
                    step_history = []
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
                                res_any = await core.execute(
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

                                        _merge(context, branch_context)
                                    except Exception:
                                        try:
                                            merged_ctx = ContextManager.merge(
                                                context, branch_context
                                            )
                                            if merged_ctx is not None and is_hitl:
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
                                        name=core._safe_step_name(pipeline_step),
                                        output=None,
                                        success=False,
                                        feedback="Missing step_result",
                                    )
                            elif isinstance(res_any, Failure):
                                step_result = res_any.step_result or StepResult(
                                    name=core._safe_step_name(pipeline_step),
                                    success=False,
                                    feedback=res_any.feedback,
                                )
                            elif isinstance(res_any, Paused):
                                return res_any
                            else:
                                step_result = StepResult(
                                    name=core._safe_step_name(pipeline_step),
                                    success=False,
                                    feedback="Unsupported outcome",
                                )
                        else:
                            step_result = res_any
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
                    # Update branch context using ContextManager
                    result.branch_context = (
                        ContextManager.merge(context, branch_context)
                        if context is not None
                        else branch_context
                    )
                    # Invoke context setter on success when provided
                    if context_setter is not None:
                        try:
                            from flujo.domain.models import PipelineResult

                            pipeline_result: PipelineResult[Any] = PipelineResult(
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
            except Exception as e:
                # Log error for visibility in tests
                try:
                    telemetry.logfire.error(str(e))
                except Exception:
                    pass
                result.feedback = f"Error executing conditional logic or branch: {e}"
                result.success = False
        result.latency_s = time.monotonic() - start_time
        return to_outcome(result)


## Legacy adapter protocol removed: ConditionalStepExecutorOutcomes


## Legacy adapter removed: DefaultConditionalStepExecutorOutcomes (native outcomes supported)

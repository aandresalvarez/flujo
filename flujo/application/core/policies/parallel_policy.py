from __future__ import annotations
# mypy: ignore-errors

from typing import Type

from ._shared import (  # noqa: F401
    Any,
    Awaitable,
    Callable,
    BranchFailureStrategy,
    ConfigurationError,
    ContextManager,
    Dict,
    Failure,
    InfiniteRedirectError,
    List,
    MockDetectionError,
    Optional,
    ParallelStep,
    Paused,
    PausedException,
    Pipeline,
    PipelineResult,
    PipelineAbortSignal,
    PricingNotConfiguredError,
    Protocol,
    Quota,
    Success,
    StepOutcome,
    StepResult,
    Tuple,
    UsageLimitExceededError,
    UsageLimits,
    MergeStrategy,
    asyncio,
    telemetry,
    time,
    to_outcome,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame


# --- Parallel Step Executor policy ---
class ParallelStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        parallel_step: Optional[ParallelStep[Any]] = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepOutcome[StepResult]: ...


class DefaultParallelStepExecutor(StepPolicy[ParallelStep]):
    @property
    def handles_type(self) -> Type[ParallelStep]:
        return ParallelStep

    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        parallel_step: Optional[ParallelStep[Any]] = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepOutcome[StepResult]:
        if isinstance(step, ExecutionFrame):
            frame = step
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            context_setter = getattr(frame, "context_setter", None)

        # Actual parallel-step execution logic extracted from legacy `_handle_parallel_step`
        if parallel_step is not None:
            step = parallel_step
        if not isinstance(step, ParallelStep):
            raise ValueError(f"Expected ParallelStep, got {type(step)}")
        parallel_step = step
        telemetry.logfire.debug(f"=== HANDLING PARALLEL STEP === {parallel_step.name}")
        telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")
        result = StepResult(name=parallel_step.name)
        result.metadata_ = {}
        start_time = time.monotonic()
        # Handle empty branches
        if not parallel_step.branches:
            result.success = False
            result.feedback = "Parallel step has no branches to execute"
            result.output = {}
            result.latency_s = time.monotonic() - start_time
            return to_outcome(result)
        # FSD-009: Pure quota-only mode
        # Do not use breach_event or any legacy governor; safety via reservations only
        # Deterministic quota splitting per branch
        branch_items: List[Tuple[str, Any]] = list(parallel_step.branches.items())
        branch_names: List[str] = [bn for bn, _ in branch_items]
        branch_pipelines: List[Any] = [bp for _, bp in branch_items]
        branch_quota_map: Dict[str, Optional[Quota]] = {bn: None for bn in branch_names}
        try:
            current_quota = (
                core._get_current_quota() if hasattr(core, "_get_current_quota") else None
            )
        except Exception:
            current_quota = None
        if current_quota is not None and len(branch_items) > 0:
            try:
                sub_quotas = current_quota.split(len(branch_items))
                for idx, bn in enumerate(branch_names):
                    branch_quota_map[bn] = sub_quotas[idx]
            except Exception:
                # Fallback: no split if quota not available
                pass
        # Tracking variables
        branch_results: Dict[str, StepResult] = {}
        branch_contexts: Dict[str, Any] = {}
        total_cost = 0.0
        total_tokens = 0
        all_successful = True
        failure_messages: List[str] = []
        # Prepare branch contexts with proper isolation
        # Phase 1: Mandatory isolation for parallel branches with verification
        for branch_name, branch_pipeline in parallel_step.branches.items():
            # Use ContextManager for proper deep isolation with purpose tracking
            branch_context = (
                ContextManager.isolate(
                    context,
                    include_keys=parallel_step.context_include_keys,
                    purpose=f"parallel_branch:{branch_name}",
                )
                if context is not None
                else None
            )

            # Phase 1: Verify isolation before execution (strict mode)
            if context is not None and branch_context is not None:
                ContextManager.verify_isolation(context, branch_context)

            branch_contexts[branch_name] = branch_context

        def _merge_branch_context_into_parent(branch_ctx: Any) -> None:
            nonlocal context
            if context is None or branch_ctx is None:
                return
            try:
                from flujo.utils.context import safe_merge_context_updates as _merge

                merged = _merge(context, branch_ctx)
                if merged is False:
                    try:
                        merged_ctx = ContextManager.merge(context, branch_ctx)
                        if merged_ctx is not None:
                            context = merged_ctx
                    except Exception:
                        pass
            except Exception:
                try:
                    merged_ctx = ContextManager.merge(context, branch_ctx)
                    if merged_ctx is not None:
                        context = merged_ctx
                except Exception:
                    pass

        # Branch executor
        async def execute_branch(
            branch_name: str,
            branch_pipeline: Any,
            branch_context: Any,
            branch_quota: Optional[Quota],
        ) -> Tuple[str, StepResult]:
            try:
                telemetry.logfire.debug(f"Executing branch: {branch_name}")
                # Set per-branch quota in this task's context
                quota_token = None
                try:
                    if hasattr(core, "_set_current_quota"):
                        quota_token = core._set_current_quota(branch_quota)
                    elif hasattr(core, "_quota_manager"):
                        quota_token = core._quota_manager.set_current_quota(branch_quota)
                except Exception:
                    quota_token = None
                # Phase 1: Verify isolation before execution (strict mode)
                if context is not None and branch_context is not None:
                    ContextManager.verify_isolation(context, branch_context)

                if step_executor is not None:
                    target = branch_pipeline
                    try:
                        if isinstance(branch_pipeline, Pipeline) and getattr(
                            branch_pipeline, "steps", None
                        ):
                            steps = list(getattr(branch_pipeline, "steps") or [])
                            if steps:
                                # Expose the first step's agent for executors that expect it
                                first_step = steps[0]
                                if getattr(first_step, "agent", None) is not None:
                                    try:
                                        setattr(branch_pipeline, "agent", first_step.agent)
                                    except Exception:
                                        pass
                    except Exception:
                        target = branch_pipeline
                    try:
                        branch_result = await step_executor(
                            target,
                            data,
                            branch_context,
                            resources,
                        )
                    except AttributeError as exc:
                        # Some custom executors expect a Step with an `agent` attribute;
                        # retry with the first step from the pipeline when available.
                        if (
                            isinstance(target, Pipeline)
                            and getattr(target, "steps", None)
                            and "agent" in str(exc)
                        ):
                            steps = list(getattr(target, "steps") or [])
                            if steps:
                                branch_result = await step_executor(
                                    steps[0],
                                    data,
                                    branch_context,
                                    resources,
                                )
                        else:
                            raise
                else:
                    # Delegate depending on type: Pipeline vs Step
                    if isinstance(branch_pipeline, Pipeline):
                        pipeline_result = await core._execute_pipeline_via_policies(
                            branch_pipeline,
                            data,
                            branch_context,
                            resources,
                            limits,
                            context_setter,
                        )
                    else:
                        # Execute a single Step via core and synthesize PipelineResult-like view
                        step_outcome = await core.execute(
                            step=branch_pipeline,
                            data=data,
                            context=branch_context,
                            resources=resources,
                            limits=limits,
                            context_setter=context_setter,
                        )
                        if isinstance(step_outcome, Success):
                            sr = step_outcome.step_result
                            if not isinstance(sr, StepResult) or getattr(sr, "name", None) in (
                                None,
                                "<unknown>",
                                "",
                            ):
                                sr = StepResult(
                                    name=getattr(branch_pipeline, "name", "<unnamed>"),
                                    success=False,
                                    feedback="Missing step_result",
                                )
                            pipeline_result = PipelineResult(
                                step_history=[sr],
                                total_cost_usd=sr.cost_usd,
                                total_tokens=sr.token_counts,
                                final_pipeline_context=branch_context,
                            )
                        elif isinstance(step_outcome, Failure):
                            sr = step_outcome.step_result or StepResult(
                                name=getattr(branch_pipeline, "name", "<unnamed>"),
                                success=False,
                                feedback=step_outcome.feedback,
                            )
                            pipeline_result = PipelineResult(
                                step_history=[sr],
                                total_cost_usd=sr.cost_usd,
                                total_tokens=sr.token_counts,
                                final_pipeline_context=branch_context,
                            )
                        elif isinstance(step_outcome, Paused):
                            # Propagate control-flow and preserve branch context
                            _merge_branch_context_into_parent(branch_context)
                            raise PausedException(step_outcome.message)
                        else:
                            # Unknown/Chunk/Aborted -> synthesize failure
                            sr = StepResult(
                                name=getattr(branch_pipeline, "name", "<unnamed>"),
                                success=False,
                                feedback=f"Unsupported outcome type: {type(step_outcome).__name__}",
                            )
                            pipeline_result = PipelineResult(
                                step_history=[sr],
                                total_cost_usd=0.0,
                                total_tokens=0,
                                final_pipeline_context=branch_context,
                            )
                    pipeline_success = (
                        all(s.success for s in pipeline_result.step_history)
                        if pipeline_result.step_history
                        else False
                    )

                    # Enhanced feedback aggregation for branch failures
                    branch_feedback = ""
                    if pipeline_result.step_history:
                        failed_steps = [s for s in pipeline_result.step_history if not s.success]
                        if failed_steps:
                            # Aggregate detailed failure information
                            failure_details = []
                            for failed_step in failed_steps:
                                step_detail = f"step '{failed_step.name}'"
                                if failed_step.attempts > 1:
                                    step_detail += f" (after {failed_step.attempts} attempts)"
                                if failed_step.feedback:
                                    step_detail += f": {failed_step.feedback}"
                                failure_details.append(step_detail)
                            branch_feedback = f"Pipeline failed - {'; '.join(failure_details)}"
                        else:
                            branch_feedback = (
                                pipeline_result.step_history[-1].feedback
                                if pipeline_result.step_history[-1].feedback
                                else ""
                            )

                    branch_result = StepResult(
                        name=f"{parallel_step.name}_{branch_name}",
                        output=(
                            pipeline_result.step_history[-1].output
                            if pipeline_result.step_history
                            else None
                        ),
                        success=pipeline_success,
                        attempts=1,
                        latency_s=sum(s.latency_s for s in pipeline_result.step_history),
                        token_counts=pipeline_result.total_tokens,
                        cost_usd=pipeline_result.total_cost_usd,
                        feedback=branch_feedback,
                        branch_context=pipeline_result.final_pipeline_context,
                        metadata_={
                            "failed_steps_count": len(
                                [s for s in pipeline_result.step_history if not s.success]
                            ),
                            "total_steps_count": len(pipeline_result.step_history),
                        },
                    )
                # No reactive post-branch checks in pure quota mode
                telemetry.logfire.debug(
                    f"Branch {branch_name} completed: success={branch_result.success}"
                )
                return branch_name, branch_result
            except (
                MockDetectionError,
                InfiniteRedirectError,
                PricingNotConfiguredError,
            ) as e:
                # Re-raise control-flow and config exceptions unmodified
                telemetry.logfire.info(
                    f"Branch {branch_name} encountered control-flow/config exception: {type(e).__name__}"
                )
                raise
            except UsageLimitExceededError as e:
                # Re-raise usage limit exceptions - these should not be converted to branch failures
                telemetry.logfire.info(f"Branch {branch_name} hit usage limit: {e}")
                raise e
            except PausedException:
                _merge_branch_context_into_parent(branch_context)
                raise
            except PipelineAbortSignal:
                _merge_branch_context_into_parent(branch_context)
                raise
            except Exception as e:
                telemetry.logfire.error(f"Branch {branch_name} failed with exception: {e}")
                failure = StepResult(
                    name=f"{parallel_step.name}_{branch_name}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed with {type(e).__name__}: {str(e)}",
                    branch_context=context,
                    metadata_={"exception_type": type(e).__name__},
                )
                return branch_name, failure
            finally:
                try:
                    if "quota_token" in locals() and quota_token is not None:
                        if hasattr(core, "_reset_current_quota"):
                            core._reset_current_quota(quota_token)
                        elif hasattr(core, "_quota_manager") and hasattr(quota_token, "old_value"):
                            core._quota_manager.set_current_quota(quota_token.old_value)  # type: ignore[attr-defined]
                except Exception:
                    pass

        # Execute branches concurrently using the shared quota, and proactively cancel on breach
        pending: set[asyncio.Task] = set()
        task_to_branch: dict[asyncio.Task, str] = {}
        for bn, bp in zip(branch_names, branch_pipelines):
            t = asyncio.create_task(
                execute_branch(bn, bp, branch_contexts[bn], branch_quota_map.get(bn))
            )
            pending.add(t)
            task_to_branch[t] = bn

        async def _handle_branch_result(branch_execution_result: Any, idx: int) -> None:
            nonlocal total_cost, total_tokens, all_successful
            branch_name_local = list(parallel_step.branches.keys())[idx]
            if isinstance(
                branch_execution_result,
                (
                    UsageLimitExceededError,
                    MockDetectionError,
                    InfiniteRedirectError,
                    PricingNotConfiguredError,
                ),
            ):
                telemetry.logfire.info(
                    f"Parallel branch hit usage limit, re-raising: {branch_execution_result}"
                )
                raise branch_execution_result
            if isinstance(branch_execution_result, Exception):
                telemetry.logfire.error(
                    f"Parallel branch raised unexpected exception: {branch_execution_result}"
                )
                raise branch_execution_result
            if isinstance(branch_execution_result, tuple) and len(branch_execution_result) == 2:
                bn2, branch_result = branch_execution_result
                branch_name_local = bn2
            else:
                telemetry.logfire.error(
                    f"Unexpected result format from branch {branch_name_local}: {branch_execution_result}"
                )
                branch_result = StepResult(
                    name=f"{parallel_step.name}_{branch_name_local}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed with unexpected result format: {branch_execution_result}",
                    metadata_={},
                )
            if isinstance(branch_result, Exception):
                telemetry.logfire.error(
                    f"Branch {branch_name_local} raised exception: {branch_result}"
                )
                branch_result = StepResult(
                    name=f"{parallel_step.name}_{branch_name_local}",
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Branch execution failed: {branch_result}",
                    metadata_={},
                )
            branch_results[branch_name_local] = branch_result
            if branch_result.success:
                total_cost += branch_result.cost_usd
                total_tokens += branch_result.token_counts
            else:
                all_successful = False
                failure_messages.append(
                    f"branch '{branch_name_local}' failed: {branch_result.feedback}"
                )

        # Consume tasks as they complete; cancel the rest if limits are breached
        completed_count = 0
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Process all finished tasks, aggregating successful results first.
            usage_limit_error: UsageLimitExceededError | None = None
            usage_limit_error_msg: str | None = None
            pause_message: str | None = None
            pause_branch: str | None = None
            abort_branch: str | None = None
            abort_signal: PipelineAbortSignal | None = None
            for d in done:
                branch_hint = task_to_branch.get(d)
                try:
                    res = d.result()
                except PausedException as paused_exc:
                    pause_message = getattr(paused_exc, "message", "")
                    pause_branch = branch_hint
                    break
                except PipelineAbortSignal as abort_exc:
                    abort_signal = abort_exc
                    abort_branch = branch_hint
                    break
                except UsageLimitExceededError as ex:
                    # Defer raising until we aggregate any other completed successes
                    usage_limit_error = ex
                    try:
                        usage_limit_error_msg = str(ex)
                    except Exception:
                        usage_limit_error_msg = None
                    continue
                except Exception:
                    # On ANY other exception from a branch, cancel all remaining branches immediately
                    for p in pending:
                        p.cancel()
                    try:
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                    except Exception:
                        pass
                    raise
                await _handle_branch_result(res, completed_count)
                completed_count += 1

            if pause_message is not None:
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                if pause_branch:
                    telemetry.logfire.info(
                        f"Parallel branch '{pause_branch}' paused: {pause_message}"
                    )
                return Paused(message=pause_message or "Paused")

            if abort_signal is not None:
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                if abort_branch:
                    telemetry.logfire.info(
                        f"Parallel branch '{abort_branch}' triggered abort: {abort_signal}"
                    )
                raise abort_signal

            # If a usage limit breach occurred in any completed branch, cancel the rest and
            # raise an error that includes the aggregated step history so far.
            if usage_limit_error is not None:
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                # Build a PipelineResult with any branch results we have so far
                try:
                    pr: PipelineResult[Any] = PipelineResult(
                        step_history=list(branch_results.values()),
                        total_cost_usd=sum(br.cost_usd for br in branch_results.values()),
                        total_tokens=sum(br.token_counts for br in branch_results.values()),
                        final_pipeline_context=context,
                    )
                except Exception:
                    pr = PipelineResult[Any](step_history=[], total_cost_usd=0.0, total_tokens=0)
                msg = usage_limit_error_msg or "Usage limit exceeded"
                raise UsageLimitExceededError(msg, pr)

            # Proactive limit check after each branch completes (dedented: always evaluated)
            if limits is not None:
                try:
                    from flujo.utils.formatting import format_cost as _fmt

                    breached_cost = getattr(
                        limits, "total_cost_usd_limit", None
                    ) is not None and total_cost > float(limits.total_cost_usd_limit)
                    breached_tokens = getattr(
                        limits, "total_tokens_limit", None
                    ) is not None and total_tokens > int(limits.total_tokens_limit)
                    if breached_cost or breached_tokens:
                        # Cancel remaining tasks promptly
                        for p in pending:
                            p.cancel()
                        if pending:
                            try:
                                await asyncio.gather(*pending, return_exceptions=True)
                            except Exception:
                                pass
                        pipeline_result: PipelineResult[Any] = PipelineResult(
                            step_history=list(branch_results.values()),
                            total_cost_usd=total_cost,
                            total_tokens=total_tokens,
                            final_pipeline_context=context,
                        )
                        if breached_cost:
                            msg = f"Cost limit of ${_fmt(float(limits.total_cost_usd_limit))} exceeded"
                        else:
                            msg = f"Token limit of {int(limits.total_tokens_limit)} exceeded"
                        raise UsageLimitExceededError(msg, pipeline_result)
                except UsageLimitExceededError:
                    raise
                except Exception:
                    # Do not disrupt normal execution on unexpected check errors
                    pass
        # FSD-009: Enforce limits deterministically at aggregation time (pure quota mode)
        if limits is not None:
            try:
                from flujo.utils.formatting import format_cost as _fmt

                if getattr(limits, "total_cost_usd_limit", None) is not None and total_cost > float(
                    limits.total_cost_usd_limit
                ):
                    pipeline_result = PipelineResult(
                        step_history=list(branch_results.values()),
                        total_cost_usd=total_cost,
                        total_tokens=total_tokens,
                        final_pipeline_context=context,
                    )
                    raise UsageLimitExceededError(
                        f"Cost limit of ${_fmt(float(limits.total_cost_usd_limit))} exceeded",
                        pipeline_result,
                    )
                if getattr(limits, "total_tokens_limit", None) is not None and total_tokens > int(
                    limits.total_tokens_limit
                ):
                    pipeline_result = PipelineResult(
                        step_history=list(branch_results.values()),
                        total_cost_usd=total_cost,
                        total_tokens=total_tokens,
                        final_pipeline_context=context,
                    )
                    raise UsageLimitExceededError(
                        f"Token limit of {int(limits.total_tokens_limit)} exceeded",
                        pipeline_result,
                    )
            except UsageLimitExceededError:
                raise
            except Exception:
                pass
        # Overall success
        if parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
            result.success = all_successful
        elif parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
            result.success = any(br.success for br in branch_results.values())
        else:
            result.success = all_successful
        # Build output
        output_dict: Dict[str, Any] = {}
        for bn, br in branch_results.items():
            output_dict[bn] = br.output if br.success else br
        result.output = output_dict
        # Apply declarative reduce mapper if present
        try:
            meta = getattr(parallel_step, "meta", {})
            reducer = meta.get("parallel_reduce_mapper") if isinstance(meta, dict) else None
            if callable(reducer):
                result.output = reducer(output_dict, context)
        except Exception:
            # On reducer error, keep original map for debuggability
            pass
        # Preserve input branch order deterministically
        result.metadata_["executed_branches"] = branch_names
        # Context merging using ContextManager
        if context is not None and parallel_step.merge_strategy != MergeStrategy.NO_MERGE:
            try:
                # Merge context updates from all branches (successful and failed)
                # This preserves context updates made before a step failed
                # Only consider branch contexts from successful branches when ignoring failures
                if parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
                    branch_ctxs = {
                        n: br.branch_context
                        for n, br in branch_results.items()
                        if br.success and br.branch_context is not None
                    }
                else:
                    branch_ctxs = {
                        n: br.branch_context
                        for n, br in branch_results.items()
                        if br.branch_context is not None
                    }

                if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                    # Helper: detect conflicts in simple fields between two contexts
                    def _detect_conflicts(target_ctx: Any, source_ctx: Any) -> None:
                        try:
                            # Prefer model_dump when available
                            if hasattr(source_ctx, "model_dump"):
                                src_fields = source_ctx.model_dump(exclude_none=True)
                            elif hasattr(source_ctx, "dict"):
                                src_fields = source_ctx.dict(exclude_none=True)
                            else:
                                src_fields = {
                                    k: v
                                    for k, v in getattr(source_ctx, "__dict__", {}).items()
                                    if not str(k).startswith("_")
                                }
                        except Exception:
                            src_fields = {}
                        for _fname, _sval in src_fields.items():
                            if str(_fname).startswith("_"):
                                continue
                            if hasattr(target_ctx, _fname):
                                _tval = getattr(target_ctx, _fname)
                                # Only consider non-container simple conflicts
                                if not isinstance(_tval, (dict, list)) and not isinstance(
                                    _sval, (dict, list)
                                ):
                                    if _tval is not None and _sval is not None:
                                        try:
                                            differs = _tval != _sval
                                        except Exception:
                                            differs = True
                                        if differs:
                                            from flujo.exceptions import (
                                                ConfigurationError as _CfgErr,
                                            )

                                            raise _CfgErr(
                                                f"Merge conflict for key '{_fname}'. Set an explicit merge strategy or field_mapping in your ParallelStep."
                                            )

                    for n, bc in branch_ctxs.items():
                        if parallel_step.field_mapping and n in parallel_step.field_mapping:
                            for f in parallel_step.field_mapping[n]:
                                if hasattr(bc, f):
                                    setattr(context, f, getattr(bc, f))
                        else:
                            # Enforce conflict detection before merging, with simple accumulator heuristic
                            _detect_conflicts(context, bc)
                            # Then perform safe merge via ContextManager to satisfy observability in tests
                            context = ContextManager.merge(context, bc)
                elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
                    if not hasattr(context, "scratchpad"):
                        setattr(context, "scratchpad", {})
                    for n in sorted(branch_ctxs):
                        bc = branch_ctxs[n]
                        if hasattr(bc, "scratchpad"):
                            for k in bc.scratchpad:
                                if k in context.scratchpad:
                                    telemetry.logfire.warning(
                                        f"Scratchpad key collision: '{k}', skipping"
                                    )
                                else:
                                    context.scratchpad[k] = bc.scratchpad[k]
                elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE and branch_ctxs:
                    last = sorted(branch_ctxs)[-1]
                    branch_ctx = branch_ctxs[last]
                    if parallel_step.context_include_keys:
                        for f in parallel_step.context_include_keys:
                            if hasattr(branch_ctx, f):
                                setattr(context, f, getattr(branch_ctx, f))
                    else:
                        if hasattr(context, "scratchpad"):
                            for bn in sorted(branch_ctxs):
                                bc = branch_ctxs[bn]
                                if hasattr(bc, "scratchpad"):
                                    for key, val in bc.scratchpad.items():
                                        context.scratchpad[key] = val
                elif parallel_step.merge_strategy == MergeStrategy.ERROR_ON_CONFLICT:
                    # Merge each branch strictly erroring on conflicts
                    from flujo.utils.context import safe_merge_context_updates as _merge

                    for n, bc in branch_ctxs.items():
                        _merge(context, bc, merge_strategy=MergeStrategy.ERROR_ON_CONFLICT)
                elif callable(parallel_step.merge_strategy):
                    parallel_step.merge_strategy(context, branch_ctxs)

                # Special handling for executed_branches field - merge it back to context
                if context is not None and hasattr(context, "executed_branches"):
                    # Get all executed branches from branch contexts
                    all_executed_branches = []
                    for bc in branch_ctxs.values():
                        if (
                            bc is not None
                            and hasattr(bc, "executed_branches")
                            and getattr(bc, "executed_branches", None)
                        ):
                            all_executed_branches.extend(getattr(bc, "executed_branches"))

                    # Handle executed_branches based on merge strategy
                    if parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
                        # For OVERWRITE, only keep the last successful branch
                        successful_branches = [
                            name for name, br in branch_results.items() if br.success
                        ]
                        if successful_branches:
                            # Get the last successful branch (alphabetically sorted)
                            last_successful_branch = sorted(successful_branches)[-1]
                            context.executed_branches = [last_successful_branch]

                            # Also handle branch_results for OVERWRITE strategy
                            if context is not None and hasattr(context, "branch_results"):
                                # Get the branch_results from the last successful branch context
                                last_branch_ctx = branch_ctxs.get(last_successful_branch)
                                if (
                                    last_branch_ctx is not None
                                    and hasattr(last_branch_ctx, "branch_results")
                                    and getattr(last_branch_ctx, "branch_results", None)
                                ):
                                    # Use branch context's results when available and non-empty
                                    context.branch_results = getattr(
                                        last_branch_ctx, "branch_results"
                                    ).copy()
                                else:
                                    # If no branch_results in context, create from current results
                                    context.branch_results = {
                                        last_successful_branch: branch_results[
                                            last_successful_branch
                                        ].output
                                    }
                        else:
                            context.executed_branches = []
                            if context is not None and hasattr(context, "branch_results"):
                                context.branch_results = {}
                    else:
                        # For other strategies, add all successful branches
                        successful_branches = [
                            name for name, br in branch_results.items() if br.success
                        ]
                        all_executed_branches.extend(successful_branches)

                        # Remove duplicates while preserving order
                        seen = set()
                        unique_branches = []
                        for branch in all_executed_branches:
                            if branch not in seen:
                                seen.add(branch)
                                unique_branches.append(branch)

                        # Update context with merged executed_branches
                        context.executed_branches = unique_branches

                        # Handle branch_results for other strategies
                        if context is not None and hasattr(context, "branch_results"):
                            # Merge branch_results from all successful branches
                            merged_branch_results = {}
                            for bc in branch_ctxs.values():
                                if (
                                    bc is not None
                                    and hasattr(bc, "branch_results")
                                    and getattr(bc, "branch_results", None)
                                ):
                                    merged_branch_results.update(getattr(bc, "branch_results"))
                            context.branch_results = merged_branch_results

                # Preserve the original branch context for downstream merges.
                # Only set if not already populated (e.g., from pipeline_result.final_pipeline_context).
                if result.branch_context is None:
                    result.branch_context = context
            except ConfigurationError as e:
                # Fail the entire parallel step with a clear error message
                result.success = False
                result.feedback = str(e)
                if result.branch_context is None:
                    result.branch_context = context
            except Exception as e:
                telemetry.logfire.error(f"Context merging failed: {e}")
        # Finalize result
        result.cost_usd = total_cost
        result.token_counts = total_tokens
        result.latency_s = time.monotonic() - start_time
        result.attempts = 1
        if result.success:
            result.feedback = (
                f"All {len(parallel_step.branches)} branches executed successfully"
                if all_successful
                else f"Parallel step completed with {len(failure_messages)} branch failures (ignored)"
            )
        else:
            # Enhanced detailed failure feedback aggregation
            # If feedback already set (e.g., ConfigurationError message), preserve it
            if not result.feedback:
                total_branches = len(parallel_step.branches)
                successful_branches_count = total_branches - len(failure_messages)

                # Format detailed failure information following Flujo best practices
                if len(failure_messages) == 1:
                    # Single failure - use direct message format for compatibility
                    result.feedback = failure_messages[0]
                else:
                    # Multiple failures - structured list with summary
                    summary = f"Parallel step failed: {len(failure_messages)} of {total_branches} branches failed"
                    if successful_branches_count > 0:
                        summary += f" ({successful_branches_count} succeeded)"
                    detailed_feedback = "; ".join(failure_messages)
                    result.feedback = f"{summary}. Failures: {detailed_feedback}"
        return to_outcome(result)


class ParallelStepExecutorOutcomes(Protocol):
    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        parallel_step: Optional[ParallelStep[Any]] = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepOutcome[StepResult]: ...


## Legacy adapter removed: DefaultParallelStepExecutorOutcomes (native outcomes supported)

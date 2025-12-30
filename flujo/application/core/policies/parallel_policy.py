from __future__ import annotations
import inspect
from flujo.type_definitions.common import JSONObject

from ._shared import (  # noqa: F401
    Awaitable,
    Callable,
    BranchFailureStrategy,
    ConfigurationError,
    ContextManager,
    Dict,
    Failure,
    InfiniteRedirectError,
    List,
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
from flujo.domain.base_model import BaseModel


# --- Parallel Step Executor policy ---
class ParallelStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultParallelStepExecutor(StepPolicy[ParallelStep[BaseModel]]):
    @property
    def handles_type(self) -> type[ParallelStep[BaseModel]]:
        return ParallelStep

    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[BaseModel],
    ) -> StepOutcome[StepResult]:
        step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        context_setter = getattr(frame, "context_setter", None)

        if not isinstance(step, ParallelStep):
            raise ValueError(f"Expected ParallelStep, got {type(step)}")
        parallel_step: ParallelStep[BaseModel] = step
        telemetry.logfire.debug(f"=== HANDLING PARALLEL STEP === {parallel_step.name}")
        telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")

        # Fail fast on removed merge strategy even when steps/pipelines were built via model_construct.
        from flujo.utils.scratchpad import is_merge_scratchpad

        if is_merge_scratchpad(getattr(parallel_step, "merge_strategy", None)):
            raise ConfigurationError(
                "merge_strategy=MERGE_SCRATCHPAD is removed. Use CONTEXT_UPDATE with "
                "explicit field_mapping or OVERWRITE/NO_MERGE."
            )

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
        branch_items: List[Tuple[str, object]] = list(parallel_step.branches.items())
        branch_names: List[str] = [bn for bn, _ in branch_items]
        branch_pipelines: List[object] = [bp for _, bp in branch_items]
        branch_quota_map: Dict[str, Optional[Quota]] = {bn: None for bn in branch_names}
        current_quota: Optional[Quota] = None
        try:
            if hasattr(core, "_get_current_quota"):
                current_quota = core._get_current_quota()
        except Exception:
            current_quota = None
        if current_quota is None:
            try:
                current_quota = getattr(frame, "quota", None)
            except Exception:
                current_quota = None
        if current_quota is None and limits is not None:
            try:
                from ..quota_manager import build_root_quota

                current_quota = build_root_quota(limits)
            except Exception:
                current_quota = None
        split_parent_quota: Optional[Quota] = None
        split_children: List[Quota] = []
        if current_quota is not None and len(branch_items) > 0:
            try:
                sub_quotas = current_quota.split(len(branch_items))
                for idx, bn in enumerate(branch_names):
                    branch_quota_map[bn] = sub_quotas[idx]
                split_parent_quota = current_quota
                split_children = list(sub_quotas)
            except Exception:
                # Fallback: no split if quota not available
                pass

        def _refund_split_quota() -> None:
            nonlocal split_parent_quota, split_children
            if split_parent_quota is None or not split_children:
                return
            try:
                from flujo.domain.models import UsageEstimate as _UsageEstimate

                for q in list(split_children):
                    try:
                        rem_cost, rem_tokens = q.get_remaining()
                        split_parent_quota.refund(
                            _UsageEstimate(cost_usd=float(rem_cost), tokens=int(rem_tokens))
                        )
                    except Exception:
                        continue
            finally:
                split_children = []

        # Tracking variables
        branch_results: Dict[str, StepResult] = {}
        branch_contexts: Dict[str, BaseModel | None] = {}
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

        def _merge_branch_context_into_parent(branch_ctx: BaseModel | None) -> None:
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
            branch_pipeline: object,
            branch_context: BaseModel | None,
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

                # Execute branch pipeline (or a single step) and normalize to a PipelineResult.
                if isinstance(branch_pipeline, Pipeline) and hasattr(
                    core, "_execute_pipeline_via_policies"
                ):
                    pipeline_result = await core._execute_pipeline_via_policies(
                        branch_pipeline,
                        data,
                        branch_context,
                        resources,
                        limits,
                        context_setter,
                    )
                else:
                    if not hasattr(core, "execute"):
                        raise RuntimeError("ExecutorCore is missing required 'execute' method")
                    step_outcome = await core.execute(
                        step=branch_pipeline,
                        data=data,
                        context=branch_context,
                        resources=resources,
                        limits=limits,
                        context_setter=context_setter,
                    )
                    if isinstance(step_outcome, StepResult):
                        sr = step_outcome
                        pipeline_result = PipelineResult(
                            step_history=[sr],
                            total_cost_usd=sr.cost_usd,
                            total_tokens=sr.token_counts,
                            final_pipeline_context=branch_context,
                        )
                    elif isinstance(step_outcome, Success):
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
                        _merge_branch_context_into_parent(branch_context)
                        raise PausedException(step_outcome.message)
                    else:
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

                # Enhanced feedback aggregation for branch failures.
                branch_feedback = ""
                if pipeline_result.step_history:
                    failed_steps = [s for s in pipeline_result.step_history if not s.success]
                    if failed_steps:
                        failure_details: list[str] = []
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
                    branch_context=branch_context,
                    metadata_={"exception_type": type(e).__name__},
                )
                return branch_name, failure
            finally:
                try:
                    if "quota_token" in locals() and quota_token is not None:
                        if hasattr(core, "_reset_current_quota"):
                            core._reset_current_quota(quota_token)
                        elif hasattr(core, "_quota_manager") and hasattr(quota_token, "old_value"):
                            core._quota_manager.set_current_quota(quota_token.old_value)
                except Exception:
                    pass

        async def _handle_branch_result(branch_execution_result: object, idx: int) -> None:
            nonlocal total_cost, total_tokens, all_successful
            branch_name_local = list(parallel_step.branches.keys())[idx]
            if isinstance(
                branch_execution_result,
                (
                    UsageLimitExceededError,
                    InfiniteRedirectError,
                    PricingNotConfiguredError,
                ),
            ):
                raise branch_execution_result
            if isinstance(branch_execution_result, Exception):
                raise branch_execution_result

            branch_result: StepResult
            if (
                isinstance(branch_execution_result, tuple)
                and len(branch_execution_result) == 2
                and isinstance(branch_execution_result[0], str)
                and isinstance(branch_execution_result[1], StepResult)
            ):
                branch_name_local = branch_execution_result[0]
                branch_result = branch_execution_result[1]
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

            branch_results[branch_name_local] = branch_result
            telemetry.logfire.debug(
                "Parallel branch result",
                extra={
                    "branch": branch_name_local,
                    "success": branch_result.success,
                    "cost_usd": branch_result.cost_usd,
                },
            )
            total_cost += float(getattr(branch_result, "cost_usd", 0.0) or 0.0)
            total_tokens += int(getattr(branch_result, "token_counts", 0) or 0)
            if not branch_result.success:
                all_successful = False
                failure_messages.append(
                    f"branch '{branch_name_local}' failed: {branch_result.feedback}"
                )

        start_index = 0

        # Execute remaining branches concurrently using the shared quota, and proactively cancel on breach
        pending: set[asyncio.Task[Tuple[str, StepResult]]] = set()
        task_to_branch: dict[asyncio.Task[Tuple[str, StepResult]], str] = {}
        for bn, bp in zip(branch_names[start_index:], branch_pipelines[start_index:]):
            t = asyncio.create_task(
                execute_branch(bn, bp, branch_contexts[bn], branch_quota_map.get(bn))
            )
            pending.add(t)
            task_to_branch[t] = bn

        # Consume tasks as they complete; cancel the rest if limits are breached
        completed_count = start_index
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
                    # Capture the failing branch's StepResult for aggregation when available.
                    try:
                        existing = getattr(ex, "result", None)
                        sr = None
                        if (
                            existing is not None
                            and hasattr(existing, "step_history")
                            and existing.step_history
                        ):
                            sr = existing.step_history[-1]
                        if isinstance(sr, StepResult):
                            branch_key = branch_hint or getattr(sr, "name", "unknown")
                            branch_results[str(branch_key)] = sr
                            total_cost += float(getattr(sr, "cost_usd", 0.0) or 0.0)
                            total_tokens += int(getattr(sr, "token_counts", 0) or 0)
                            if not sr.success:
                                all_successful = False
                                failure_messages.append(
                                    f"branch '{branch_key}' failed: {sr.feedback}"
                                )
                    except Exception:
                        pass
                    continue
                except Exception:
                    # On ANY other exception from a branch, cancel all remaining branches immediately
                    for p in pending:
                        p.cancel()
                    try:
                        if pending:
                            results = await asyncio.gather(*pending, return_exceptions=True)
                            for r in results:
                                if isinstance(r, Exception) and not isinstance(
                                    r, asyncio.CancelledError
                                ):
                                    telemetry.logfire.error(
                                        "Parallel branch task error during cancellation",
                                        extra={"error": str(r)},
                                    )
                    except Exception:
                        pass
                    _refund_split_quota()
                    raise
                await _handle_branch_result(res, completed_count)
                completed_count += 1

            if pause_message is not None:
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        results = await asyncio.gather(*pending, return_exceptions=True)
                        for r in results:
                            if isinstance(r, UsageLimitExceededError):
                                try:
                                    existing = getattr(r, "result", None)
                                    sr = None
                                    if (
                                        existing is not None
                                        and hasattr(existing, "step_history")
                                        and existing.step_history
                                    ):
                                        sr = existing.step_history[-1]
                                    if isinstance(sr, StepResult):
                                        branch_key = getattr(sr, "name", "unknown")
                                        branch_results[str(branch_key)] = sr
                                        total_cost += float(getattr(sr, "cost_usd", 0.0) or 0.0)
                                        total_tokens += int(getattr(sr, "token_counts", 0) or 0)
                                        if not sr.success:
                                            all_successful = False
                                            failure_messages.append(
                                                f"branch '{branch_key}' failed: {sr.feedback}"
                                            )
                                except Exception:
                                    pass
                                continue
                            if isinstance(r, Exception) and not isinstance(
                                r, asyncio.CancelledError
                            ):
                                telemetry.logfire.error(
                                    "Parallel branch task error during cancellation",
                                    extra={"error": str(r)},
                                )
                except Exception:
                    pass
                if pause_branch:
                    telemetry.logfire.info(
                        f"Parallel branch '{pause_branch}' paused: {pause_message}"
                    )
                _refund_split_quota()
                return Paused(message=pause_message or "Paused")

            if abort_signal is not None:
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        results = await asyncio.gather(*pending, return_exceptions=True)
                        for r in results:
                            if isinstance(r, Exception) and not isinstance(
                                r, asyncio.CancelledError
                            ):
                                telemetry.logfire.error(
                                    "Parallel branch task error during cancellation",
                                    extra={"error": str(r)},
                                )
                except Exception:
                    pass
                if abort_branch:
                    telemetry.logfire.info(
                        f"Parallel branch '{abort_branch}' triggered abort: {abort_signal}"
                    )
                _refund_split_quota()
                raise abort_signal

            # If a usage limit breach occurred in any completed branch, cancel the rest and
            # raise an error that includes the aggregated step history so far.
            if usage_limit_error is not None:
                # Give near-complete sibling tasks a brief grace window to finish so we can
                # aggregate their StepResults (improves determinism in CI and avoids missing
                # already-produced usage).
                try:
                    if pending:
                        more_done, pending = await asyncio.wait(pending, timeout=0.05)
                        for d in more_done:
                            branch_hint = task_to_branch.get(d)
                            try:
                                res = d.result()
                                await _handle_branch_result(res, completed_count)
                                completed_count += 1
                            except UsageLimitExceededError as ex:
                                usage_limit_error = usage_limit_error or ex
                                try:
                                    usage_limit_error_msg = usage_limit_error_msg or str(ex)
                                except Exception:
                                    pass
                                try:
                                    existing = getattr(ex, "result", None)
                                    sr = None
                                    if (
                                        existing is not None
                                        and hasattr(existing, "step_history")
                                        and existing.step_history
                                    ):
                                        sr = existing.step_history[-1]
                                    if isinstance(sr, StepResult):
                                        branch_key = branch_hint or getattr(sr, "name", "unknown")
                                        branch_results[str(branch_key)] = sr
                                        total_cost += float(getattr(sr, "cost_usd", 0.0) or 0.0)
                                        total_tokens += int(getattr(sr, "token_counts", 0) or 0)
                                        if not sr.success:
                                            all_successful = False
                                            failure_messages.append(
                                                f"branch '{branch_key}' failed: {sr.feedback}"
                                            )
                                except Exception:
                                    pass
                            except Exception:
                                # Ignore other branch errors here; cancellation will handle them.
                                continue
                except Exception:
                    pass
                for p in pending:
                    p.cancel()
                try:
                    if pending:
                        results = await asyncio.gather(*pending, return_exceptions=True)
                        for r in results:
                            if isinstance(r, Exception) and not isinstance(
                                r, asyncio.CancelledError
                            ):
                                telemetry.logfire.error(
                                    "Parallel branch task error during cancellation",
                                    extra={"error": str(r)},
                                )
                except Exception:
                    pass

                msg = usage_limit_error_msg or "Usage limit exceeded"
                try:
                    pr: PipelineResult[BaseModel] = PipelineResult(
                        step_history=list(branch_results.values()),
                        total_cost_usd=sum(br.cost_usd for br in branch_results.values()),
                        total_tokens=sum(br.token_counts for br in branch_results.values()),
                        final_pipeline_context=context,
                    )
                except Exception:
                    pr = PipelineResult[BaseModel](
                        step_history=[], total_cost_usd=0.0, total_tokens=0
                    )
                _refund_split_quota()
                raise UsageLimitExceededError(msg, pr)
        # Overall success
        if parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
            result.success = all_successful
        elif parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
            result.success = any(br.success for br in branch_results.values())
        else:
            result.success = all_successful
        # Build output
        output_dict: JSONObject = {}
        for bn, br in branch_results.items():
            output_dict[bn] = br.output if br.success else br
        result.output = output_dict
        # Apply reducer if present (StepResult-aware first, meta reducer as fallback)
        try:
            reducer = getattr(parallel_step, "reduce", None)
            reduced: object | None = None
            if callable(reducer):
                ordered_results = [
                    branch_results[name] for name in branch_names if name in branch_results
                ]
                try:
                    reduced = reducer(ordered_results, context)
                except TypeError:
                    try:
                        reduced = reducer(ordered_results)
                    except TypeError:
                        try:
                            reduced = reducer(output_dict, context)
                        except TypeError:
                            reduced = reducer(output_dict)
                if inspect.isawaitable(reduced):
                    reduced = await reduced
            else:
                meta = getattr(parallel_step, "meta", {})
                mapper = meta.get("parallel_reduce_mapper") if isinstance(meta, dict) else None
                if callable(mapper):
                    reduced = mapper(output_dict, context)
            if isinstance(reduced, StepResult):
                result.output = reduced.output
                result.success = reduced.success
                result.feedback = reduced.feedback
                try:
                    result.metadata_["reduced_from"] = {
                        "strategy": getattr(reducer, "__name__", "reduce"),
                        "selected_step": reduced.name,
                    }
                except Exception:
                    pass
            elif reduced is not None:
                result.output = reduced
        except Exception:
            # On reducer error, keep original map for debuggability
            pass
        # Preserve input branch order deterministically
        result.metadata_["executed_branches"] = branch_names
        # Context merging using ContextManager
        if context is not None and parallel_step.merge_strategy != MergeStrategy.NO_MERGE:
            try:
                # Merge context updates only from successful branches.
                # Failed branches must not poison the parent context.
                branch_ctxs = {
                    n: br.branch_context
                    for n, br in branch_results.items()
                    if br.success and br.branch_context is not None
                }

                def _last_declared_branch(candidates: set[str]) -> str | None:
                    for name in reversed(branch_names):
                        if name in candidates:
                            return name
                    return None

                if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                    # Helper: detect conflicts between TWO branch contexts (NOT parent vs branch)
                    def _detect_branch_conflicts(
                        ctx_a: object, ctx_b: object, name_a: str, name_b: str
                    ) -> None:
                        """Detect leaf conflicts between two branch contexts."""
                        from flujo.exceptions import ConfigurationError as _CfgErr

                        try:
                            if hasattr(ctx_a, "model_dump"):
                                fields_a = ctx_a.model_dump(exclude_none=True)
                            elif hasattr(ctx_a, "dict"):
                                fields_a = ctx_a.dict(exclude_none=True)
                            else:
                                fields_a = {
                                    k: v
                                    for k, v in getattr(ctx_a, "__dict__", {}).items()
                                    if not str(k).startswith("_")
                                }

                            if hasattr(ctx_b, "model_dump"):
                                fields_b = ctx_b.model_dump(exclude_none=True)
                            elif hasattr(ctx_b, "dict"):
                                fields_b = ctx_b.dict(exclude_none=True)
                            else:
                                fields_b = {
                                    k: v
                                    for k, v in getattr(ctx_b, "__dict__", {}).items()
                                    if not str(k).startswith("_")
                                }
                        except Exception:  # noqa: BLE001
                            return  # Can't compare, skip

                        def _walk(val_a: object, val_b: object, path: str) -> None:
                            # Skip branch_results/context_updates which are intentionally merged
                            if path in {"branch_results", "context_updates"}:
                                return
                            if isinstance(val_a, dict) and isinstance(val_b, dict):
                                for k in set(val_a.keys()) & set(val_b.keys()):
                                    if str(k).startswith("_"):
                                        continue
                                    _walk(val_a[k], val_b[k], f"{path}.{k}" if path else str(k))
                                return
                            if isinstance(val_a, (list, tuple, set)) or isinstance(
                                val_b, (list, tuple, set)
                            ):
                                return
                            if val_a is not None and val_b is not None:
                                # Skip numeric differences
                                if isinstance(val_a, (int, float)) and isinstance(
                                    val_b, (int, float)
                                ):
                                    return  # numeric counters intentionally not conflict-checked
                                try:
                                    differs = val_a != val_b
                                except Exception:  # noqa: BLE001
                                    differs = True
                                if differs:
                                    raise _CfgErr(
                                        f"Merge conflict for key '{path or '<root>'}'. Branches '{name_a}' and '{name_b}' set different values. "
                                        "Set an explicit merge strategy or field_mapping in your ParallelStep."
                                    )

                        # Compare common fields
                        for fname in set(fields_a.keys()) & set(fields_b.keys()):
                            if str(fname).startswith("_"):
                                continue
                            try:
                                _walk(fields_a[fname], fields_b[fname], str(fname))
                            except _CfgErr:
                                raise
                            except (AttributeError, TypeError, KeyError):
                                continue

                    # Phase 1: Detect conflicts between sibling branches (if multiple branches)
                    branch_names_list = list(branch_ctxs.keys())
                    if len(branch_names_list) > 1:
                        from flujo.exceptions import ConfigurationError as _CfgErr

                        for i, name_a in enumerate(branch_names_list):
                            for name_b in branch_names_list[i + 1 :]:
                                ctx_a = branch_ctxs[name_a]
                                ctx_b = branch_ctxs[name_b]
                                try:
                                    _detect_branch_conflicts(ctx_a, ctx_b, name_a, name_b)
                                except _CfgErr:
                                    raise

                    # Phase 2: Merge all branch contexts (branch values override parent)
                    for n, bc in branch_ctxs.items():
                        if parallel_step.field_mapping and n in parallel_step.field_mapping:
                            for f in parallel_step.field_mapping[n]:
                                if hasattr(bc, f):
                                    setattr(context, f, getattr(bc, f))
                        else:
                            # Merge branch_results/context_updates explicitly
                            for attr in ("branch_results", "context_updates"):
                                if hasattr(bc, attr) and hasattr(context, attr):
                                    try:
                                        tgt_attr = getattr(context, attr)
                                        src_attr = getattr(bc, attr)
                                        if isinstance(tgt_attr, dict) and isinstance(
                                            src_attr, dict
                                        ):
                                            tgt_attr.update(src_attr)
                                        elif isinstance(tgt_attr, list) and isinstance(
                                            src_attr, list
                                        ):
                                            # Extend lists, avoiding duplicates
                                            for item in src_attr:
                                                if item not in tgt_attr:
                                                    tgt_attr.append(item)
                                    except Exception:  # noqa: BLE001, S110
                                        pass
                            ContextManager.merge(context, bc)
                elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE and branch_ctxs:
                    last = _last_declared_branch(set(branch_ctxs.keys()))
                    if last is None:
                        branch_ctx = None
                    else:
                        branch_ctx = branch_ctxs[last]
                    if branch_ctx is not None:
                        if parallel_step.context_include_keys:
                            for f in parallel_step.context_include_keys:
                                if hasattr(branch_ctx, f):
                                    setattr(context, f, getattr(branch_ctx, f))
                        else:
                            from flujo.utils.context import safe_merge_context_updates as _merge

                            try:
                                _merge(context, branch_ctx, merge_strategy=MergeStrategy.OVERWRITE)
                            except Exception:
                                pass

                            # Merge step_outputs dict across branches for observability.
                            try:
                                if hasattr(context, "step_outputs") and isinstance(
                                    getattr(context, "step_outputs", None), dict
                                ):
                                    for bn in sorted(branch_ctxs):
                                        bc = branch_ctxs[bn]
                                        if bc is None:
                                            continue
                                        if hasattr(bc, "step_outputs") and isinstance(
                                            getattr(bc, "step_outputs", None), dict
                                        ):
                                            try:
                                                context.step_outputs.update(bc.step_outputs)
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                elif parallel_step.merge_strategy == MergeStrategy.ERROR_ON_CONFLICT:
                    # Merge each branch strictly erroring on conflicts
                    from flujo.utils.context import safe_merge_context_updates as _merge

                    for n, bc in branch_ctxs.items():
                        _merge(context, bc, merge_strategy=MergeStrategy.ERROR_ON_CONFLICT)
                elif parallel_step.merge_strategy == MergeStrategy.KEEP_FIRST:
                    from flujo.utils.context import safe_merge_context_updates as _merge

                    for name in branch_names:
                        bc = branch_ctxs.get(name)
                        if bc is None:
                            continue
                        _merge(context, bc, merge_strategy=MergeStrategy.KEEP_FIRST)
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
                            # Get the last successful branch in declared order
                            last_successful_branch = _last_declared_branch(set(successful_branches))
                            if last_successful_branch is None:
                                last_successful_branch = successful_branches[-1]
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
        if result.branch_context is None and context is not None:
            result.branch_context = context
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
        _refund_split_quota()
        return to_outcome(result)


class ParallelStepExecutorOutcomes(Protocol):
    async def execute(
        self,
        core: object,
        step: object,
        data: object,
        context: Optional[BaseModel],
        resources: Optional[object],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[BaseModel], Optional[BaseModel]], None]],
        parallel_step: Optional[ParallelStep[BaseModel]] = None,
    ) -> StepOutcome[StepResult]: ...


## Legacy adapter removed: DefaultParallelStepExecutorOutcomes (native outcomes supported)

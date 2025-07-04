from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Dict, Optional, TYPE_CHECKING, TypeVar

from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.parallel import ParallelStep
from ..domain.dsl.step import MergeStrategy, BranchFailureStrategy
from ..domain.models import BaseModel, PipelineResult, StepResult, UsageLimits
from ..exceptions import UsageLimitExceededError
from ..infra.telemetry import logfire
from ..domain.resources import AppResources

TContext = TypeVar("TContext", bound=BaseModel)

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .runner import StepExecutor

__all__ = ["_execute_parallel_step_logic"]


async def _execute_parallel_step_logic(
    parallel_step: ParallelStep[TContext],
    parallel_input: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: "StepExecutor[TContext]",
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Execute all branch pipelines concurrently and aggregate results."""

    result = StepResult(name=parallel_step.name)
    outputs: Dict[str, Any] = {}
    branch_results: Dict[str, StepResult] = {}

    # Shared state for proactive cancellation
    limit_breached = asyncio.Event()
    limit_breach_error: Optional[UsageLimitExceededError] = None

    async def run_branch(key: str, branch_pipe: Pipeline[Any, Any]) -> None:
        nonlocal limit_breach_error

        # Optimized context copying strategy
        if context is not None:
            if parallel_step.context_include_keys is not None:
                # Create a new context with only the specified fields
                branch_context_data = {}
                for field_key in parallel_step.context_include_keys:
                    if hasattr(context, field_key):
                        branch_context_data[field_key] = getattr(context, field_key)
                # Deepcopy only the selected data to maintain isolation
                ctx_copy = context.__class__(**copy.deepcopy(branch_context_data))
            else:
                # Fallback to the original safe behavior
                ctx_copy = copy.deepcopy(context)
        else:
            ctx_copy = None

        current = parallel_input
        branch_res = StepResult(name=f"{parallel_step.name}:{key}")

        try:
            for s in branch_pipe.steps:
                # Check if limit has been breached by another branch
                if limit_breached.is_set():
                    logfire.info(f"Branch '{key}' cancelled due to limit breach in sibling branch")
                    branch_res.success = False
                    branch_res.feedback = "Cancelled due to usage limit breach in sibling branch"
                    break

                sr = await step_executor(s, current, ctx_copy, resources)
                branch_res.latency_s += sr.latency_s
                branch_res.cost_usd += getattr(sr, "cost_usd", 0.0)
                branch_res.token_counts += getattr(sr, "token_counts", 0)
                branch_res.attempts += sr.attempts

                # Check usage limits after each step within the branch
                if usage_limits is not None:
                    # Calculate cumulative usage across all branches
                    total_cost = branch_res.cost_usd
                    total_tokens = branch_res.token_counts

                    # Add usage from other completed branches
                    for other_key, other_br in branch_results.items():
                        if other_key != key:
                            total_cost += other_br.cost_usd
                            total_tokens += other_br.token_counts

                    # Check if limits are breached
                    if (
                        usage_limits.total_cost_usd_limit is not None
                        and total_cost > usage_limits.total_cost_usd_limit
                    ):
                        limit_breach_error = UsageLimitExceededError(
                            f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded",
                            PipelineResult(step_history=[result], total_cost_usd=total_cost),
                        )
                        limit_breached.set()
                        break

                    if (
                        usage_limits.total_tokens_limit is not None
                        and total_tokens > usage_limits.total_tokens_limit
                    ):
                        limit_breach_error = UsageLimitExceededError(
                            f"Token limit of {usage_limits.total_tokens_limit} exceeded",
                            PipelineResult(step_history=[result], total_cost_usd=total_cost),
                        )
                        limit_breached.set()
                        break

                if not sr.success:
                    branch_res.success = False
                    branch_res.feedback = sr.feedback
                    branch_res.output = sr.output
                    break
                current = sr.output
            else:
                branch_res.success = True
                branch_res.output = current

        except Exception as e:
            logfire.error(f"Error in branch '{key}': {e}")
            branch_res.success = False
            branch_res.feedback = f"Branch execution error: {e}"
            branch_res.output = None

        branch_res.branch_context = ctx_copy

        outputs[key] = branch_res.output
        branch_results[key] = branch_res

    start = time.monotonic()

    # Create tasks for all branches
    branch_order = list(parallel_step.branches.keys())
    tasks = {
        asyncio.create_task(run_branch(k, pipe)): k for k, pipe in parallel_step.branches.items()
    }

    # Monitor tasks for completion or cancellation
    while tasks:
        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        # Check if any completed task was due to a limit breach
        if limit_breached.is_set():
            logfire.info("Usage limit breached, cancelling remaining tasks...")
            for task in pending:
                task.cancel()

            # Wait for cancellations to complete
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            # Re-raise the usage limit error
            if limit_breach_error is not None:
                from .runner import Flujo  # Local import to avoid circular

                Flujo._set_final_context(limit_breach_error.result, context)
                raise limit_breach_error
            break

        # Process completed tasks
        for task in done:
            try:
                await task  # This will re-raise any exception from the task
            except Exception as e:
                logfire.error(f"Task failed: {e}")
            tasks.pop(task)

    result.latency_s = time.monotonic() - start

    # Aggregate usage metrics
    for br in branch_results.values():
        result.cost_usd += br.cost_usd
        result.token_counts += br.token_counts

    succeeded_branches: Dict[str, StepResult] = {}
    failed_branches: Dict[str, StepResult] = {}
    for name, br in branch_results.items():
        if br.success:
            succeeded_branches[name] = br
        else:
            failed_branches[name] = br

    # --- 1. Handle Failures ---
    if failed_branches and parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
        result.success = False
        fail_name = next(iter(failed_branches))
        result.feedback = f"Branch '{fail_name}' failed. Propagating failure."
        result.output = {
            **{k: v.output for k, v in succeeded_branches.items()},
            **{k: v for k, v in failed_branches.items()},
        }
        result.attempts = 1
        return result

    # --- 2. Handle Merging for Successful Branches ---
    if parallel_step.merge_strategy != MergeStrategy.NO_MERGE and context is not None:
        if callable(parallel_step.merge_strategy):
            for res in succeeded_branches.values():
                if res.branch_context is not None:
                    parallel_step.merge_strategy(context, res.branch_context)
        elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
            if succeeded_branches:
                for name in branch_order:
                    if name in succeeded_branches:
                        branch_ctx = succeeded_branches[name].branch_context
                        if branch_ctx is not None:
                            merged = context.model_dump()
                            branch_data = branch_ctx.model_dump()
                            keys = parallel_step.context_include_keys or list(branch_data.keys())
                            for key in keys:
                                if key in branch_data:
                                    if (
                                        key == "scratchpad"
                                        and key in merged
                                        and isinstance(merged[key], dict)
                                        and isinstance(branch_data[key], dict)
                                    ):
                                        merged[key].update(branch_data[key])
                                    else:
                                        merged[key] = branch_data[key]
                            validated = context.__class__.model_validate(merged)
                            context.__dict__.update(validated.__dict__)
        elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
            if hasattr(context, "scratchpad"):
                for res in succeeded_branches.values():
                    bc = res.branch_context
                    if bc is not None and hasattr(bc, "scratchpad"):
                        context.scratchpad.update(bc.scratchpad)

    # --- 3. Finalize the Result ---
    result.success = bool(succeeded_branches)
    final_output = {k: v.output for k, v in succeeded_branches.items()}
    if parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
        final_output.update(failed_branches)

    result.output = final_output
    result.attempts = 1

    # Final usage limit check
    if usage_limits is not None:
        if (
            usage_limits.total_cost_usd_limit is not None
            and result.cost_usd > usage_limits.total_cost_usd_limit
        ):
            result.success = False
            result.feedback = f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded"
            pr_cost: PipelineResult[BaseModel] = PipelineResult(
                step_history=[result], total_cost_usd=result.cost_usd
            )
            from .runner import Flujo

            Flujo._set_final_context(pr_cost, context)
            raise UsageLimitExceededError(
                result.feedback,
                pr_cost,
            )
        if (
            usage_limits.total_tokens_limit is not None
            and result.token_counts > usage_limits.total_tokens_limit
        ):
            result.success = False
            result.feedback = f"Token limit of {usage_limits.total_tokens_limit} exceeded"
            pr_tokens: PipelineResult[BaseModel] = PipelineResult(
                step_history=[result], total_cost_usd=result.cost_usd
            )
            from .runner import Flujo

            Flujo._set_final_context(pr_tokens, context)
            raise UsageLimitExceededError(
                result.feedback,
                pr_tokens,
            )

    return result

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import time
from typing import Any, Dict, Optional, TypeVar, Callable, Awaitable, cast
from unittest.mock import Mock

from ...domain.dsl.pipeline import Pipeline
from ...domain.dsl.loop import LoopStep, MapStep
from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.dsl.step import (
    Step,
    MergeStrategy,
    BranchFailureStrategy,
    BranchKey,
    HumanInTheLoopStep,
)
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
    UsageLimits,
    PipelineContext,
)
from ...domain.plugins import PluginOutcome
from ...domain.validation import ValidationResult
from ...exceptions import (
    UsageLimitExceededError,
    MissingAgentError,
    InfiniteRedirectError,
    InfiniteFallbackError,
    PausedException,
    PricingNotConfiguredError,
)
from ...infra import telemetry
from ...domain.resources import AppResources
from flujo.steps.cache_step import CacheStep, _generate_cache_key
from ...signature_tools import SignatureAnalysis
from ..context_manager import (
    _accepts_param,
    _get_validation_flags,
    _apply_validation_metadata,
)
from ...utils.context import safe_merge_context_updates

TContext = TypeVar("TContext", bound=BaseModel)


def _deep_merge_dict(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep merge two dictionaries, preserving nested structures.

    Args:
        target: The target dictionary to merge into
        source: The source dictionary to merge from
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            _deep_merge_dict(target[key], value)
        else:
            # For non-dict values or new keys, simply assign
            target[key] = value


def _safe_merge_list(target: list[Any], source: list[Any]) -> None:
    """Safely merge two lists with type validation.

    Args:
        target: The target list to merge into
        source: The source list to merge from

    Raises:
        TypeError: If the lists contain incompatible types for merging
    """
    # Validate that both lists contain similar types for safe merging
    if not target or not source:
        # Empty lists are always safe to merge
        target.extend(source)
        return

    # Check if the lists contain similar types (all dicts, all strings, etc.)
    target_types = {type(item) for item in target}
    source_types = {type(item) for item in source}

    # If both lists have consistent types and they're compatible, merge safely
    if len(target_types) <= 1 and len(source_types) <= 1:
        # Check if the types are compatible for merging
        if not target_types or not source_types or target_types == source_types:
            # Both lists have consistent types, safe to merge
            target.extend(source)
        else:
            # Types are consistent but incompatible - skip merging
            # This prevents mixing int and str, for example
            return
    else:
        # Mixed types - use a more conservative approach
        # Only merge if the source list items are compatible with target types
        for item in source:
            if not target:
                # Empty target, add first item and update target_types
                target.append(item)
                target_types = {type(item)}
            elif type(item) in target_types:
                # Item is compatible with existing target types
                target.append(item)
            else:
                # Skip incompatible items to prevent type errors
                # Log a warning for debugging
                import logging

                logging.warning(
                    f"Skipped incompatible item of type {type(item)} during list merging."
                )
                continue


# Alias used across step logic helpers
StepExecutor = Callable[
    [Step[Any, Any], Any, Optional[TContext], Optional[AppResources], Optional[Any]],
    Awaitable[StepResult],
]

__all__ = [
    "StepExecutor",
    "_execute_parallel_step_logic",
    "_execute_loop_step_logic",
    "_execute_conditional_step_logic",
    "_execute_dynamic_router_step_logic",
    "_run_step_logic",
]


# Default context setter used when running step logic outside the Flujo runner
def _default_set_final_context(result: PipelineResult[TContext], ctx: Optional[TContext]) -> None:
    """Write ``ctx`` into ``result`` if present."""

    if ctx is not None:
        result.final_pipeline_context = ctx


def _should_pass_context(
    spec: SignatureAnalysis, context: Optional[TContext], func: Callable[..., Any]
) -> bool:
    """Determine if context should be passed to a function based on signature analysis.

    Args:
        spec: Signature analysis result from analyze_signature()
        context: The context object to potentially pass
        func: The function to analyze

    Returns:
        True if context should be passed to the function, False otherwise
    """
    # Check if function accepts context parameter (either explicitly or via **kwargs)
    # This is different from spec.needs_context which only checks if context is required
    accepts_context = _accepts_param(func, "context")
    return spec.needs_context or (context is not None and bool(accepts_context))


# Context variables for tracking fallback relationships and chains
_fallback_relationships_var: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "fallback_relationships", default={}
)
_fallback_chain_var: contextvars.ContextVar[list[Step[Any, Any]]] = contextvars.ContextVar(
    "fallback_chain", default=[]
)

# Cache for fallback relationship loop detection (True if loop detected, False otherwise)
_fallback_graph_cache: contextvars.ContextVar[Dict[str, bool]] = contextvars.ContextVar(
    "fallback_graph_cache", default={}
)

# Maximum length for fallback chains to prevent infinite loops
# This limit is chosen based on typical pipeline complexity in enterprise applications
# where fallback chains rarely exceed 10 steps. For healthcare/legal/finance applications,
# this provides a good balance between safety and flexibility.
_MAX_FALLBACK_CHAIN_LENGTH = 10

# Maximum iterations for fallback loop detection to prevent infinite loops
# This limit is chosen to handle complex fallback relationships while preventing
# performance issues in large pipelines.
_DEFAULT_MAX_FALLBACK_ITERATIONS = 100


def _manage_fallback_relationships(
    step: Step[Any, Any],
) -> Optional[contextvars.Token[Dict[str, str]]]:
    """Helper function to manage fallback relationship tracking.

    Args:
        step: The step with a fallback to track

    Returns:
        Token for resetting the context variable, or None if no fallback relationship
    """
    if not hasattr(step, "fallback_step") or step.fallback_step is None:
        # If fallback_step is None, no fallback relationship needs to be managed
        return None

    # Clear the graph cache when relationships change to prevent stale cache entries
    _fallback_graph_cache.set({})

    relationships = _fallback_relationships_var.get()
    relationships_token = _fallback_relationships_var.set(
        {**relationships, step.name: step.fallback_step.name}
    )
    return relationships_token


def _detect_fallback_loop(step: Step[Any, Any], chain: list[Step[Any, Any]]) -> bool:
    """Detect fallback loops using robust strategies for healthcare/legal/finance applications.

    Uses both local chain analysis and global relationship tracking to detect loops
    that could occur across the entire pipeline execution. Implements caching for
    improved performance in large pipelines.

    1. Object identity check (current implementation)
    2. Immediate name match (current step name matches last step in chain)
    3. Chain length limit (prevents extremely long chains)
    4. Global relationship loop detection with caching
    """
    # Strategy 1: Object identity check
    if step in chain:
        return True

    # Strategy 2: Immediate name match (current step name matches last step in chain)
    if chain and chain[-1].name == step.name:
        return True

    # Strategy 3: Chain length limit
    if len(chain) >= _MAX_FALLBACK_CHAIN_LENGTH:
        return True

    # Strategy 4: Global relationship loop detection with caching
    relationships = _fallback_relationships_var.get()
    if step.name in relationships:
        # Use cached graph for improved performance
        graph_cache = _fallback_graph_cache.get()

        # Create a robust cache key that includes the actual relationship content
        # This prevents cache collisions when different relationship sets have the same length
        # but different content (e.g., {'A': 'B', 'C': 'D'} vs {'A': 'C', 'C': 'A'})
        relationships_hash = hashlib.sha256(
            str(sorted(relationships.items())).encode("utf-8")
        ).hexdigest()  # Use SHA-256 for improved collision resistance
        cache_key = f"{step.name}_{len(relationships)}_{relationships_hash}"

        if cache_key not in graph_cache:
            # Build the graph for this step and cache it
            visited: set[str] = set()
            current_step = step.name
            next_step = relationships.get(current_step)

            # Add maximum iteration limit to prevent infinite loops
            max_iterations = _DEFAULT_MAX_FALLBACK_ITERATIONS
            iteration_count = 0

            while next_step and iteration_count < max_iterations:
                # If next_step is in visited, we've found a cycle
                if next_step in visited:
                    graph_cache[cache_key] = True
                    return True  # Loop detected
                visited.add(next_step)
                next_step = relationships.get(next_step)
                iteration_count += 1

            # Cache the result (no loop found)
            graph_cache[cache_key] = False
        else:
            # Use cached result
            cached_result = graph_cache[cache_key]
            return bool(cached_result)

    return False  # No loop detected or iteration limit reached


class ParallelUsageGovernor:
    """Helper to track and enforce usage limits atomically across parallel branches."""

    def __init__(self, usage_limits: Optional[UsageLimits]) -> None:
        self.usage_limits = usage_limits
        self.lock = asyncio.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.limit_breached = asyncio.Event()
        self.limit_breach_error: Optional[UsageLimitExceededError] = None

    def _create_breach_error(
        self, result: StepResult, limit_type: str, limit_value: Any, current_value: Any
    ) -> UsageLimitExceededError:
        """Create a UsageLimitExceededError with proper PipelineResult."""
        from flujo.exceptions import UsageLimitExceededError
        from flujo.domain.models import PipelineResult

        pipeline_result: PipelineResult[Any] = PipelineResult(
            step_history=[result],
            total_cost_usd=self.total_cost,
        )

        if limit_type == "cost":
            message = f"Cost limit of ${limit_value} exceeded. Current cost: ${current_value}"
        else:  # token
            message = f"Token limit of {limit_value} exceeded. Current tokens: {current_value}"

        return UsageLimitExceededError(message, result=pipeline_result)

    async def add_usage(self, cost_delta: float, token_delta: int, result: StepResult) -> bool:
        async with self.lock:
            self.total_cost += cost_delta
            self.total_tokens += token_delta
            if self.usage_limits is not None:
                if (
                    self.usage_limits.total_cost_usd_limit is not None
                    and self.total_cost > self.usage_limits.total_cost_usd_limit
                ):
                    self.limit_breach_error = self._create_breach_error(
                        result, "cost", self.usage_limits.total_cost_usd_limit, self.total_cost
                    )
                    self.limit_breached.set()
                elif (
                    self.usage_limits.total_tokens_limit is not None
                    and self.total_tokens > self.usage_limits.total_tokens_limit
                ):
                    self.limit_breach_error = self._create_breach_error(
                        result, "token", self.usage_limits.total_tokens_limit, self.total_tokens
                    )
                    self.limit_breached.set()
            return self.limit_breached.is_set()

    def breached(self) -> bool:
        return self.limit_breached.is_set()

    def get_error(self) -> Optional[UsageLimitExceededError]:
        return self.limit_breach_error


async def _execute_parallel_step_logic(
    parallel_step: ParallelStep[TContext],
    parallel_input: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    """Execute branch pipelines concurrently and merge their results.

    Uses a semaphore to bound concurrency and passes the breach event to agents for early cancellation.
    """

    result = StepResult(name=parallel_step.name)
    outputs: Dict[str, Any] = {}
    branch_results: Dict[str, StepResult] = {}
    branch_contexts: Dict[str, Optional[TContext]] = {}

    usage_governor = ParallelUsageGovernor(usage_limits)
    # Limit concurrency to prevent resource exhaustion while maintaining parallelism
    # This is a reasonable default that balances performance with resource usage
    DEFAULT_MAX_CONCURRENCY = 3
    max_concurrency = min(DEFAULT_MAX_CONCURRENCY, len(parallel_step.branches))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_branch(key: str, branch_pipe: Pipeline[Any, Any]) -> None:
        # Initialize branch result even if we don't execute
        branch_res = StepResult(name=f"{parallel_step.name}:{key}")
        branch_res.success = False
        branch_res.feedback = "Branch not executed due to usage limit breach"

        if usage_governor.breached():
            # Still populate dictionaries for cancelled branches
            outputs[key] = None
            branch_results[key] = branch_res
            branch_contexts[key] = None
            return

        async with semaphore:
            if usage_governor.breached():
                # Still populate dictionaries for cancelled branches
                outputs[key] = None
                branch_results[key] = branch_res
                branch_contexts[key] = None
                return
            if context is not None:
                if parallel_step.context_include_keys is not None:
                    branch_context_data = {}
                    for field_key in parallel_step.context_include_keys:
                        if hasattr(context, field_key):
                            branch_context_data[field_key] = getattr(context, field_key)
                    ctx_copy = context.__class__(**copy.deepcopy(branch_context_data))
                else:
                    ctx_copy = copy.deepcopy(context)
            else:
                ctx_copy = None

            current = parallel_input
            # Reset branch_res for actual execution
            branch_res.success = True
            branch_res.feedback = None

            try:
                for s in branch_pipe.steps:
                    if usage_governor.breached():
                        telemetry.logfire.info(
                            f"Branch '{key}' cancelled due to limit breach in sibling branch"
                        )
                        branch_res.success = False
                        branch_res.feedback = (
                            "Cancelled due to usage limit breach in sibling branch"
                        )
                        break

                    # Pass the breach event to the agent if it supports it
                    # (Test agents can check for this and exit early)
                    sr = await step_executor(
                        s, current, ctx_copy, resources, usage_governor.limit_breached
                    )
                    branch_res.latency_s += sr.latency_s
                    cost_delta = getattr(sr, "cost_usd", 0.0)
                    token_delta = getattr(sr, "token_counts", 0)
                    branch_res.cost_usd += cost_delta
                    branch_res.token_counts += token_delta
                    branch_res.attempts += sr.attempts

                    breached = await usage_governor.add_usage(cost_delta, token_delta, sr)
                    if breached:
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

            except UsageLimitExceededError:
                # Re-raise usage limit errors to propagate them up
                raise
            except Exception as e:
                telemetry.logfire.error(f"Error in branch '{key}': {e}")
                branch_res.success = False
                branch_res.feedback = f"Branch execution error: {e}"
                branch_res.output = None

            branch_res.branch_context = ctx_copy
            outputs[key] = branch_res.output
            branch_results[key] = branch_res
            branch_contexts[key] = ctx_copy

    start = time.monotonic()

    branch_order = list(parallel_step.branches.keys())
    tasks = {}
    for k, pipe in parallel_step.branches.items():
        if usage_governor.breached():
            break
        tasks[asyncio.create_task(run_branch(k, pipe))] = k

    while tasks:
        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        if usage_governor.breached():
            telemetry.logfire.info("Usage limit breached, cancelling remaining tasks...")
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            error = usage_governor.get_error()
            if error is not None:
                context_setter(error.result, context)
                raise error
            break

        for task in done:
            try:
                await task
            except UsageLimitExceededError:
                # Re-raise usage limit errors to propagate them up
                raise
            except Exception as e:
                telemetry.logfire.error(f"Task failed: {e}")
            tasks.pop(task)

    error = usage_governor.get_error()
    if error is not None:
        telemetry.logfire.error(f"Raising UsageLimitExceededError after all branches: {error}")
        context_setter(error.result, context)
        raise error

    result.latency_s = time.monotonic() - start

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

    # Context merging should happen BEFORE checking for failures
    if parallel_step.merge_strategy != MergeStrategy.NO_MERGE and context is not None:
        base_snapshot: Dict[str, Any] = {}
        seen_keys: set[str] = set()
        if parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
            if hasattr(context, "scratchpad"):
                base_snapshot = dict(getattr(context, "scratchpad") or {})
            else:
                raise ValueError(
                    "MERGE_SCRATCHPAD strategy requires context with 'scratchpad' attribute"
                )

        branch_iter = (
            sorted(succeeded_branches)
            if parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD
            else branch_order
        )

        # For CONTEXT_UPDATE strategy, we want to iterate over ALL branches (both succeeded and failed)
        if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
            branch_iter = branch_order

        merged: Dict[str, Any] | None = None
        if parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
            merged = context.model_dump()

        for branch_name in branch_iter:
            # For CONTEXT_UPDATE strategy, merge context from ALL branches (both succeeded and failed)
            # For other strategies, only merge from succeeded branches
            if (
                parallel_step.merge_strategy != MergeStrategy.CONTEXT_UPDATE
                and branch_name not in succeeded_branches
            ):
                continue
            branch_ctx = branch_contexts.get(branch_name)
            if branch_ctx is None:
                continue

            if callable(parallel_step.merge_strategy):
                parallel_step.merge_strategy(context, branch_ctx)
                continue

            if parallel_step.merge_strategy == MergeStrategy.OVERWRITE and merged is not None:
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
            elif parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                # New strategy: properly handle context updates without branch name conflicts
                branch_data = branch_ctx.model_dump()

                # If field_mapping is provided, use it to determine which fields to merge
                if parallel_step.field_mapping and branch_name in parallel_step.field_mapping:
                    fields_to_merge = parallel_step.field_mapping[branch_name]
                    for field in fields_to_merge:
                        if field in branch_data and hasattr(context, field):
                            setattr(context, field, branch_data[field])
                else:
                    # Default behavior: merge all fields that exist in the context
                    for key, value in branch_data.items():
                        if hasattr(context, key):
                            current_value = getattr(context, key)
                            # Enhanced merging with type validation and deep merge
                            if isinstance(current_value, dict) and isinstance(value, dict):
                                # Deep merge dictionaries to prevent data loss
                                _deep_merge_dict(current_value, value)
                            elif isinstance(current_value, list) and isinstance(value, list):
                                # Type-safe list merging with validation
                                _safe_merge_list(current_value, value)
                            else:
                                # For other types, overwrite (original behavior)
                                setattr(context, key, value)
            elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD and hasattr(
                branch_ctx, "scratchpad"
            ):
                branch_pc = cast(PipelineContext, branch_ctx)
                context_pc = cast(PipelineContext, context)
                if getattr(context_pc, "scratchpad", None) is None:
                    context_pc.scratchpad = {}
                for key, val in branch_pc.scratchpad.items():
                    if key in base_snapshot and base_snapshot[key] == val:
                        continue
                    if key in context_pc.scratchpad and context_pc.scratchpad[key] != val:
                        raise ValueError(
                            f"Scratchpad key collision for '{key}' in branch '{branch_name}'"
                        )
                    if key in seen_keys:
                        raise ValueError(
                            f"Scratchpad key collision for '{key}' in branch '{branch_name}'"
                        )
                    context_pc.scratchpad[key] = val
                    seen_keys.add(key)

        if parallel_step.merge_strategy == MergeStrategy.OVERWRITE and merged is not None:
            validated = context.__class__.model_validate(merged)
            context.__dict__.update(validated.__dict__)

    # Now check for failures and return early if needed
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

    result.success = bool(succeeded_branches)
    final_output = {k: v.output for k, v in succeeded_branches.items()}
    if parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
        final_output.update(failed_branches)

    result.output = final_output
    result.attempts = 1

    return result


async def _execute_loop_step_logic(
    loop_step: LoopStep[TContext],
    loop_step_initial_input: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    """Logic for executing a LoopStep without engine coupling."""
    # Each iteration operates on a deep copy of the context so any modifications
    # are isolated unless explicitly merged back by mappers.
    loop_overall_result = StepResult(name=loop_step.name)

    if loop_step.initial_input_to_loop_body_mapper:
        try:
            current_body_input = loop_step.initial_input_to_loop_body_mapper(
                loop_step_initial_input, context
            )
        except Exception as e:
            telemetry.logfire.error(
                f"Error in initial_input_to_loop_body_mapper for LoopStep '{loop_step.name}': {e}"
            )
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Initial input mapper raised an exception: {e}"
            return loop_overall_result
    else:
        current_body_input = loop_step_initial_input

    last_successful_iteration_body_output: Any = None
    final_body_output_of_last_iteration: Any = None
    loop_exited_successfully_by_condition = False

    for i in range(1, loop_step.max_loops + 1):
        loop_overall_result.attempts = i
        telemetry.logfire.info(
            f"LoopStep '{loop_step.name}': Starting Iteration {i}/{loop_step.max_loops}"
        )

        iteration_succeeded_fully = True
        current_iteration_data_for_body_step = current_body_input
        # Create deep copy of the current context (which may have been updated by previous iterations)
        # FIXED: Ensure we're copying from the updated context, not the original context
        iteration_context = copy.deepcopy(context) if context is not None else None

        with telemetry.logfire.span(f"Loop '{loop_step.name}' - Iteration {i}"):
            for body_s in loop_step.loop_body_pipeline.steps:
                try:
                    body_step_result_obj = await step_executor(
                        body_s,
                        current_iteration_data_for_body_step,
                        iteration_context,
                        resources,
                        None,  # breach_event
                    )
                except PausedException:
                    if context is not None and iteration_context is not None:
                        if hasattr(context, "__dict__") and hasattr(iteration_context, "__dict__"):
                            context.__dict__.update(iteration_context.__dict__)
                        elif hasattr(iteration_context, "__dict__"):
                            for key, value in iteration_context.__dict__.items():
                                try:
                                    setattr(context, key, value)
                                except Exception as e:
                                    telemetry.logfire.error(
                                        f"Failed to set attribute '{key}' on context during PausedException handling: {e}"
                                    )
                    # Re-raise PausedException to propagate it up the call stack
                    raise

                loop_overall_result.latency_s += body_step_result_obj.latency_s
                loop_overall_result.cost_usd += getattr(body_step_result_obj, "cost_usd", 0.0)
                loop_overall_result.token_counts += getattr(body_step_result_obj, "token_counts", 0)

                if usage_limits is not None:
                    if (
                        usage_limits is not None
                        and usage_limits.total_cost_usd_limit is not None
                        and loop_overall_result.cost_usd > usage_limits.total_cost_usd_limit
                    ):
                        telemetry.logfire.warn(
                            f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded"
                        )
                        loop_overall_result.success = False
                        loop_overall_result.feedback = (
                            f"Cost limit of ${usage_limits.total_cost_usd_limit} exceeded"
                        )
                        pr: PipelineResult[TContext] = PipelineResult(
                            step_history=[loop_overall_result],
                            total_cost_usd=loop_overall_result.cost_usd,
                        )
                        context_setter(pr, context)
                        raise UsageLimitExceededError(loop_overall_result.feedback, pr)
                    if (
                        usage_limits is not None
                        and usage_limits.total_tokens_limit is not None
                        and loop_overall_result.token_counts > usage_limits.total_tokens_limit
                    ):
                        telemetry.logfire.warn(
                            f"Token limit of {usage_limits.total_tokens_limit} exceeded"
                        )
                        loop_overall_result.success = False
                        loop_overall_result.feedback = (
                            f"Token limit of {usage_limits.total_tokens_limit} exceeded"
                        )
                        pr_tokens: PipelineResult[TContext] = PipelineResult(
                            step_history=[loop_overall_result],
                            total_cost_usd=loop_overall_result.cost_usd,
                        )
                        context_setter(pr_tokens, context)
                        raise UsageLimitExceededError(loop_overall_result.feedback, pr_tokens)

                if not body_step_result_obj.success:
                    telemetry.logfire.warn(
                        f"Body Step '{body_s.name}' in LoopStep '{loop_step.name}' (Iteration {i}) failed."
                    )
                    iteration_succeeded_fully = False
                    final_body_output_of_last_iteration = body_step_result_obj.output
                    break

                current_iteration_data_for_body_step = body_step_result_obj.output

            # ------------------------------------------------------------------
            # END of body pipeline for this iteration.  If the body executed
            # fully *without* failure, propagate the output so that:
            #   1. `exit_condition_callable` receives a meaningful `out` value
            #   2. Variables used later in the function (e.g. for final
            #      `loop_overall_result.output`) are correctly populated.
            # ------------------------------------------------------------------
            if iteration_succeeded_fully:
                final_body_output_of_last_iteration = current_iteration_data_for_body_step
                last_successful_iteration_body_output = current_iteration_data_for_body_step

            # Context merging for MapStep, RefineUntil, AgenticLoop: must happen after body step execution
            if isinstance(loop_step, MapStep):  # Replace hasattr with type check
                if context is not None and iteration_context is not None:
                    try:
                        merge_success = safe_merge_context_updates(
                            target_context=context,
                            source_context=iteration_context,
                            excluded_fields=set(),
                        )
                        if not merge_success:
                            telemetry.logfire.warn(
                                f"Context merge failed in MapStep '{loop_step.name}' iteration {i} "
                                f"(context fields: {list(context.__dict__.keys()) if context else 'None'}, "
                                f"iteration context fields: {list(iteration_context.__dict__.keys()) if iteration_context else 'None'}), "
                                "but continuing execution"
                            )
                    except Exception as e:
                        telemetry.logfire.error(
                            f"Failed to merge context updates in MapStep '{loop_step.name}' iteration {i}: {e}"
                        )
            elif (
                hasattr(loop_step, "loop_output_mapper")
                and loop_step.loop_output_mapper is not None
            ):
                if context is not None and iteration_context is not None:
                    try:
                        merge_success = safe_merge_context_updates(
                            target_context=context,
                            source_context=iteration_context,
                            excluded_fields=set(),
                        )
                        if not merge_success:
                            telemetry.logfire.warn(
                                f"Context merge failed in {loop_step.name} iteration {i} "
                                f"(context fields: {list(context.__dict__.keys()) if context else 'None'}, "
                                f"iteration context fields: {list(iteration_context.__dict__.keys()) if iteration_context else 'None'}), "
                                "but continuing execution"
                            )
                    except Exception as e:
                        telemetry.logfire.error(
                            f"Failed to merge context updates in {loop_step.name} iteration {i}: {e}"
                        )
            else:
                # FIXED: Add context merging for regular LoopStep with @step(updates_context=True)
                # This ensures context updates from loop body steps are properly applied
                if context is not None and iteration_context is not None:
                    telemetry.logfire.debug(
                        f"Attempting context merge for LoopStep '{loop_step.name}' iteration {i}"
                    )
                    try:
                        # First, apply any context updates from the step execution to the iteration context
                        # This ensures that the iteration context has the latest updates before merging
                        if isinstance(final_body_output_of_last_iteration, dict):
                            # If the output is a dict-like object, apply it to the iteration context
                            update_data = final_body_output_of_last_iteration
                            for key, value in update_data.items():
                                if hasattr(iteration_context, key):
                                    setattr(iteration_context, key, value)

                        merge_success = safe_merge_context_updates(
                            target_context=context,
                            source_context=iteration_context,
                            excluded_fields=set(),
                        )

                        if merge_success:
                            telemetry.logfire.debug(
                                f"Successfully merged context updates in LoopStep '{loop_step.name}' iteration {i}"
                            )

                        else:
                            telemetry.logfire.warn(
                                f"Context merge failed in LoopStep '{loop_step.name}' iteration {i} "
                                f"(context fields: {list(context.__dict__.keys()) if context else 'None'}, "
                                f"iteration context fields: {list(iteration_context.__dict__.keys()) if iteration_context else 'None'}), "
                                "but continuing execution"
                            )
                    except Exception as e:
                        telemetry.logfire.error(
                            f"Failed to merge context updates in LoopStep '{loop_step.name}' iteration {i}: {e}"
                        )
        # Regular LoopStep: ensure iterations remain isolated to prevent unintended
        # side effects between iterations. Isolation ensures that each iteration operates
        # independently, maintaining the integrity of the loop's logic and results.

        # --- FIRST PRINCIPLES GUARANTEE ---
        # After each iteration, always merge all updates from iteration_context into context.
        # This ensures that all loop types (LoopStep, MapStep, etc.) have their context updates preserved.
        # No code path (deepcopy, serialization, etc.) can reset or shadow updated fields between merge and exit condition.
        if context is not None and iteration_context is not None:
            try:
                merge_success = safe_merge_context_updates(
                    target_context=context,
                    source_context=iteration_context,
                    excluded_fields=set(),
                )
                if not merge_success:
                    telemetry.logfire.warn(
                        f"Universal context merge failed in LoopStep '{loop_step.name}' iteration {i} "
                        f"(context fields: {list(context.__dict__.keys()) if context else 'None'}, "
                        f"iteration context fields: {list(iteration_context.__dict__.keys()) if iteration_context else 'None'}), "
                        "but continuing execution"
                    )
            except Exception as e:
                telemetry.logfire.error(
                    f"Failed to perform universal context merge in LoopStep '{loop_step.name}' iteration {i}: {e}"
                )

        # Now check the exit condition on the main context after merging updates
        # The exit condition should evaluate based on the updated main context
        try:
            should_exit = loop_step.exit_condition_callable(
                final_body_output_of_last_iteration, context
            )
        except Exception as e:
            telemetry.logfire.error(
                f"Error in exit_condition_callable for LoopStep '{loop_step.name}': {e}"
            )
            loop_overall_result.success = False
            loop_overall_result.feedback = f"Exit condition callable raised an exception: {e}"
            break

        if should_exit:
            telemetry.logfire.info(
                f"LoopStep '{loop_step.name}' exit condition met at iteration {i}."
            )
            loop_overall_result.success = iteration_succeeded_fully
            if not iteration_succeeded_fully:
                loop_overall_result.feedback = (
                    "Loop exited by condition, but last iteration body failed."
                )
            loop_exited_successfully_by_condition = True
            break

        if i < loop_step.max_loops:
            if loop_step.iteration_input_mapper:
                try:
                    current_body_input = loop_step.iteration_input_mapper(
                        final_body_output_of_last_iteration, context, i
                    )
                except Exception as e:
                    telemetry.logfire.error(
                        f"Error in iteration_input_mapper for LoopStep '{loop_step.name}': {e}"
                    )
                    loop_overall_result.success = False
                    loop_overall_result.feedback = (
                        f"Iteration input mapper raised an exception: {e}"
                    )
                    break
            else:
                current_body_input = final_body_output_of_last_iteration
    else:
        telemetry.logfire.warn(
            f"LoopStep '{loop_step.name}' reached max_loops ({loop_step.max_loops}) without exit condition being met."
        )
        loop_overall_result.success = False
        loop_overall_result.feedback = (
            f"Reached max_loops ({loop_step.max_loops}) without meeting exit condition."
        )
        # Note: Context updates from iterations are NOT automatically merged back to the main context
        # This preserves loop iteration isolation. Context updates should be handled explicitly
        # through iteration_input_mapper or loop_output_mapper if needed.

    if loop_overall_result.success and loop_exited_successfully_by_condition:
        if loop_step.loop_output_mapper:
            try:
                loop_overall_result.output = loop_step.loop_output_mapper(
                    last_successful_iteration_body_output, context
                )
            except Exception as e:
                telemetry.logfire.error(
                    f"Error in loop_output_mapper for LoopStep '{loop_step.name}': {e}"
                )
                loop_overall_result.success = False
                loop_overall_result.feedback = f"Loop output mapper raised an exception: {e}"
                loop_overall_result.output = None
        else:
            loop_overall_result.output = last_successful_iteration_body_output
    else:
        loop_overall_result.output = final_body_output_of_last_iteration
        if not loop_overall_result.feedback:
            loop_overall_result.feedback = (
                "Loop did not complete successfully or exit condition not met positively."
            )

    return loop_overall_result


async def _execute_conditional_step_logic(
    conditional_step: ConditionalStep[TContext],
    conditional_step_input: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Logic for executing a ConditionalStep without engine coupling."""
    conditional_overall_result = StepResult(name=conditional_step.name)
    executed_branch_key: BranchKey | None = None
    branch_output: Any = None
    branch_succeeded = False

    try:
        branch_key_to_execute = conditional_step.condition_callable(conditional_step_input, context)
        telemetry.logfire.info(
            f"ConditionalStep '{conditional_step.name}': Condition evaluated to branch key '{branch_key_to_execute}'."
        )
        executed_branch_key = branch_key_to_execute

        selected_branch_pipeline = conditional_step.branches.get(branch_key_to_execute)
        if selected_branch_pipeline is None:
            selected_branch_pipeline = conditional_step.default_branch_pipeline
            if selected_branch_pipeline is None:
                err_msg = f"ConditionalStep '{conditional_step.name}': No branch found for key '{branch_key_to_execute}' and no default branch defined."
                telemetry.logfire.warn(err_msg)
                conditional_overall_result.success = False
                conditional_overall_result.feedback = err_msg
                return conditional_overall_result
            telemetry.logfire.info(
                f"ConditionalStep '{conditional_step.name}': Executing default branch."
            )
        else:
            telemetry.logfire.info(
                f"ConditionalStep '{conditional_step.name}': Executing branch for key '{branch_key_to_execute}'."
            )

        if conditional_step.branch_input_mapper:
            input_for_branch = conditional_step.branch_input_mapper(conditional_step_input, context)
        else:
            input_for_branch = conditional_step_input

        current_branch_data = input_for_branch
        branch_pipeline_failed_internally = False

        for branch_s in selected_branch_pipeline.steps:
            with telemetry.logfire.span(
                f"ConditionalStep '{conditional_step.name}' Branch '{branch_key_to_execute}' - Step '{branch_s.name}'"
            ) as span:
                if executed_branch_key is not None:
                    try:
                        span.set_attribute("executed_branch_key", str(executed_branch_key))
                    except Exception as e:
                        telemetry.logfire.error(f"Error setting span attribute: {e}")
                branch_step_result_obj = await step_executor(
                    branch_s,
                    current_branch_data,
                    context,
                    resources,
                    None,  # breach_event
                )

            conditional_overall_result.latency_s += branch_step_result_obj.latency_s
            conditional_overall_result.cost_usd += getattr(branch_step_result_obj, "cost_usd", 0.0)
            conditional_overall_result.token_counts += getattr(
                branch_step_result_obj, "token_counts", 0
            )

            if not branch_step_result_obj.success:
                telemetry.logfire.warn(
                    f"Step '{branch_s.name}' in branch '{branch_key_to_execute}' of ConditionalStep '{conditional_step.name}' failed."
                )
                branch_pipeline_failed_internally = True
                branch_output = branch_step_result_obj.output
                conditional_overall_result.feedback = f"Failure in branch '{branch_key_to_execute}', step '{branch_s.name}': {branch_step_result_obj.feedback}"
                break

            current_branch_data = branch_step_result_obj.output

        if not branch_pipeline_failed_internally:
            branch_output = current_branch_data
            branch_succeeded = True

    except Exception as e:
        telemetry.logfire.error(
            f"Error during ConditionalStep '{conditional_step.name}' execution: {e}",
            exc_info=True,
        )
        conditional_overall_result.success = False
        conditional_overall_result.feedback = f"Error executing conditional logic or branch: {e}"
        return conditional_overall_result

    conditional_overall_result.success = branch_succeeded
    if branch_succeeded:
        if conditional_step.branch_output_mapper:
            try:
                conditional_overall_result.output = conditional_step.branch_output_mapper(
                    branch_output, executed_branch_key, context
                )
            except Exception as e:
                telemetry.logfire.error(
                    f"Error in branch_output_mapper for ConditionalStep '{conditional_step.name}': {e}"
                )
                conditional_overall_result.success = False
                conditional_overall_result.feedback = (
                    f"Branch output mapper raised an exception: {e}"
                )
                conditional_overall_result.output = None
        else:
            conditional_overall_result.output = branch_output
    else:
        conditional_overall_result.output = branch_output

    conditional_overall_result.attempts = 1
    if executed_branch_key is not None:
        conditional_overall_result.metadata_ = conditional_overall_result.metadata_ or {}
        conditional_overall_result.metadata_["executed_branch_key"] = str(executed_branch_key)

    return conditional_overall_result


async def _execute_dynamic_router_step_logic(
    router_step: DynamicParallelRouterStep[TContext],
    router_input: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    """Run router agent then execute selected branches in parallel."""

    result = StepResult(name=router_step.name)
    try:
        from ...signature_tools import analyze_signature

        func = getattr(router_step.router_agent, "run", router_step.router_agent)
        spec = analyze_signature(func)

        router_kwargs: Dict[str, Any] = {}

        # Handle context parameter passing - follow the same pattern as regular step logic
        if spec.needs_context:
            if context is None:
                raise TypeError(
                    f"Router agent in step '{router_step.name}' requires a context, but no context model was provided to the Flujo runner."
                )
            router_kwargs["context"] = context
        elif _should_pass_context(spec, context, func):
            router_kwargs["context"] = context

        # Handle resources parameter passing
        if resources is not None:
            if _accepts_param(func, "resources"):
                router_kwargs["resources"] = resources

        raw = await func(router_input, **router_kwargs)
        branch_keys = getattr(raw, "output", raw)
    except Exception as e:  # pragma: no cover - defensive
        telemetry.logfire.error(f"Router agent error in '{router_step.name}': {e}")
        result.success = False
        result.feedback = f"Router agent error: {e}"
        return result

    if not isinstance(branch_keys, list):
        branch_keys = [branch_keys]

    selected: Dict[str, Step[Any, Any] | Pipeline[Any, Any]] = {
        k: v for k, v in router_step.branches.items() if k in branch_keys
    }
    if not selected:
        result.success = True
        result.output = {}
        result.attempts = 1
        result.metadata_ = {"executed_branches": []}
        return result

    config_kwargs = router_step.config.model_dump()

    parallel_step = Step.parallel(
        name=f"{router_step.name}_parallel",
        branches=selected,
        context_include_keys=router_step.context_include_keys,
        merge_strategy=router_step.merge_strategy,
        on_branch_failure=router_step.on_branch_failure,
        field_mapping=router_step.field_mapping,
        **config_kwargs,
    )

    parallel_result = await _execute_parallel_step_logic(
        parallel_step,
        router_input,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )

    parallel_result.name = router_step.name
    parallel_result.metadata_ = parallel_result.metadata_ or {}
    parallel_result.metadata_["executed_branches"] = list(selected.keys())
    return parallel_result


async def _handle_cache_step(
    step: CacheStep[Any, Any],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
) -> StepResult:
    key = _generate_cache_key(step.wrapped_step, data, context=context, resources=resources)
    cached: StepResult | None = None
    if key:
        try:
            cached = await step.cache_backend.get(key)
        except Exception as e:  # pragma: no cover - defensive
            telemetry.logfire.warn(f"Cache get failed for key {key}: {e}")
    if isinstance(cached, StepResult):
        cache_result = cached.model_copy(deep=True)
        cache_result.metadata_ = cache_result.metadata_ or {}
        cache_result.metadata_["cache_hit"] = True

        # CRITICAL FIX: Apply context updates even for cache hits
        if step.wrapped_step.updates_context and context is not None:
            try:
                # Apply the cached output to context as if the step had executed
                if isinstance(cache_result.output, dict):
                    for key, value in cache_result.output.items():
                        if hasattr(context, key):
                            setattr(context, key, value)
                elif hasattr(context, "result"):
                    # Fallback: store in generic result field
                    setattr(context, "result", cache_result.output)
            except Exception as e:
                telemetry.logfire.error(f"Failed to apply context updates from cache hit: {e}")

        return cache_result

    cache_result = await step_executor(
        step.wrapped_step, data, context, resources, None
    )  # breach_event
    if cache_result.success and key:
        try:
            await step.cache_backend.set(key, cache_result)
        except Exception as e:  # pragma: no cover - defensive
            telemetry.logfire.warn(f"Cache set failed for key {key}: {e}")
    return cache_result


async def _handle_loop_step(
    step: LoopStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    return await _execute_loop_step_logic(
        step,
        data,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )


async def _handle_conditional_step(
    step: ConditionalStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None,
) -> StepResult:
    return await _execute_conditional_step_logic(
        conditional_step=step,
        conditional_step_input=data,
        context=context,
        resources=resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
    )


async def _handle_dynamic_router_step(
    step: DynamicParallelRouterStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    return await _execute_dynamic_router_step_logic(
        step,
        data,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )


async def _handle_parallel_step(
    step: ParallelStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    return await _execute_parallel_step_logic(
        step,
        data,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )


async def _handle_hitl_step(
    step: HumanInTheLoopStep,
    data: Any,
    context: Optional[TContext],
) -> StepResult:
    message = step.message_for_user if step.message_for_user is not None else str(data)
    if isinstance(context, PipelineContext):
        context.scratchpad["status"] = "paused"
    raise PausedException(message)


async def _run_step_logic(
    step: Step[Any, Any],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
    context_setter: (Callable[[PipelineResult[TContext], Optional[TContext]], None] | None) = None,
    stream: bool = False,
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
) -> StepResult:
    """Core logic for executing a single step without engine coupling."""
    from ...infra import telemetry

    telemetry.logfire.info(f"Starting _run_step_logic for step: {step.name}")
    if context_setter is None:
        context_setter = _default_set_final_context

    visited_ids: set[int] = set()
    if step.agent is not None:
        visited_ids.add(id(step.agent))
    if isinstance(step, CacheStep):
        return await _handle_cache_step(step, data, context, resources, step_executor)
    if isinstance(step, LoopStep):
        return await _handle_loop_step(
            step,
            data,
            context,
            resources,
            step_executor,
            context_model_defined,
            usage_limits,
            context_setter,
        )
    if isinstance(step, ConditionalStep):
        return await _handle_conditional_step(
            step, data, context, resources, step_executor, context_model_defined, usage_limits
        )
    if isinstance(step, DynamicParallelRouterStep):
        return await _handle_dynamic_router_step(
            step,
            data,
            context,
            resources,
            step_executor,
            context_model_defined,
            usage_limits,
            context_setter,
        )
    if isinstance(step, ParallelStep):
        return await _handle_parallel_step(
            step,
            data,
            context,
            resources,
            step_executor,
            context_model_defined,
            usage_limits,
            context_setter,
        )
    if isinstance(step, HumanInTheLoopStep):
        return await _handle_hitl_step(step, data, context)

    result: StepResult = StepResult(name=step.name)
    original_agent = step.agent
    current_agent = original_agent
    last_feedback = None
    last_unpacked_output = None
    validation_failed = False
    accumulated_feedbacks: list[str] = []
    last_exception: Exception = Exception("Unknown error")
    for attempt in range(1, step.config.max_retries + 1):
        validation_failed = False
        result.attempts = attempt
        feedbacks: list[str] = []  # feedbacks for this attempt only
        plugin_failed_this_attempt = False  # Always initialize at start of attempt
        if current_agent is None:
            raise MissingAgentError(
                f"Step '{step.name}' is missing an agent. Assign one via `Step('name', agent=...)` "
                "or by using a step factory like `@step` or `Step.from_callable()`."
            )

        try:
            start = time.monotonic()
            agent_kwargs: Dict[str, Any] = {}
            # Apply prompt processors
            if step.processors.prompt_processors:
                telemetry.logfire.info(
                    f"Running {len(step.processors.prompt_processors)} prompt processors for step '{step.name}'..."
                )
                processed = data
                for proc in step.processors.prompt_processors:
                    try:
                        processed = await proc.process(processed, context)
                    except Exception as e:  # pragma: no cover - defensive
                        telemetry.logfire.error(f"Processor {proc.name} failed: {e}")
                    data = processed
            from ...signature_tools import analyze_signature

            # FR-35.1: Properly inspect the underlying agent's signature for AsyncAgentWrapper
            # The decision to inject context must be based on the ultimate target agent's signature
            target = getattr(current_agent, "_agent", current_agent)
            func = getattr(target, "_step_callable", None)
            if func is None:
                func = target.stream if stream and hasattr(target, "stream") else target.run
            func = cast(Callable[..., Any], func)
            spec = analyze_signature(func)

            # FR-35a & FR-35b: Use signature-aware context injection
            # First check if the agent requires context but none is provided
            if spec.needs_context and context is None:
                raise TypeError(
                    f"Component in step '{step.name}' requires a context, but no context model was provided to the Flujo runner."
                )
            # Then check if we should pass context based on signature analysis
            if _should_pass_context(spec, context, func):
                agent_kwargs["context"] = context

            if resources is not None:
                if _accepts_param(func, "resources"):
                    agent_kwargs["resources"] = resources
            if step.config.temperature is not None and _accepts_param(func, "temperature"):
                agent_kwargs["temperature"] = step.config.temperature
            stream_failed = False
            if stream and hasattr(current_agent, "stream"):
                chunks: list[Any] = []
                try:
                    async for chunk in current_agent.stream(data, **agent_kwargs):
                        if on_chunk is not None:
                            await on_chunk(chunk)
                        chunks.append(chunk)
                    result.latency_s += time.monotonic() - start
                    raw_output = chunks[0] if len(chunks) == 1 else chunks
                except Exception as e:
                    stream_failed = True
                    result.latency_s += time.monotonic() - start
                    partial = (
                        "".join(chunks)
                        if chunks and all(isinstance(c, str) for c in chunks)
                        else chunks
                    )
                    raw_output = partial
                    result.output = partial
                    result.feedback = str(e)
                    feedbacks.append(str(e))
                    last_feedback = str(e)
            else:
                raw_output = await current_agent.run(data, **agent_kwargs)
                result.latency_s += time.monotonic() - start

            # Extract usage metrics using shared helper function
            from ...cost import extract_usage_metrics

            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output, current_agent, step.name
            )

            # Update result with usage metrics
            result.token_counts = prompt_tokens + completion_tokens
            result.cost_usd = cost_usd

            # Check for Mock objects and usage limits
            if isinstance(raw_output, Mock):
                raise TypeError("Mock object as output")

            # Usage limits are checked by the UsageGovernor at the pipeline level
            # No need to check them here as it would interfere with cumulative tracking
        except (TypeError, UsageLimitExceededError, PricingNotConfiguredError) as e:
            # Re-raise critical exceptions immediately
            if isinstance(e, TypeError) and "Mock object as output" in str(e):
                raise
            if isinstance(e, UsageLimitExceededError):
                raise
            if isinstance(e, PricingNotConfiguredError):
                raise
            # For other TypeErrors, continue retrying
            last_exception = e
            continue
        except PausedException:
            # For PausedException, just raise after updating context; context already contains the log.
            raise
        except Exception as e:
            # Retry on all other exceptions
            last_exception = e
            continue

        unpacked_output = getattr(raw_output, "output", raw_output)
        # Apply output processors
        if step.processors.output_processors:
            telemetry.logfire.info(
                f"Running {len(step.processors.output_processors)} output processors for step '{step.name}'..."
            )
            processed = unpacked_output
            for proc in step.processors.output_processors:
                try:
                    processed = await proc.process(processed, context)
                except Exception as e:  # pragma: no cover - defensive
                    telemetry.logfire.error(f"Processor {proc.name} failed: {e}")
            unpacked_output = processed
        last_unpacked_output = unpacked_output

        success = not stream_failed
        redirect_to = None
        final_plugin_outcome: PluginOutcome | None = None
        is_validation_step, is_strict = _get_validation_flags(step)

        sorted_plugins = sorted(step.plugins, key=lambda p: p[1], reverse=True)
        for plugin, _ in sorted_plugins:
            try:
                from ...signature_tools import analyze_signature

                plugin_kwargs: Dict[str, Any] = {}
                func = cast(
                    Callable[..., Any],
                    getattr(plugin, "_plugin_callable", plugin.validate),
                )
                spec = analyze_signature(func)

                if spec.needs_context:
                    if context is None:
                        raise TypeError(
                            f"Plugin in step '{step.name}' requires a context, but no context model was provided to the Flujo runner."
                        )
                    plugin_kwargs["context"] = context

                if resources is not None:
                    if _accepts_param(func, "resources"):
                        plugin_kwargs["resources"] = resources
                validated = await asyncio.wait_for(
                    plugin.validate(
                        {"output": last_unpacked_output, "feedback": last_feedback},
                        **plugin_kwargs,
                    ),
                    timeout=step.config.timeout_s,
                )
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Plugin timeout in step {step.name}") from e

            if not validated.success:
                validation_failed = True
                plugin_failed_this_attempt = True
                if validated.feedback:
                    feedbacks.append(validated.feedback)
                redirect_to = validated.redirect_to
                final_plugin_outcome = validated
            if validated.new_solution is not None:
                final_plugin_outcome = validated

        if final_plugin_outcome and final_plugin_outcome.new_solution is not None:
            unpacked_output = final_plugin_outcome.new_solution
            last_unpacked_output = unpacked_output

        # Run programmatic validators regardless of plugin outcome
        if step.validators:
            telemetry.logfire.info(
                f"Running {len(step.validators)} programmatic validators for step '{step.name}'..."
            )
            validation_tasks = [
                validator.validate(unpacked_output, context=context)
                for validator in step.validators
            ]
            try:
                validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            except Exception as e:  # pragma: no cover - defensive
                validation_results = [e]

            failed_checks_feedback: list[str] = []
            collected_results: list[ValidationResult] = []
            for validator, res in zip(step.validators, validation_results):
                if isinstance(res, Exception):
                    vname = getattr(
                        validator,
                        "name",
                        getattr(validator, "__class__", type(validator)).__name__,
                    )
                    failed_checks_feedback.append(f"Validator '{vname}' crashed: {res}")
                    continue
                vres = cast(ValidationResult, res)
                collected_results.append(vres)
                if not vres.is_valid:
                    fb = vres.feedback or "No details provided."
                    failed_checks_feedback.append(f"Check '{vres.validator_name}' failed: {fb}")

            if step.persist_validation_results_to and context is not None:
                if hasattr(context, step.persist_validation_results_to):
                    history_list = getattr(context, step.persist_validation_results_to)
                    if isinstance(history_list, list):
                        history_list.extend(collected_results)

            if failed_checks_feedback:
                validation_failed = True
                feedbacks.extend(failed_checks_feedback)
                # For non-strict validation steps, don't fail the step when validation fails
                if is_strict or not is_validation_step:
                    success = False

        # --- RETRY LOGIC FIX ---
        if plugin_failed_this_attempt:
            success = False
        # --- END FIX ---
        # --- JOIN ALL FEEDBACKS ---
        feedback = "\n".join(feedbacks).strip() if feedbacks else None
        if feedback:
            accumulated_feedbacks.extend(feedbacks)
        # --- END JOIN ---
        if not success and attempt == step.config.max_retries:
            pass  # Could use last_unpacked_output here if needed
        if success:
            result.output = unpacked_output
            result.success = True
            result.feedback = feedback
            # Metrics are already calculated above in extract_usage_metrics
            _apply_validation_metadata(
                result,
                validation_failed=validation_failed,
                is_validation_step=is_validation_step,
                is_strict=is_strict,
            )
            return result
        elif validation_failed and not is_strict:
            # For non-strict validation steps, preserve output even when validation fails
            result.output = unpacked_output
            result.success = False
            result.feedback = feedback
            # Do not add metrics here - they will be added in the "no fallback" else block
            _apply_validation_metadata(
                result,
                validation_failed=validation_failed,
                is_validation_step=is_validation_step,
                is_strict=is_strict,
            )
            # Do not return here; allow fallback logic to execute after all retries

        for handler in step.failure_handlers:
            handler()

        if redirect_to:
            redirect_id = id(redirect_to)
            if redirect_id in visited_ids:
                raise InfiniteRedirectError(f"Redirect loop detected in step {step.name}")
            visited_ids.add(redirect_id)
            current_agent = redirect_to
        else:
            current_agent = original_agent

        if feedback:
            if isinstance(data, dict):
                data["feedback"] = data.get("feedback", "") + "\n" + feedback
            else:
                data = f"{str(data)}\n{feedback}"
        # Store all feedback so far for the next iteration
        last_feedback = "\n".join(accumulated_feedbacks).strip() if accumulated_feedbacks else None

    # If we get here, all retries failed
    result.success = False
    if accumulated_feedbacks:
        result.feedback = "\n".join(accumulated_feedbacks).strip()
    else:
        # FR-36: Populate feedback with the actual error type and message
        result.feedback = (
            f"Agent execution failed with {type(last_exception).__name__}: {last_exception}"
        )
    result.attempts = step.config.max_retries
    result.latency_s = 0.0

    # If the step failed and a fallback is defined, execute it.
    if not result.success and step.fallback_step:
        # Add primary step's metrics before fallback logic
        metrics_source = (
            raw_output
            if "raw_output" in locals() and raw_output is not None
            else last_unpacked_output
        )
        if metrics_source is not None:
            # Robust default: 1 token for string outputs, else use token_counts attribute or 0
            if hasattr(metrics_source, "token_counts"):
                result.token_counts += metrics_source.token_counts
            elif isinstance(metrics_source, str):
                result.token_counts += 1
            else:
                result.token_counts += 0
            result.cost_usd += getattr(metrics_source, "cost_usd", 0.0)

        # --- Fallback Metrics Policy ---
        # On fallback, only the fallback's cost is used in the final result (not summed with primary),
        # but token counts are summed (primary + fallback). This avoids double-counting costs and
        # matches user/test expectations. If you need full traceability, consider adding a metrics_history field.
        # This is the most robust and production-safe default.
        telemetry.logfire.info(
            f"Step '{step.name}' failed. Attempting fallback step '{step.fallback_step.name}'."
        )
        original_failure_feedback = result.feedback

        chain = _fallback_chain_var.get()

        # Use helper function to manage fallback relationships
        relationships_token = _manage_fallback_relationships(step)

        # Detect fallback loop after tracking the relationship globally
        # Use step.fallback_step for detection since we're checking if the fallback would create a loop
        if _detect_fallback_loop(step.fallback_step, chain):
            # Reset relationships before raising to prevent global state pollution
            if relationships_token is not None:
                _fallback_relationships_var.reset(relationships_token)
            raise InfiniteFallbackError(f"Fallback loop detected in step '{step.name}'")

        token = _fallback_chain_var.set(chain + [step])
        # Store primary token counts for summing, but do not add cost yet
        primary_token_counts = result.token_counts
        try:
            fallback_result = await step_executor(
                step.fallback_step,
                data,
                context,
                resources,
                None,  # breach_event
            )
            if not isinstance(fallback_result, StepResult):
                raise TypeError("step_executor did not return StepResult in fallback logic")
        finally:
            _fallback_chain_var.reset(token)
            # Reset relationships to prevent global state pollution across multiple pipeline runs
            if relationships_token is not None:
                _fallback_relationships_var.reset(relationships_token)

        result.latency_s += fallback_result.latency_s
        if fallback_result.success:
            result.success = True
            result.output = fallback_result.output
            result.feedback = None
            result.metadata_ = {
                **(result.metadata_ or {}),
                "fallback_triggered": True,
                "original_error": original_failure_feedback,
            }
            result.cost_usd = fallback_result.cost_usd
            result.token_counts = primary_token_counts + fallback_result.token_counts
            return result
        else:
            # On fallback failure, set cost to fallback's cost, and sum token counts
            result.cost_usd = 0.0  # Reset before setting fallback cost
            result.cost_usd = fallback_result.cost_usd
            result.token_counts = primary_token_counts + fallback_result.token_counts
            result.feedback = (
                f"Original error: {original_failure_feedback}\n"
                f"Fallback error: {fallback_result.feedback}"
            )
            return result
    else:
        # No fallback - metrics are already calculated above
        pass

    if not result.success and step.persist_feedback_to_context:
        if context is not None and hasattr(context, step.persist_feedback_to_context):
            history_list = getattr(context, step.persist_feedback_to_context)
            if isinstance(history_list, list) and result.feedback:
                history_list.append(result.feedback)

    return result


async def _run_step_logic_iterative(
    step: Step[Any, Any],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    *,
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    usage_limits: UsageLimits | None = None,
    context_setter: (Callable[[PipelineResult[TContext], Optional[TContext]], None] | None) = None,
    stream: bool = False,
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
    use_iterative_executor: bool = False,
) -> StepResult:
    """Core logic for executing a single step with optional iterative execution."""

    # Use ultra executor for optimal performance (replaces iterative executor)
    if use_iterative_executor:
        # Lazy import to avoid circular dependency
        from .ultra_executor import UltraStepExecutor

        executor: UltraStepExecutor[Any] = UltraStepExecutor()
        return await executor.execute_step(
            step=step,
            data=data,
            context=context,
            resources=resources,
            usage_limits=usage_limits,
            stream=stream,
            on_chunk=on_chunk,
        )

    # Fall back to original implementation
    return await _run_step_logic(
        step=step,
        data=data,
        context=context,
        resources=resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
        stream=stream,
        on_chunk=on_chunk,
    )

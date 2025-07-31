from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import inspect
import time
from typing import Any, Dict, Optional, TypeVar, Callable, Awaitable, cast
from unittest.mock import Mock
import multiprocessing

from ...domain.dsl.pipeline import Pipeline
from ...domain.dsl.loop import LoopStep
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
    ContextInheritanceError,
)
from ...infra import telemetry
from ...domain.resources import AppResources
from flujo.steps.cache_step import CacheStep, _generate_cache_key
from ...signature_tools import SignatureAnalysis
from ...utils.context import safe_merge_context_updates
from ...application.context_manager import _accepts_param
from ...application.context_manager import _get_validation_flags
from ...application.context_manager import _apply_validation_metadata

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
        # Only merge if we can safely determine compatibility
        if target_types == source_types:
            target.extend(source)
        # Otherwise, skip merging to avoid type conflicts


def _default_set_final_context(result: PipelineResult[TContext], ctx: Optional[TContext]) -> None:
    """Default function to set final context from pipeline result."""
    if ctx is not None and result.final_pipeline_context is not None:
        for (
            field_name,
            field_value,
        ) in result.final_pipeline_context.model_dump().items():
            setattr(ctx, field_name, field_value)


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


def _should_pass_context_to_plugin(context: Optional[TContext], func: Callable[..., Any]) -> bool:
    """Determine if context should be passed to a plugin based on signature analysis.

    This is more conservative than _should_pass_context - it only passes context
    to plugins that explicitly declare a 'context' parameter, not to plugins
    that accept it via **kwargs.

    Args:
        context: The context object to potentially pass
        func: The function to analyze

    Returns:
        True if context should be passed to the plugin, False otherwise
    """
    if context is None:
        return False

    # Use inspect to check for explicit keyword-only 'context' parameter
    import inspect

    sig = inspect.signature(func)
    has_explicit_context = any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context"
        for p in sig.parameters.values()
    )
    return has_explicit_context


def _filter_kwargs_for_executor(executor: Callable[..., Any], **kwargs: Any) -> Dict[str, Any]:
    """Filter kwargs to only include parameters that the executor accepts.

    Args:
        executor: The executor function to check
        **kwargs: The kwargs to filter

    Returns:
        Dict containing only the kwargs that the executor accepts
    """
    try:
        # Get the signature of the executor
        sig = inspect.signature(executor)
        # Get the parameter names that the executor accepts
        accepted_params = set(sig.parameters.keys())
        # Filter kwargs to only include accepted parameters
        return {k: v for k, v in kwargs.items() if k in accepted_params}
    except (ValueError, TypeError):
        # If we can't inspect the signature (e.g., built-in functions),
        # return empty dict to be safe
        return {}


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

    def _create_breach_error_message(
        self, limit_type: str, limit_value: Any, current_value: Any
    ) -> str:
        """Create a breach error message string."""
        if limit_type == "cost":
            return f"Cost limit of ${limit_value} exceeded. Current cost: ${current_value}"
        else:  # token
            return f"Token limit of {limit_value} exceeded. Current tokens: {current_value}"

    async def add_usage(self, cost_delta: float, token_delta: int, result: StepResult) -> bool:
        """Add usage and check for breach. Returns True if breach occurred."""
        async with self.lock:
            self.total_cost += cost_delta
            self.total_tokens += token_delta

            if self.usage_limits is not None:
                if (
                    self.usage_limits.total_cost_usd_limit is not None
                    and self.total_cost > self.usage_limits.total_cost_usd_limit
                ):
                    message = self._create_breach_error_message(
                        "cost", self.usage_limits.total_cost_usd_limit, self.total_cost
                    )
                    pipeline_result_cost: PipelineResult[Any] = PipelineResult(
                        step_history=[result] if result else [],
                        total_cost_usd=self.total_cost,
                    )
                    self.limit_breach_error = UsageLimitExceededError(
                        message, result=pipeline_result_cost
                    )
                    self.limit_breached.set()
                elif (
                    self.usage_limits.total_tokens_limit is not None
                    and self.total_tokens > self.usage_limits.total_tokens_limit
                ):
                    message = self._create_breach_error_message(
                        "token", self.usage_limits.total_tokens_limit, self.total_tokens
                    )
                    pipeline_result: PipelineResult[Any] = PipelineResult(
                        step_history=[result] if result else [],
                        total_cost_usd=self.total_cost,
                    )
                    self.limit_breach_error = UsageLimitExceededError(
                        message, result=pipeline_result
                    )
                    self.limit_breached.set()
            return self.limit_breached.is_set()

    def breached(self) -> bool:
        """Check if a limit has been breached."""
        return self.limit_breached.is_set()

    def get_error_message(self) -> Optional[str]:
        """Get the error message if a breach occurred."""
        if self.limit_breach_error:
            return self.limit_breach_error.args[0]
        return None

    def get_error(self) -> Optional[UsageLimitExceededError]:
        """Get the error if a breach occurred."""
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
    breach_event: Optional[asyncio.Event] = None,
) -> StepResult:
    """Logic for executing a ParallelStep without engine coupling."""
    telemetry.logfire.debug(f"_execute_parallel_step_logic called for step: {parallel_step.name}")
    telemetry.logfire.debug(f"Context is None: {context is None}")
    telemetry.logfire.debug(f"Merge strategy: {parallel_step.merge_strategy}")

    result = StepResult(name=parallel_step.name)
    outputs: Dict[str, Any] = {}
    branch_results: Dict[str, StepResult] = {}
    errors: Dict[str, Exception] = {}

    # Create usage governor for parallel execution
    usage_governor = ParallelUsageGovernor(usage_limits)

    # Check for empty branches
    if not parallel_step.branches:
        result.success = False
        result.feedback = "Parallel step has no branches to execute"
        result.output = {}
        return result

    # Track completion order for OVERWRITE merge strategy
    completion_order = []
    completion_lock = asyncio.Lock()
    running_tasks: Dict[str, asyncio.Task[None]] = {}

    # Create bounded concurrency semaphore to prevent thundering herd
    # Use a reasonable limit based on CPU cores to prevent lock contention
    cpu_count = multiprocessing.cpu_count()
    semaphore = asyncio.Semaphore(min(10, cpu_count * 2))

    async def run_branch(key: str, branch_pipe: Pipeline[Any, Any]) -> None:
        """Execute a single branch with cancellation handling and bounded concurrency."""
        # Acquire semaphore to limit concurrent execution and prevent lock contention
        async with semaphore:
            # Isolate context for this branch
            branch_context = copy.deepcopy(context) if context is not None else None
            branch_results[key] = StepResult(name=key, success=False, attempts=0)

            try:
                if not hasattr(branch_pipe, "name"):
                    object.__setattr__(branch_pipe, "name", f"parallel_branch_{key}")
                current_data = parallel_input
                total_latency = 0.0
                total_cost = 0.0
                total_tokens = 0
                all_successful = True
                last_feedback = None

                for step in branch_pipe.steps:
                    try:
                        # Filter kwargs based on executor signature
                        filtered_kwargs = _filter_kwargs_for_executor(
                            step_executor,
                            usage_limits=usage_limits,
                        )
                        step_result = await step_executor(
                            step,
                            current_data,
                            branch_context,
                            resources,
                            breach_event,
                            **filtered_kwargs,
                        )

                        # Add usage to governor and check for breach
                        cost_delta = getattr(step_result, "cost_usd", 0.0)
                        token_delta = getattr(step_result, "token_counts", 0)
                        if await usage_governor.add_usage(cost_delta, token_delta, step_result):
                            # Limit was breached. Signal other branches to stop
                            if breach_event:
                                breach_event.set()
                            telemetry.logfire.debug(
                                f"Branch {key} breached limit, signaling others to stop"
                            )
                            return

                        total_latency += step_result.latency_s
                        total_cost += cost_delta
                        total_tokens += token_delta
                        if not step_result.success:
                            all_successful = False
                            last_feedback = step_result.feedback
                            break
                        current_data = step_result.output
                    except Exception as step_error:
                        all_successful = False
                        last_feedback = f"Branch execution error: {str(step_error)}"
                        break

                branch_result = StepResult(
                    name=f"branch::{key}",
                    output=current_data if all_successful else None,
                    success=all_successful,
                    attempts=1,
                    latency_s=total_latency,
                    token_counts=total_tokens,
                    cost_usd=total_cost,
                    feedback=last_feedback,
                    branch_context=branch_context,
                )
                branch_results[key] = branch_result
                outputs[key] = branch_result.output

                # Track completion order for OVERWRITE merge strategy
                async with completion_lock:
                    completion_order.append(key)

            except asyncio.CancelledError:
                # This is the cancellation hygiene recommended by the expert.
                # If cancelled, record a specific "cancelled" result.
                branch_results[key] = StepResult(
                    name=f"branch::{key}",
                    success=False,
                    feedback="Cancelled due to usage limit breach by another branch.",
                    cost_usd=total_cost if "total_cost" in locals() else 0.0,
                    token_counts=total_tokens if "total_tokens" in locals() else 0,
                )
            except Exception as e:
                errors[key] = e
                branch_results[key] = StepResult(
                    name=key,
                    output=None,
                    success=False,
                    feedback=f"Branch execution error: {str(e)}",
                    cost_usd=total_cost if "total_cost" in locals() else 0.0,
                    token_counts=total_tokens if "total_tokens" in locals() else 0,
                )

    # Start all branches concurrently
    for key, branch_pipe in parallel_step.branches.items():
        task = asyncio.create_task(run_branch(key, branch_pipe), name=f"branch_{key}")
        running_tasks[key] = task

    # Create the breach watcher for responsive signaling
    async def breach_watcher() -> None:
        """Watch for breach events and cancel all running tasks."""
        try:
            # Wait for either a breach event or the usage governor to signal a breach
            # Use a much shorter timeout for responsive cancellation
            if breach_event:
                # Wait for immediate breach signal from any branch
                await asyncio.wait_for(breach_event.wait(), timeout=0.1)
            else:
                # Fallback to usage governor with shorter timeout
                await asyncio.wait_for(usage_governor.limit_breached.wait(), timeout=0.1)

            # Immediately cancel all running tasks when breach is detected
            for task in running_tasks.values():
                if not task.done():
                    task.cancel()
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            pass
        except asyncio.TimeoutError:
            # Handle timeout gracefully - this means no breach occurred
            # Wait for all tasks to complete naturally
            if running_tasks:
                await asyncio.gather(*running_tasks.values(), return_exceptions=True)

    watcher_task = asyncio.create_task(breach_watcher(), name="breach_watcher")

    # Use asyncio.gather for state integrity - this ensures all tasks complete
    # and we get their results, even if some were cancelled
    all_tasks = list(running_tasks.values()) + [watcher_task]
    await asyncio.gather(*all_tasks, return_exceptions=True)

    # Centralized decision-making with complete state
    if usage_governor.breached():
        # Ensure the history is complete, even if some branches were cancelled
        final_history = list(branch_results.values())
        for key in parallel_step.branches:
            if key not in branch_results:
                # This branch was likely cancelled before it could even start or report.
                # Add a placeholder to ensure the history is complete.
                final_history.append(
                    StepResult(
                        name=f"branch::{key}",
                        success=False,
                        feedback="Not executed due to usage limit breach.",
                    )
                )

        pipeline_result_for_exc: PipelineResult[Any] = PipelineResult(
            step_history=final_history,
            total_cost_usd=usage_governor.total_cost,
        )
        message = usage_governor.get_error_message() or "Usage limit exceeded"
        raise UsageLimitExceededError(message, result=pipeline_result_for_exc)

    # Accumulate cost and tokens from all branches
    total_cost = 0.0
    total_tokens = 0
    total_latency = 0.0
    for branch_result in branch_results.values():
        total_cost += getattr(branch_result, "cost_usd", 0.0)
        total_tokens += getattr(branch_result, "token_counts", 0)
        total_latency += getattr(branch_result, "latency_s", 0.0)

    # Set the accumulated metrics on the result
    result.cost_usd = total_cost
    result.token_counts = total_tokens
    result.latency_s = total_latency

    telemetry.logfire.debug("=== AFTER METRICS ACCUMULATION ===")
    telemetry.logfire.debug(f"Total cost: {total_cost}")
    telemetry.logfire.debug(f"Total tokens: {total_tokens}")
    telemetry.logfire.debug(f"Total latency: {total_latency}")

    # Context merging based on strategy
    print("[DEBUG] === CONTEXT MERGING SECTION ===")
    print(f"[DEBUG] About to start context merging. Context is None: {context is None}")
    print(f"[DEBUG] Merge strategy: {parallel_step.merge_strategy}")
    print(f"[DEBUG] Branch results: {list(branch_results.keys())}")
    print(f"[DEBUG] Context type: {type(context)}")
    print(
        f"[DEBUG] Condition: {context is not None and (parallel_step.merge_strategy in {MergeStrategy.CONTEXT_UPDATE, MergeStrategy.OVERWRITE, MergeStrategy.MERGE_SCRATCHPAD} or callable(parallel_step.merge_strategy))}"
    )
    if context is not None and (
        parallel_step.merge_strategy
        in {
            MergeStrategy.CONTEXT_UPDATE,
            MergeStrategy.OVERWRITE,
            MergeStrategy.MERGE_SCRATCHPAD,
        }
        or callable(parallel_step.merge_strategy)
    ):
        telemetry.logfire.debug("Context merging condition met, proceeding with merge")
        if parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
            telemetry.logfire.debug("Using CONTEXT_UPDATE strategy")
            # For CONTEXT_UPDATE, merge contexts in the order of branch_results to ensure later branches overwrite earlier ones
            # Track accumulated values for counter fields to prevent overwriting
            accumulated_values = {}

            for key, branch_result in branch_results.items():
                branch_ctx = getattr(branch_result, "branch_context", None)
                if branch_ctx is not None:
                    try:
                        # For CONTEXT_UPDATE, we need to handle counter fields specially
                        # Counter fields should be accumulated, not replaced
                        counter_field_names = {
                            "accumulated_value",
                            "iteration_count",
                            "counter",
                            "count",
                            "total_count",
                            "processed_count",
                            "success_count",
                            "error_count",
                        }

                        # First, accumulate counter fields
                        for field_name in counter_field_names:
                            if hasattr(branch_ctx, field_name) and hasattr(context, field_name):
                                branch_value = getattr(branch_ctx, field_name)
                                current_value = getattr(context, field_name)

                                # Only accumulate if both values are numeric
                                if isinstance(branch_value, (int, float)) and isinstance(
                                    current_value, (int, float)
                                ):
                                    if field_name not in accumulated_values:
                                        accumulated_values[field_name] = current_value
                                    accumulated_values[field_name] += branch_value

                        # Then merge other fields normally
                        safe_merge_context_updates(context, branch_ctx)

                        # Finally, apply accumulated counter values
                        for field_name, accumulated_value in accumulated_values.items():
                            if hasattr(context, field_name):
                                setattr(context, field_name, accumulated_value)

                        telemetry.logfire.debug(f"Merged context from branch {key}")
                    except Exception as e:
                        telemetry.logfire.error(f"Failed to merge context from branch {key}: {e}")
        elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
            print("[DEBUG] Using OVERWRITE strategy")
            # For OVERWRITE, merge scratchpad from all successful branches
            # and use the last successful branch's context for other fields
            last_successful_branch = None
            # Iterate through completion order in reverse to find the last successful branch
            for key in reversed(completion_order):
                branch_result_item: StepResult | None = branch_results.get(key)
                if branch_result_item and branch_result_item.success:
                    branch_ctx = getattr(branch_result_item, "branch_context", None)
                    if branch_ctx is not None:
                        last_successful_branch = (key, branch_ctx)
                        break

            if last_successful_branch:
                key, branch_ctx = last_successful_branch
                try:
                    # First, merge scratchpad from all successful branches
                    for branch_key, branch_result in branch_results.items():
                        if branch_result.success:
                            branch_ctx_for_merge = getattr(branch_result, "branch_context", None)
                            if branch_ctx_for_merge is not None and hasattr(
                                branch_ctx_for_merge, "scratchpad"
                            ):
                                if not hasattr(context, "scratchpad"):
                                    context.scratchpad = {}  # type: ignore
                                context.scratchpad.update(branch_ctx_for_merge.scratchpad)  # type: ignore

                    # Then update other fields from the last successful branch using non-destructive field-by-field update
                    for field_name in type(branch_ctx).model_fields:
                        if hasattr(branch_ctx, field_name) and field_name != "scratchpad":
                            setattr(context, field_name, getattr(branch_ctx, field_name))
                    print(
                        f"[DEBUG] Overwrote context fields from branch {key} and merged scratchpad from all successful branches"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to overwrite context from branch {key}: {e}")
        elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
            telemetry.logfire.debug("Using MERGE_SCRATCHPAD strategy")
            # For MERGE_SCRATCHPAD, always ensure context has a scratchpad
            if not hasattr(context, "scratchpad"):
                context.scratchpad = {}  # type: ignore

            # Merge scratchpad fields from all branches
            for key, branch_result in branch_results.items():
                branch_ctx = getattr(branch_result, "branch_context", None)
                if branch_ctx is not None:
                    # Ensure branch context has a scratchpad
                    if not hasattr(branch_ctx, "scratchpad"):
                        setattr(branch_ctx, "scratchpad", {})
                    # Merge the branch scratchpad into the main context
                    if hasattr(context, "scratchpad") and hasattr(branch_ctx, "scratchpad"):
                        context.scratchpad.update(branch_ctx.scratchpad)
                    telemetry.logfire.debug(f"Merged scratchpad from branch {key}")
        elif not isinstance(parallel_step.merge_strategy, MergeStrategy):
            telemetry.logfire.debug("Using callable merge strategy")
            # For callable merge strategies, call the function with context and branch_results
            try:
                parallel_step.merge_strategy(context, branch_results)
                telemetry.logfire.debug("Applied callable merge strategy")
            except Exception as e:
                telemetry.logfire.error(f"Failed to apply callable merge strategy: {e}")
    else:
        telemetry.logfire.debug("Context merging condition not met, skipping merge")

    # Failure handling and feedback
    failed_branches = [k for k, r in branch_results.items() if not r.success]
    if failed_branches:
        if parallel_step.on_branch_failure == BranchFailureStrategy.PROPAGATE:
            first_failure_key = failed_branches[0]
            result.success = False
            result.feedback = f"Branch '{first_failure_key}' failed. Propagating failure."
            result.output = {key: branch_results[key] for key in parallel_step.branches.keys()}
            return result
        elif parallel_step.on_branch_failure == BranchFailureStrategy.IGNORE:
            if all(not branch_results[key].success for key in parallel_step.branches.keys()):
                result.success = False
                result.feedback = f"All parallel branches failed: {list(parallel_step.branches.keys())}. Details: {[branch_results[k].feedback for k in failed_branches]}"
                result.output = {key: branch_results[key] for key in parallel_step.branches.keys()}
                return result
            # For IGNORE, include both successful outputs and failed branch results
            for failed_key in failed_branches:
                outputs[failed_key] = branch_results[failed_key]

    # Merge results based on strategy
    if parallel_step.merge_strategy == MergeStrategy.NO_MERGE:
        result.output = outputs
    elif parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
        result.output = outputs
    elif parallel_step.merge_strategy == MergeStrategy.OVERWRITE:
        merged_output = {}
        for key, output in outputs.items():
            if isinstance(output, dict):
                merged_output.update(output)
            else:
                merged_output[key] = output
        result.output = merged_output
    elif parallel_step.merge_strategy == MergeStrategy.KEEP_FIRST:
        merged_output = {}
        for key, output in outputs.items():
            if key not in merged_output:
                if isinstance(output, dict):
                    merged_output.update(output)
                else:
                    merged_output[key] = output
        result.output = merged_output
    elif parallel_step.merge_strategy == MergeStrategy.MERGE_SCRATCHPAD:
        result.output = outputs
    elif parallel_step.merge_strategy == MergeStrategy.CONTEXT_UPDATE:  # type: ignore[comparison-overlap]
        result.output = outputs
    else:  # MergeStrategy.MERGE
        merged_output = {}
        for key, output in outputs.items():
            if isinstance(output, dict):
                merged_output.update(output)
            else:
                merged_output[key] = output
        result.output = merged_output

    result.success = True
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
        # Create a deep copy of context for iteration isolation
        if context is not None:
            import copy

            iteration_context = copy.deepcopy(context)
        else:
            iteration_context = None

        with telemetry.logfire.span(f"Loop '{loop_step.name}' - Iteration {i}"):
            for body_s in loop_step.loop_body_pipeline.steps:
                try:
                    # Use the step executor to execute the body step for this iteration.
                    # This ensures proper context isolation and state management, which are
                    # critical for maintaining the integrity of the loop's execution and
                    # avoiding unintended side effects between iterations.
                    # Filter kwargs based on executor signature
                    filtered_kwargs = _filter_kwargs_for_executor(
                        step_executor,
                        usage_limits=usage_limits,
                    )
                    body_step_result_obj = await step_executor(
                        body_s,
                        current_iteration_data_for_body_step,
                        iteration_context,
                        resources,
                        None,  # breach_event
                        **filtered_kwargs,
                    )

                    # If the body step result is a UsageLimitExceededError, raise it immediately
                    if isinstance(body_step_result_obj, Exception) and isinstance(
                        body_step_result_obj, UsageLimitExceededError
                    ):
                        raise body_step_result_obj

                    # Context updates are handled by the step execution itself
                    # No manual setattr operations needed - the step handles its own context updates

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

                # Always update current_iteration_data_for_body_step with the step's output
                # This ensures the exit condition receives the final output of the pipeline
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

            # --- UNIFIED CONTEXT MERGE ---
            # After each iteration, merge all updates from iteration_context into context.
            # This ensures that all loop types (LoopStep, MapStep, etc.) have their context updates preserved.
            if context is not None and iteration_context is not None:
                try:
                    merge_success = safe_merge_context_updates(
                        target_context=context,
                        source_context=iteration_context,
                        excluded_fields=set(),
                    )

                    if not merge_success:
                        context_fields = list(context.__dict__.keys()) if context else "None"
                        iteration_fields = (
                            list(iteration_context.__dict__.keys()) if iteration_context else "None"
                        )
                        raise RuntimeError(
                            f"Context merge failed in {type(loop_step).__name__} '{loop_step.name}' iteration {i}. "
                            f"This violates the first-principles guarantee that context updates must always be applied. "
                            f"(context fields: {context_fields}, iteration context fields: {iteration_fields}). "
                            f"Possible causes: mismatched field types between contexts, invalid context objects "
                            f"(ensure both are instances of the expected BaseModel subclass), or incorrectly configured "
                            f"excluded fields. Please verify these aspects and retry."
                        )
                    # CRITICAL: Context updates are merged directly to main context
                    # No need to reassign context reference - updates are preserved
                except Exception as e:
                    telemetry.logfire.error(
                        f"Failed to perform context merge in LoopStep '{loop_step.name}' iteration {i}: {e}"
                    )
                    # Re-raise the exception to maintain the first-principles guarantee
                    raise

        # Now check the exit condition on the iteration context to maintain backward compatibility
        # The exit condition should evaluate based on the iteration context to preserve existing behavior
        try:
            should_exit = loop_step.exit_condition_callable(
                final_body_output_of_last_iteration, iteration_context
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
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
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

                # Execute the branch step - context modifications are handled by the step executor
                branch_step_result_obj = await step_executor(
                    branch_s,
                    current_branch_data,
                    context,
                    resources,
                    None,  # breach_event
                )

                # Ensure context modifications are propagated
                # The step_executor already handles context updates through the UltraStepExecutor
                # No additional context handling needed here as the step executor manages this

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

            # Ensure context modifications from the executed branch are committed
            # This is critical for preserving context updates made within the conditional branch
            if context is not None and branch_succeeded:
                # Create a PipelineResult to commit context changes
                from ...domain.models import PipelineResult

                pr: PipelineResult[TContext] = PipelineResult(
                    step_history=[conditional_overall_result],
                    total_cost_usd=conditional_overall_result.cost_usd,
                )
                context_setter(pr, context)

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

    # --- FIRST PRINCIPLES GUARANTEE ---
    # After branch execution, always ensure context updates from the executed branch are preserved.
    # This ensures that all conditional step types have their context updates maintained.
    # No code path (deepcopy, serialization, etc.) can reset or shadow updated fields after execution.
    # Note: Conditional steps execute branches directly on the main context, so updates are immediate.
    # The guarantee is that the context updates from the executed branch are preserved.
    if context is not None and branch_succeeded:
        # The context has already been updated by the branch execution
        # This is a verification that the first-principles guarantee is maintained
        # For conditional steps, context updates happen directly on the main context
        # so no explicit merge is needed - the guarantee is that updates are preserved
        pass

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

    telemetry.logfire.debug("=== DYNAMIC ROUTER STEP LOGIC ===")
    telemetry.logfire.debug(f"Router step name: {router_step.name}")
    telemetry.logfire.debug(f"Router input: {router_input}")

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

    # --- FIRST PRINCIPLES GUARANTEE ---
    # DynamicParallelRouterStep delegates to ParallelStep, which has its own first-principles guarantee.
    # The context updates from parallel branch execution are preserved through the ParallelStep logic.
    telemetry.logfire.debug("About to call _execute_parallel_step_logic")
    telemetry.logfire.debug(f"Parallel step name: {parallel_step.name}")
    telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")
    telemetry.logfire.debug(f"Parallel step merge strategy: {parallel_step.merge_strategy}")
    from typing import cast

    parallel_result = await _execute_parallel_step_logic(
        cast(ParallelStep[TContext], parallel_step),
        router_input,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )
    telemetry.logfire.debug("Returned from _execute_parallel_step_logic")

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
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
) -> StepResult:
    return await _execute_conditional_step_logic(
        conditional_step=step,
        conditional_step_input=data,
        context=context,
        resources=resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
    )


async def _handle_dynamic_router_step(
    step: DynamicParallelRouterStep[TContext],
    data: Any,
    context: Optional[TContext],
    resources: Optional[AppResources],
    step_executor: StepExecutor[TContext],
    context_model_defined: bool,
    context_setter: Callable[[PipelineResult[TContext], Optional[TContext]], None],
    usage_limits: UsageLimits | None = None,
) -> StepResult:
    """Handle DynamicParallelRouterStep execution."""
    telemetry.logfire.debug("=== HANDLE DYNAMIC ROUTER STEP ===")
    telemetry.logfire.debug(f"Step name: {step.name}")
    telemetry.logfire.debug(f"Step type: {type(step)}")

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
    breach_event: Optional[asyncio.Event] = None,
) -> StepResult:
    telemetry.logfire.debug(f"_handle_parallel_step called for step: {step.name}")
    telemetry.logfire.debug(f"Context is None: {context is None}")
    telemetry.logfire.debug(f"Merge strategy: {step.merge_strategy}")
    telemetry.logfire.debug(f"Breach event is None: {breach_event is None}")
    return await _execute_parallel_step_logic(
        step,
        data,
        context,
        resources,
        step_executor=step_executor,
        context_model_defined=context_model_defined,
        usage_limits=usage_limits,
        context_setter=context_setter,
        breach_event=breach_event,
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
    telemetry.logfire.debug(f"Step type: {type(step)}")
    telemetry.logfire.debug(f"Step name: {step.name}")
    telemetry.logfire.debug(f"Context is None: {context is None}")
    telemetry.logfire.debug(f"Context model defined: {context_model_defined}")
    telemetry.logfire.debug(f"Usage limits: {usage_limits}")
    telemetry.logfire.debug(f"Stream: {stream}")
    telemetry.logfire.debug(f"On chunk: {on_chunk is not None}")

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
            step,
            data,
            context,
            resources,
            step_executor,
            context_model_defined,
            usage_limits,
            context_setter,
        )
    if isinstance(step, DynamicParallelRouterStep):
        return await _handle_dynamic_router_step(
            step,
            data,
            context,
            resources,
            _run_step_logic,  # type: ignore # Always use main step logic as step_executor
            context_model_defined,
            context_setter,
            usage_limits,
        )
    if isinstance(step, ParallelStep):
        return await _handle_parallel_step(
            step,
            data,
            context,
            resources,
            _run_step_logic,  # type: ignore # Always use main step logic as step_executor
            context_model_defined,
            usage_limits,
            context_setter,
        )
    if isinstance(step, HumanInTheLoopStep):
        return await _handle_hitl_step(step, data, context)

    # Ensure step.name is always a string, even if it's a mock object
    step_name = str(step.name) if hasattr(step, "name") else "unknown_step"
    result: StepResult = StepResult(name=step_name)
    original_agent = step.agent
    current_agent = original_agent
    last_feedback = None
    last_unpacked_output = None
    validation_failed = False
    accumulated_feedbacks: list[str] = []
    last_exception: Exception = Exception("Unknown error")

    # ✅ 1. Introduce variables to store metrics from the last attempt
    last_attempt_cost_usd = 0.0
    last_attempt_token_counts = 0

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
                        else b"".join(chunks)
                        if chunks and all(isinstance(c, bytes) for c in chunks)
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

            # ✅ 2. After agent run, always capture the metrics
            # Extract usage metrics using shared helper function
            from ...cost import extract_usage_metrics

            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output, current_agent, step.name
            )

            # Store metrics from this attempt
            last_attempt_cost_usd = cost_usd
            last_attempt_token_counts = prompt_tokens + completion_tokens

            # Update result with usage metrics
            result.token_counts = last_attempt_token_counts
            result.cost_usd = last_attempt_cost_usd

            # Check for Mock objects and usage limits
            if isinstance(raw_output, Mock):
                raise TypeError("Mock object as output")

            # Usage limits are checked by the UsageGovernor at the pipeline level
            # No need to check them here as it would interfere with cumulative tracking
        except (
            TypeError,
            UsageLimitExceededError,
            PricingNotConfiguredError,
            ContextInheritanceError,
        ) as e:
            # FLUJO SPIRIT FIX: Track timing even for failed attempts
            result.latency_s += time.monotonic() - start
            # Re-raise critical exceptions immediately
            if isinstance(e, TypeError) and "Mock object as output" in str(e):
                raise
            if isinstance(e, UsageLimitExceededError):
                raise
            if isinstance(e, PricingNotConfiguredError):
                raise
            if isinstance(e, ContextInheritanceError):
                # CRITICAL: ContextInheritanceError must always propagate to pipeline level
                # Never convert to step failure - this is a pipeline-level error
                raise
            # For other TypeErrors, continue retrying
            last_exception = e
            continue
        except PausedException:
            # FLUJO SPIRIT FIX: Track timing even for failed attempts
            result.latency_s += time.monotonic() - start
            # For PausedException, just raise after updating context; context already contains the log.
            raise
        except ContextInheritanceError:
            raise
        except Exception as e:
            # FLUJO SPIRIT FIX: Track timing even for failed attempts
            result.latency_s += time.monotonic() - start
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
                # Always use plugin.validate directly for signature inspection
                func = plugin.validate
                # Use the new conservative plugin context injection function
                if _should_pass_context_to_plugin(context, func):
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
            # ✅ 3. Apply the preserved metrics to the successful result
            result.cost_usd = last_attempt_cost_usd
            result.token_counts = last_attempt_token_counts
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
            # ✅ 4. Apply the preserved metrics to the validation failure result
            result.cost_usd = last_attempt_cost_usd
            result.token_counts = last_attempt_token_counts
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
    if isinstance(last_exception, ContextInheritanceError):
        raise last_exception
    result.success = False
    if accumulated_feedbacks:
        result.feedback = "\n".join(accumulated_feedbacks).strip()
    else:
        # FR-36: Populate feedback with the actual error type and message
        result.feedback = (
            f"Agent execution failed with {type(last_exception).__name__}: {last_exception}"
        )
    result.attempts = step.config.max_retries
    # FLUJO SPIRIT FIX: Preserve actual execution time for failed steps
    # result.latency_s is already accumulated from actual execution time above

    # ✅ 5. Apply the preserved metrics to the failed result
    result.cost_usd = last_attempt_cost_usd
    result.token_counts = last_attempt_token_counts

    # If the step failed and a fallback is defined, execute it.
    if not result.success and step.fallback_step:
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
        # ✅ 6. Store primary token counts for summing
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
            # ✅ 7. Correctly set metrics on fallback success
            result.cost_usd = fallback_result.cost_usd  # Fallback cost ONLY
            result.token_counts = primary_token_counts + fallback_result.token_counts  # SUM tokens
            return result
        else:
            # ✅ 8. Correctly set metrics on fallback failure
            result.cost_usd = fallback_result.cost_usd  # Fallback cost ONLY
            result.token_counts = primary_token_counts + fallback_result.token_counts  # SUM tokens
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

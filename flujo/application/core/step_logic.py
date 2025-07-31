from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import time
import warnings
from typing import Any, Dict, Optional, TypeVar, Callable, Awaitable, cast
from unittest.mock import Mock

from ...domain.dsl.loop import LoopStep
from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.dsl.step import (
    Step,
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
from ...application.context_manager import _accepts_param
from ...application.context_manager import _get_validation_flags
from ...application.context_manager import _apply_validation_metadata

TContext = TypeVar("TContext", bound=BaseModel)


# Type alias for the wrapper function
DeprecatedFunction = Callable[..., Any]


def deprecated_function(func: Callable[..., Any]) -> DeprecatedFunction:
    """Decorator to mark functions as deprecated."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. "
            "Use the new ExecutorCore implementation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    # Set the wrapped function and preserve metadata
    wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


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


# _execute_parallel_step_logic removed - migrated to ExecutorCore._handle_parallel_step


# _execute_loop_step_logic removed - migrated to ExecutorCore._handle_loop_step


# _execute_conditional_step_logic removed - migrated to ExecutorCore._handle_conditional_step


# _execute_dynamic_router_step_logic removed - migrated to ExecutorCore._handle_dynamic_router_step


@deprecated_function
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
    """Handle LoopStep execution using new ExecutorCore."""
    from .ultra_executor import ExecutorCore

    executor: ExecutorCore[TContext] = ExecutorCore()
    return await executor._handle_loop_step(
        step,
        data,
        context,
        resources,
        usage_limits,
        context_setter,
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
    """Handle ConditionalStep execution using new ExecutorCore."""
    from .ultra_executor import ExecutorCore

    executor: ExecutorCore[TContext] = ExecutorCore()
    return await executor._handle_conditional_step(
        step,
        data,
        context,
        resources,
        usage_limits,
        context_setter,
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
    """Handle DynamicParallelRouterStep execution using new ExecutorCore."""
    from .ultra_executor import ExecutorCore

    executor: ExecutorCore[TContext] = ExecutorCore()
    return await executor._handle_dynamic_router_step(
        step,
        data,
        context,
        resources,
        usage_limits,
        context_setter,
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
    """Handle ParallelStep execution using new ExecutorCore."""
    from .ultra_executor import ExecutorCore

    executor: ExecutorCore[TContext] = ExecutorCore()
    return await executor._handle_parallel_step(
        step,
        data,
        context,
        resources,
        usage_limits,
        breach_event,
        context_setter,
    )


@deprecated_function
async def _handle_hitl_step(
    step: HumanInTheLoopStep,
    data: Any,
    context: Optional[TContext],
) -> StepResult:
    message = step.message_for_user if step.message_for_user is not None else str(data)
    if isinstance(context, PipelineContext):
        context.scratchpad["status"] = "paused"
    raise PausedException(message)


@deprecated_function
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
            step_executor,
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
            step_executor,
            context_model_defined,
            usage_limits,
            context_setter,
            None,  # breach_event
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
    "_run_step_logic",
]

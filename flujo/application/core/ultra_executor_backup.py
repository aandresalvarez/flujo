"""
Ultra-optimized step executor v2 with modular, policy-driven architecture.

This is a complete rewrite of the UltraStepExecutor with:
- Modular design with clear separation of concerns
- Deterministic behavior across processes and restarts
- Pluggable components via dependency injection
- Robust isolation between concerns
- Exhaustive accounting of successful and failed attempts
- Backward compatibility with existing SDK signatures

Author: Flujo Team
Version: 2.0
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import hashlib
import copy
import multiprocessing
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import cached_property, wraps
from multiprocessing import cpu_count
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    cast,
)
import types
from types import SimpleNamespace
from asyncio import Task
import weakref
from weakref import WeakKeyDictionary

from ...domain.dsl.step import HumanInTheLoopStep, Step, MergeStrategy, BranchFailureStrategy
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.conditional import ConditionalStep
from .types import TContext_w_Scratch, ExecutionFrame
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.models import BaseModel, StepResult, UsageLimits, PipelineResult, PipelineContext
from ...domain.processors import AgentProcessors
from pydantic import Field
from ...domain.validation import ValidationResult
from ...exceptions import (
    UsageLimitExceededError,
    PausedException,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PricingNotConfiguredError,
    ContextInheritanceError,
    MissingAgentError,
)
from flujo.utils.formatting import format_cost

# Type alias for step executor function signature
StepExecutor = Callable[
    [Step[Any, Any], Any, Optional[Any], Optional[Any], Optional[Any]],
    Awaitable[StepResult],
]


# Exception classification for retry logic
class RetryableError(Exception):
    """Base class for errors that should trigger retries."""

    pass


class NonRetryableError(Exception):
    """Base class for errors that should not trigger retries."""

    pass


# Classify common exceptions
class ValidationError(RetryableError):
    """Validation failures that can be retried."""

    pass


class PluginError(RetryableError):
    """Plugin failures that can be retried."""

    pass


class AgentError(RetryableError):
    """Agent execution errors that can be retried."""

    pass


# Removed unused imports: _manage_fallback_relationships, _detect_fallback_loop
from ...steps.cache_step import CacheStep
from ...utils.performance import time_perf_ns, time_perf_ns_to_seconds
from ...cost import extract_usage_metrics
from ...infra import telemetry
from ...signature_tools import analyze_signature
from ...application.context_manager import _accepts_param
from ...utils.context import safe_merge_context_updates


# --------------------------------------------------------------------------- #
# ★ Pipeline-to-Step Adapter
# --------------------------------------------------------------------------- #


class _PipelineStepAdapter(Step[Any, Any]):
    """
    Adapter that wraps a Pipeline object to satisfy the Step interface.

    This ensures algebraic closure: every higher-order control-flow helper
    returns a Step, enabling auto-insertion of features anywhere in the graph.
    """

    def __init__(self, pipeline: Any, name: str):
        super().__init__(name=name, agent=self)  # `self` is the "agent"
        self._pipeline = pipeline

        # Copy only safe attributes from the pipeline to maintain compatibility
        # Avoid copying methods and attributes that conflict with Step fields
        safe_attrs = [
            "name",
            "config",
            "plugins",
            "validators",
            "processors",
            "fallback_step",
            "usage_limits",
            "persist_feedback_to_context",
            "persist_validation_results_to",
            "updates_context",
            "validate_fields",
            "meta",
        ]
        for attr in safe_attrs:
            if hasattr(pipeline, attr) and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(pipeline, attr))
                except (ValueError, AttributeError):
                    # Skip attributes that can't be set
                    pass

    # --------------------------------------------------------------------- #
    # Agent interface used by DefaultAgentRunner
    # --------------------------------------------------------------------- #
    async def run(
        self, payload: Any, *, context: Any = None, resources: Any = None, **kwargs: Any
    ) -> Any:
        """Execute the wrapped pipeline by executing its steps."""
        # Execute the pipeline by running its steps through the executor
        # This is a simplified approach - in practice, we'd need the executor
        # For now, let's execute the first step as a fallback
        if self._pipeline.steps:
            first_step = self._pipeline.steps[0]
            # Execute the first step directly
            if hasattr(first_step.agent, "run"):
                # Only pass the parameters that the agent's run method expects
                # Check the signature of the agent's run method
                import inspect

                run_method = first_step.agent.run
                sig = inspect.signature(run_method)

                # Only pass parameters that the method actually accepts
                run_kwargs = {}
                if "context" in sig.parameters:
                    run_kwargs["context"] = context

                # Add any additional kwargs that the method accepts
                for key, value in kwargs.items():
                    if key in sig.parameters and key not in run_kwargs:
                        run_kwargs[key] = value

                return await first_step.agent.run(payload, **run_kwargs)
            else:
                # Fallback: return the payload as-is
                return payload
        else:
            # Empty pipeline, return payload as-is
            return payload

    # --------------------------------------------------------------------- #
    # Step interface attributes (no-op stubs for compatibility)
    # --------------------------------------------------------------------- #
    config: Any = SimpleNamespace(temperature=0.0, max_retries=1)
    validators: List[Any] = []
    plugins: List[Any] = []
    processors: AgentProcessors = Field(default_factory=AgentProcessors)
    persist_validation_results_to: Optional[str] = None
    persist_feedback_to_context: Optional[str] = None
    updates_context: bool = False
    fallback_step: Optional[Any] = None

    # --------------------------------------------------------------------- #
    # Attribute access delegation
    # --------------------------------------------------------------------- #
    def __getattribute__(self, name: str) -> Any:
        # Special handling for agent field
        if name == "agent":
            return self
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Special handling for agent field
        if name == "agent":
            # Ignore agent assignment as we're the agent
            return
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        # Fallback for any other attributes not found
        if name == "agent":
            return self
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# --------------------------------------------------------------------------- #
# ★ Interfaces (Protocols)
# --------------------------------------------------------------------------- #


class ISerializer(Protocol):
    """Interface for object serialization."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    @abstractmethod
    def deserialize(self, blob: bytes) -> Any:
        """Deserialize bytes back to an object."""
        ...


class IHasher(Protocol):
    """Interface for deterministic hashing."""

    @abstractmethod
    def digest(self, data: bytes) -> str:
        """Generate a deterministic hash digest from bytes."""
        ...


class ICacheBackend(Protocol):
    """Interface for caching step results."""

    @abstractmethod
    async def get(self, key: str) -> Optional[StepResult]:
        """Retrieve a cached result by key."""
        ...

    @abstractmethod
    async def put(self, key: str, value: StepResult, ttl_s: int) -> None:
        """Store a result in cache with TTL."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...


class IUsageMeter(Protocol):
    """Interface for tracking and enforcing usage limits."""

    @abstractmethod
    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        """Add usage metrics to cumulative totals."""
        ...

    @abstractmethod
    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None) -> None:
        """Check if current usage exceeds limits, raise if so."""
        ...

    @abstractmethod
    async def snapshot(self) -> tuple[float, int, int]:
        """Get current (cost, prompt_tokens, completion_tokens)."""
        ...


class IAgentRunner(Protocol):
    """Interface for running agents with proper parameter handling."""

    @abstractmethod
    async def run(
        self,
        agent: Any,
        payload: Any,
        *,
        context: Any,
        resources: Any,
        options: Dict[str, Any],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
    ) -> Any:
        """Run an agent and return raw output."""
        ...


class IProcessorPipeline(Protocol):
    """Interface for running prompt and output processors."""

    @abstractmethod
    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors to input data."""
        ...

    @abstractmethod
    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors to agent output."""
        ...


class IValidatorRunner(Protocol):
    """Interface for running validators."""

    @abstractmethod
    async def validate(self, validators: List[Any], data: Any, *, context: Any) -> None:
        """Run validators and raise ValueError on first failure."""
        if not validators:
            return

        for validator in validators:
            try:
                result = await validator.validate(data, context=context)
                if isinstance(result, ValidationResult) and not result.is_valid:
                    # Use feedback field instead of message
                    feedback = result.feedback or "Validation failed"
                    raise ValueError(f"Validation failed: {feedback}")
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                raise ValueError(f"Validator {type(validator).__name__} failed: {e}")


class IPluginRunner(Protocol):
    """Interface for running plugins."""

    @abstractmethod
    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any) -> Any:
        """Run plugins and return processed data."""
        ...


class ITelemetry(Protocol):
    """Interface for telemetry operations."""

    @abstractmethod
    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Create a telemetry trace decorator."""
        ...


# --------------------------------------------------------------------------- #
# ★ Default Implementations
# --------------------------------------------------------------------------- #


class OrjsonSerializer:
    """Fast JSON serializer using orjson if available."""

    def __init__(self) -> None:
        try:
            import orjson

            self._orjson = orjson
            self._use_orjson = True
        except ImportError:
            import json

            self._json = json
            self._use_orjson = False

    def serialize(self, obj: Any) -> bytes:
        if self._use_orjson:
            return self._orjson.dumps(obj, option=self._orjson.OPT_SORT_KEYS)
        else:
            s = self._json.dumps(obj, sort_keys=True, separators=( "," , ":" ))
            return s.encode("utf-8")

    def deserialize(self, blob: bytes) -> Any:
        if self._use_orjson:
            return self._orjson.loads(blob)
        else:
            return self._json.loads(blob.decode("utf-8"))


class Blake3Hasher:
    """Fast cryptographic hasher using Blake3 if available."""

    def __init__(self) -> None:
        try:
            import blake3

            self._blake3 = blake3
            self._use_blake3 = True
        except ImportError:
            self._use_blake3 = False

    def digest(self, data: bytes) -> str:
        if self._use_blake3:
            return self._blake3.blake3(data).hexdigest()
        else:
            return hashlib.blake2b(data, digest_size=32).hexdigest()


@dataclass
class InMemoryLRUBackend:
    """O(1) LRU cache with TTL support."""

    max_size: int = 1024
    ttl_s: int = 3600
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _store: OrderedDict[str, tuple[StepResult, float]] = field(
        init=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl_s < 0:
            raise ValueError("ttl must be non-negative")

    async def get(self, key: str) -> Optional[StepResult]:
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None

            result, timestamp = item
            now = time.monotonic()

            # Check TTL (0 means never expire)
            if self.ttl_s > 0 and now - timestamp > self.ttl_s:
                self._store.pop(key, None)
                return None

            # LRU promotion
            self._store.move_to_end(key)
            return result.model_copy(deep=True)  # Return a deep copy to prevent mutation

    async def put(self, key: str, value: StepResult, ttl_s: int) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self.max_size:
                self._store.popitem(last=False)  # Remove oldest

            self._store[key] = (
                value.model_copy(deep=True),
                time.monotonic(),
            )

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    # Backward-compatibility alias
    async def set(self, key: str, value: StepResult, ttl_s: int) -> None:
        """Alias to :py:meth:`put` retained for older call-sites."""
        await self.put(key, value, ttl_s)


@dataclass
class ThreadSafeMeter:
    """Thread-safe usage meter with atomic operations."""

    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        async with self._lock:
            self.total_cost_usd += cost_usd
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None) -> None:
        async with self._lock:
            # Use precise comparison for floating point
            if (
                limits.total_cost_usd_limit is not None
                and self.total_cost_usd - limits.total_cost_usd_limit > 1e-9
            ):
                # Format cost values to remove trailing zeros (e.g., 0.5 instead of 0.50)
                formatted_limit = format_cost(limits.total_cost_usd_limit)
                formatted_current = format_cost(self.total_cost_usd)
                raise UsageLimitExceededError(
                    f"Cost limit of ${formatted_limit} exceeded. Current cost: ${formatted_current}"
                )

            total_tokens = self.prompt_tokens + self.completion_tokens
            if (
                limits.total_tokens_limit is not None
                and total_tokens - limits.total_tokens_limit > 0
            ):
                raise UsageLimitExceededError(
                    f"Token limit of {limits.total_tokens_limit} exceeded. Current tokens: {total_tokens}"
                )

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._lock:
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens


class DefaultAgentRunner:
    """Default agent runner with parameter filtering and streaming support."""

    async def run(
        self,
        agent: Any,
        payload: Any,
        *,
        context: Any,
        resources: Any,
        options: Dict[str, Any],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
    ) -> Any:
        """Run agent with proper parameter filtering and fallback strategies."""
        import inspect
        from unittest.mock import Mock, MagicMock, AsyncMock
        from ..context_manager import _accepts_param
        from ...signature_tools import analyze_signature

        print(f"🔍 DefaultAgentRunner.run called with agent: {agent}")
        print(f"🔍 Agent type: {type(agent)}")

        if agent is None:
            raise RuntimeError("Agent is None")

        # Step 1: Extract the target agent (handle wrapped agents)
        target_agent = getattr(agent, "_agent", agent)
        print(f"🔍 Target agent: {target_agent}")

        # Step 2: Find the executable function
        executable_func = None

        # For Mock objects, be more careful about method resolution
        # Prioritize explicitly set methods over auto-generated mock attributes
        if isinstance(target_agent, (Mock, MagicMock, AsyncMock)) or isinstance(
            agent, (Mock, MagicMock, AsyncMock)
        ):
            # For mocks, check if run method was explicitly set first
            if hasattr(agent, "run") and not isinstance(
                getattr(agent, "run"), (Mock, MagicMock, AsyncMock)
            ):
                executable_func = getattr(agent, "run")
            elif hasattr(target_agent, "run") and not isinstance(
                getattr(target_agent, "run"), (Mock, MagicMock, AsyncMock)
            ):
                executable_func = getattr(target_agent, "run")
            elif (
                stream
                and hasattr(target_agent, "stream")
                and not isinstance(getattr(target_agent, "stream"), (Mock, MagicMock, AsyncMock))
            ):
                executable_func = getattr(target_agent, "stream")
            elif hasattr(agent, "run"):
                # Use run method even if it's a mock (for test fixtures)
                executable_func = getattr(agent, "run")
            elif hasattr(target_agent, "run"):
                # Use run method even if it's a mock (for test fixtures)
                executable_func = getattr(target_agent, "run")
            elif hasattr(target_agent, "_step_callable"):
                executable_func = getattr(target_agent, "_step_callable")
            else:
                # Fall back to run method even if it's a mock
                executable_func = getattr(agent, "run", getattr(target_agent, "run", None))
        else:
            # For non-mock objects, use the original priority order
            # Try stream method first for streaming
            if stream and hasattr(target_agent, "stream"):
                executable_func = getattr(target_agent, "stream")

            # Try run method on the original agent first (wrapper)
            elif hasattr(agent, "run"):
                executable_func = getattr(agent, "run")

            # Try run method on the target agent (wrapped)
            elif hasattr(target_agent, "run"):
                executable_func = getattr(target_agent, "run")

            # Try if the agent itself is callable
            elif callable(target_agent):
                executable_func = target_agent

            # Last resort: check the original agent for run method
            elif hasattr(agent, "run"):
                executable_func = getattr(agent, "run")

        print(f"🔍 Executable func: {executable_func}")
        print(f"🔍 Executable func type: {type(executable_func)}")

        if executable_func is None:
            raise RuntimeError(
                f"Agent {type(agent).__name__} has no executable method (run, stream, _step_callable) and is not callable"
            )

        # Step 3: Handle mocks specially
        is_mock = isinstance(executable_func, (Mock, MagicMock, AsyncMock))

        # Step 4: Build filtered kwargs based on function signature
        filtered_kwargs: Dict[str, Any] = {}

        if not is_mock:
            try:
                spec = analyze_signature(executable_func)

                # Add context if the function accepts it
                if spec.needs_context:
                    if context is None:
                        raise TypeError(f"Agent requires a context, but no context was provided.")
                    filtered_kwargs["context"] = context
                elif context is not None and _accepts_param(executable_func, "context"):
                    filtered_kwargs["context"] = context

                # Add resources if the function accepts it
                if resources is not None and _accepts_param(executable_func, "resources"):
                    filtered_kwargs["resources"] = resources

                # Add other options based on function signature
                for key, value in options.items():
                    if value is not None and _accepts_param(executable_func, key):
                        filtered_kwargs[key] = value

                # Add breach_event if the function accepts it
                if breach_event is not None and _accepts_param(executable_func, "breach_event"):
                    filtered_kwargs["breach_event"] = breach_event

            except Exception:
                # If signature analysis fails, try basic parameter passing
                # Pass all non-None options (legacy behavior for backward compatibility)
                for key, value in options.items():
                    if value is not None:
                        filtered_kwargs[key] = value
                # Use proper context filtering
                if context is not None and (
                    _accepts_param(executable_func, "context") is not False
                ):
                    filtered_kwargs["context"] = context
                if resources is not None and (
                    _accepts_param(executable_func, "resources") is not False
                ):
                    filtered_kwargs["resources"] = resources
                if breach_event is not None and (
                    _accepts_param(executable_func, "breach_event") is not False
                ):
                    filtered_kwargs["breach_event"] = breach_event
        else:
            # For mocks, pass all parameters but filter context properly
            for key, value in options.items():
                if value is not None:
                    filtered_kwargs[key] = value
            # Use proper context filtering for mocks too
            if context is not None and (_accepts_param(executable_func, "context") is not False):
                filtered_kwargs["context"] = context
            if resources is not None and (
                _accepts_param(executable_func, "resources") is not False
            ):
                filtered_kwargs["resources"] = resources
            if breach_event is not None and (
                _accepts_param(executable_func, "breach_event") is not False
            ):
                filtered_kwargs["breach_event"] = breach_event

        # Step 5: Execute the agent
        if stream and hasattr(target_agent, "stream"):
            # Handle streaming
            chunks = []
            stream_func = getattr(target_agent, "stream")

            try:
                # Call the stream function and check if it returns an async iterator
                if inspect.iscoroutinefunction(stream_func):
                    stream_result = await stream_func(payload, **filtered_kwargs)
                else:
                    stream_result = stream_func(payload, **filtered_kwargs)

                if stream_result is None:
                    raise RuntimeError("Stream function returned None")

                if not hasattr(stream_result, "__aiter__"):
                    raise RuntimeError(
                        f"Stream function did not return an async iterator: {type(stream_result)}"
                    )

                async for chunk in stream_result:
                    if on_chunk:
                        await on_chunk(chunk)
                    chunks.append(chunk)
            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                raise e
            except Exception as e:
                # If streaming fails, fall back to regular execution
                if hasattr(target_agent, "run"):
                    return await target_agent.run(payload, **filtered_kwargs)
                else:
                    raise e

            # Combine chunks safely
            if chunks and all(isinstance(c, str) for c in chunks):
                return "".join(chunks)
            elif chunks and all(isinstance(c, bytes) for c in chunks):
                return b"".join(chunks)
            elif chunks:
                return str(chunks)
            else:
                return ""
        else:
            # Handle regular execution
            try:
                if inspect.iscoroutinefunction(executable_func):
                    return await executable_func(payload, **filtered_kwargs)
                else:
                    # Handle sync functions by wrapping them
                    result = executable_func(payload, **filtered_kwargs)
                    # If it returns a coroutine, await it
                    if inspect.iscoroutine(result):
                        return await result
                    return result
            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                raise e

    def _clone_payload_for_retry(self, original_data: Any, accumulated_feedbacks: list[str]) -> Any:
        """Clone payload for retry attempts with accumulated feedback injection.
        
        This method creates a safe copy of the original payload and injects accumulated
        feedback from previous failed attempts. It handles various data types:
        
        - Dict objects: Creates a shallow copy and appends feedback to 'feedback' key
        - Pydantic models: Uses model_copy() for efficient copying and appends to feedback field
        - Dataclasses: Creates a copy using copy.deepcopy() and appends to feedback field
        - Other types: Converts to string and appends feedback as newlines
        
        Args:
            original_data: The original payload to clone
            accumulated_feedbacks: List of error messages from previous attempts
            
        Returns:
            A cloned payload with accumulated feedback injected
        """
        if not accumulated_feedbacks:
            # No feedback to add, return original data unchanged
            return original_data

        feedback_text = "\n".join(accumulated_feedbacks)

        # Handle dict objects (most common case)
        if isinstance(original_data, dict):
            cloned_data = original_data.copy()
            existing_feedback = cloned_data.get("feedback", "")
            cloned_data["feedback"] = (existing_feedback + "\n" + feedback_text).strip()
            return cloned_data
            
        # Handle Pydantic models with efficient model_copy
        elif hasattr(original_data, "model_copy"):
            cloned_data = original_data.model_copy(deep=False)
            if hasattr(cloned_data, "feedback"):
                existing_feedback = getattr(cloned_data, "feedback", "")
                setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
            return cloned_data
            
        # Handle dataclasses and other objects with copy support
        elif hasattr(original_data, "__dict__"):
            import copy
            try:
                cloned_data = copy.deepcopy(original_data)
                if hasattr(cloned_data, "feedback"):
                    existing_feedback = getattr(cloned_data, "feedback", "")
                    setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
                return cloned_data
            except (TypeError, RecursionError):
                # Fall back to string conversion if deep copy fails
                pass
                
        # Handle list/tuple types
        elif isinstance(original_data, (list, tuple)):
            try:
                cloned_data = original_data.copy() if isinstance(original_data, list) else list(original_data)
                # Try to add feedback to the first element if it's a dict
                if cloned_data and isinstance(cloned_data[0], dict):
                    cloned_data[0]["feedback"] = cloned_data[0].get("feedback", "") + "\n" + feedback_text
                return cloned_data
            except (AttributeError, TypeError):
                pass
                
        # Fallback: convert to string and append feedback
        return f"{str(original_data)}\n{feedback_text}"


class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with prompt_processors attribute
        processor_list = processors
        if hasattr(processors, "prompt_processors"):
            processor_list = processors.prompt_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Log error but continue with original data
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Prompt processor failed: {e}")
                except Exception:
                    pass
                processed_data = data

        return processed_data

    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with output_processors attribute
        processor_list = processors
        if hasattr(processors, "output_processors"):
            processor_list = processors.output_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Re-raise processor exceptions to cause step failure
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                # Re-raise the exception to cause step failure
                raise

        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(self, validators: List[Any], data: Any, *, context: Any) -> None:
        """Run validators and raise ValueError on first failure."""
        if not validators:
            return

        for validator in validators:
            try:
                result = await validator.validate(data, context=context)
                if isinstance(result, ValidationResult) and not result.is_valid:
                    # Use feedback field instead of message
                    feedback = result.feedback or "Validation failed"
                    raise ValueError(f"Validation failed: {feedback}")
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                raise ValueError(f"Validator {type(validator).__name__} failed: {e}")


class DefaultPluginRunner:
    """Default plugin runner implementation."""

    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any) -> Any:
        """Run plugins in priority order."""
        processed_data = data

        if not plugins:
            return processed_data

        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                from ...domain.plugins import PluginOutcome

                # Call plugin with proper parameter filtering
                plugin_kwargs = {}
                if hasattr(plugin, "validate"):
                    func = plugin.validate
                else:
                    func = plugin

                # Check if plugin accepts context using conservative logic
                # Create a temporary ExecutorCore instance to access the method
                temp_executor: ExecutorCore[Any] = ExecutorCore()
                if temp_executor._should_pass_context_to_plugin(context, func):
                    plugin_kwargs["context"] = context

                # Call plugin
                result = await func(processed_data, **plugin_kwargs)

                # Handle PluginOutcome
                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # NEW: Raise an exception to fail the step
                        raise ValueError(f"Plugin validation failed: {result.feedback}")
                    if result.new_solution is not None:
                        processed_data = result.new_solution
                else:
                    # Plugin returned new data
                    processed_data = result

            except Exception as e:
                # Log error and re-raise to cause step failure (explicit plugin failure handling)
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Plugin {type(plugin).__name__} failed: {e}")
                except Exception:
                    pass
                # Re-raise the exception to cause step failure
                raise e

        return processed_data


class DefaultTelemetry:
    """Default telemetry implementation."""

    def __init__(self) -> None:
        try:
            from ...infra import telemetry

            self._telemetry = telemetry
            self._available = True
        except Exception:
            self._available = False

    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if self._available:
                try:
                    return self._telemetry.logfire.instrument(name)(func)
                except Exception:
                    pass
            return func

        return decorator


# --------------------------------------------------------------------------- #
# ★ Cache Key Generation
# --------------------------------------------------------------------------- #


@dataclass
class CacheKeyComponents:
    """Components for deterministic cache key generation."""

    step_name: str
    step_type: str
    payload_digest: str
    context_digest: Optional[str]
    resource_digest: Optional[str]
    agent_id: Optional[str]


class CacheKeyGenerator:
    """Generates deterministic cache keys from step execution parameters."""

    def __init__(self, serializer: ISerializer, hasher: IHasher):
        self._serializer = serializer
        self._hasher = hasher
        self._seen_hashes: WeakKeyDictionary[Any, str] = WeakKeyDictionary()

    def generate_key(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
    ) -> str:
        """Generate a deterministic cache key."""
        step_name = getattr(step, "name", None)
        if step_name is None:
            # Fallback for objects (like Pipeline) that lack a stable name attribute
            step_name = f"{type(step).__name__}:{id(step)}"

        components = CacheKeyComponents(
            step_name=step_name,
            step_type=type(step).__name__,
            payload_digest=self._hash_obj(data),
            context_digest=self._hash_obj(context) if context is not None else None,
            resource_digest=self._hash_obj(resources) if resources is not None else None,
            agent_id=self._get_agent_id(step),
        )

        # Serialize components for final hash
        serialized = self._serializer.serialize(
            {
                "step_name": components.step_name,
                "step_type": components.step_type,
                "payload_digest": components.payload_digest,
                "context_digest": components.context_digest,
                "resource_digest": components.resource_digest,
                "agent_id": components.agent_id,
            }
        )

        return self._hasher.digest(serialized)

    def _hash_obj(self, obj: Any) -> str:
        """Hash any Python object deterministically."""
        # Fast path for common types
        if obj is None:
            return "null"
        if isinstance(obj, (str, int, float, bool)):
            return self._hasher.digest(str(obj).encode())
        if isinstance(obj, bytes):
            return self._hasher.digest(obj)

        # Check cache first
        try:
            if obj in self._seen_hashes:
                return self._seen_hashes[obj]
        except (TypeError, KeyError):
            pass

        # Handle BaseModel efficiently
        if isinstance(obj, BaseModel):
            try:
                serialized = obj.model_dump_json(sort_keys=True).encode()
                h = self._hasher.digest(serialized)
            except Exception:
                try:
                    data = obj.model_dump()
                    serialized_data = self._serialize_for_hash(data)
                    h = self._hasher.digest(self._serializer.serialize(serialized_data))
                except Exception:
                    h = self._hasher.digest(repr(obj).encode())
        else:
            try:
                h = self._hasher.digest(self._serializer.serialize(obj))
            except (TypeError, ValueError):
                try:
                    serialized = self._serialize_for_hash(obj)
                    h = self._hasher.digest(self._serializer.serialize(serialized))
                except Exception:
                    h = self._hasher.digest(repr(obj).encode())

        # Cache result
        try:
            self._seen_hashes[obj] = h
        except (TypeError, KeyError):
            pass

        return h

    def _serialize_for_hash(self, obj: Any) -> Any:
        """Serialize object for hashing, handling special cases."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_hash(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_hash(item) for item in obj]
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__class__") and "Mock" in obj.__class__.__name__:
            return f"Mock({type(obj).__name__})"
        elif hasattr(obj, "__dict__"):
            try:
                return {k: self._serialize_for_hash(v) for k, v in obj.__dict__.items()}
            except Exception:
                return repr(obj)
        else:
            return obj

    def _get_agent_id(self, step: Any) -> Optional[str]:
        """Get stable agent identifier."""
        agent = getattr(step, "agent", None)
        if agent is None:
            return None

        # Use agent type and configuration for stable identification
        agent_type = f"{type(agent).__module__}.{type(agent).__name__}"
        agent_config = getattr(agent, "config", None)

        if agent_config:
            config_hash = self._hash_obj(agent_config)
            return f"{agent_type}:{config_hash}"
        else:
            return agent_type


# --------------------------------------------------------------------------- #
# ★ Internal Types
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ★ ExecutorCore (Main Implementation)
# --------------------------------------------------------------------------- #




class ExecutorCore(Generic[TContext_w_Scratch]):
    """
    Ultra-optimized step executor with modular, policy-driven architecture.

    This maintains the exact same API as the original UltraStepExecutor
    while providing enhanced performance, reliability, and extensibility.
    """

    # Context variables for tracking fallback relationships and chains
    _fallback_relationships: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
        "fallback_relationships", default={}
    )
    _fallback_chain: contextvars.ContextVar[list[Step[Any, Any]]] = contextvars.ContextVar(
        "fallback_chain", default=[]
    )

    # Cache for fallback relationship loop detection (True if loop detected, False otherwise)
    _fallback_graph_cache: contextvars.ContextVar[Dict[str, bool]] = contextvars.ContextVar(
        "fallback_graph_cache", default={}
    )

    # Maximum length for fallback chains to prevent infinite loops
    _MAX_FALLBACK_CHAIN_LENGTH = 10

    # Maximum iterations for fallback loop detection to prevent infinite loops
    _DEFAULT_MAX_FALLBACK_ITERATIONS = 100

    class _ParallelUsageGovernor:
        """Helper to track and enforce usage limits atomically across parallel branches."""

        def __init__(self, usage_limits: Optional[UsageLimits]) -> None:
            self.usage_limits = usage_limits
            self.lock = asyncio.Lock()
            self.total_cost = 0.0
            self.total_tokens = 0
            self.limit_breached = asyncio.Event()
            self.limit_breach_error: Optional[UsageLimitExceededError] = None
            self._breach_detected = False  # Add explicit flag to prevent race conditions

        def _create_breach_error_message(
            self,
            limit_type: str,
            limit_value: Any,
            current_value: Any,
        ) -> str:
            """Create a breach error message string."""
            if limit_type == "cost":
                # Format cost values to remove trailing zeros (e.g., 0.5 instead of 0.50)
                formatted_limit = format_cost(limit_value)
                formatted_current = format_cost(current_value)
                return f"Cost limit of ${formatted_limit} exceeded. Current cost: ${formatted_current}"
            else:  # token
                return f"Token limit of {limit_value} exceeded. Current tokens: {current_value}"

        async def add_usage(self, cost_delta: float, token_delta: int, result: StepResult) -> bool:
            """Add usage and check for breach. Returns True if breach occurred."""
            try:
                # Add timeout to prevent infinite lock waits
                async with asyncio.timeout(5.0):  # 5 second timeout
                    async with self.lock:
                        # Always update totals for accurate accounting, even after breach
                        self.total_cost += cost_delta
                        self.total_tokens += token_delta

                        # Check for breach only if limits are configured
                        if self.usage_limits is not None:
                            breach_occurred = False

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
                                breach_occurred = True
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
                                breach_occurred = True

                            if breach_occurred:
                                self._breach_detected = True
                                self.limit_breached.set()

                        return self._breach_detected
            except asyncio.TimeoutError:
                # If we can't acquire the lock within timeout, assume breach to be safe
                return True

        def breached(self) -> bool:
            """Check if a limit has been breached."""
            return self._breach_detected or self.limit_breached.is_set()

        def get_error_message(self) -> Optional[str]:
            """Get the error message if a breach occurred."""
            if self.limit_breach_error:
                return self.limit_breach_error.args[0]
            return None

        def get_error(self) -> Optional[UsageLimitExceededError]:
            """Get the error if a breach occurred."""
            return self.limit_breach_error

    def _should_pass_context(
        self,
        spec: Any,
        context: Optional[Any],
        func: Callable[..., Any],
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

    def _should_pass_context_to_plugin(
        self,
        context: Optional[Any],
        func: Callable[..., Any],
    ) -> bool:
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

    def _default_set_final_context(
        self,
        result: "PipelineResult[Any]",
        ctx: Optional[Any],
    ) -> None:
        """Default function to set final context from pipeline result."""
        if ctx is not None and result.final_pipeline_context is not None:
            for (
                field_name,
                field_value,
            ) in result.final_pipeline_context.model_dump().items():
                setattr(ctx, field_name, field_value)

    def _manage_fallback_relationships(
        self,
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
        self.__class__._fallback_graph_cache.set({})

        relationships = self.__class__._fallback_relationships.get()
        relationships_token = self.__class__._fallback_relationships.set(
            {**relationships, step.name: step.fallback_step.name}
        )
        return relationships_token

    def _detect_fallback_loop(self, step: Step[Any, Any], chain: list[Step[Any, Any]]) -> bool:
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
        if len(chain) >= self._MAX_FALLBACK_CHAIN_LENGTH:
            return True

        # Strategy 4: Global relationship loop detection with caching
        relationships = self.__class__._fallback_relationships.get()
        if step.name in relationships:
            # Use cached graph for improved performance
            graph_cache = self.__class__._fallback_graph_cache.get()

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
                max_iterations = self._DEFAULT_MAX_FALLBACK_ITERATIONS
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

    def __init__(
        self,
        *,
        serializer: Optional[ISerializer] = None,
        hasher: Optional[IHasher] = None,
        cache_backend: Optional[ICacheBackend] = None,
        usage_meter: Optional[IUsageMeter] = None,
        agent_runner: Optional[IAgentRunner] = None,
        processor_pipeline: Optional[IProcessorPipeline] = None,
        validator_runner: Optional[IValidatorRunner] = None,
        plugin_runner: Optional[IPluginRunner] = None,
        telemetry: Optional[ITelemetry] = None,
        concurrency_limit: Optional[int] = None,
        enable_cache: bool = True,
    ) -> None:
        # Initialize components with defaults
        self._serializer = serializer or OrjsonSerializer()
        self._hasher = hasher or Blake3Hasher()
        self._cache_backend = cache_backend if enable_cache else None
        self._usage_meter = usage_meter or ThreadSafeMeter()
        self._agent_runner = agent_runner or DefaultAgentRunner()
        self._processor_pipeline = processor_pipeline or DefaultProcessorPipeline()
        self._validator_runner = validator_runner or DefaultValidatorRunner()
        self._plugin_runner = plugin_runner or DefaultPluginRunner()
        self._telemetry = telemetry or DefaultTelemetry()
        self._enable_cache = enable_cache

        # Initialize cache key generator
        self._cache_key_generator = CacheKeyGenerator(self._serializer, self._hasher)

        # Concurrency control
        self._concurrency_limit = concurrency_limit or cpu_count() * 2
        self._concurrency = asyncio.Semaphore(concurrency_limit or cpu_count() * 2)

        # Monitoring task
        self._monitoring_task: Optional[Task[Any]] = None

        # Initialize step history accumulator
        self._step_history_so_far: list[StepResult] = []

    async def execute(
        self,
        frame: Optional[ExecutionFrame[TContext_w_Scratch]] = None,
        *,
        step: Optional[Any] = None,
        data: Optional[Any] = None,
        context: Optional[TContext_w_Scratch] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        result: Optional[StepResult] = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        # Handle both ExecutionFrame and keyword arguments for backward compatibility
        if frame is not None:
            # Extract parameters from the ExecutionFrame
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            stream = frame.stream
            on_chunk = frame.on_chunk
            breach_event = frame.breach_event
            context_setter = frame.context_setter
            result = frame.result
            _fallback_depth = frame._fallback_depth
        elif step is None:
            raise ValueError("Either frame or step must be provided")
        
        print(f"🔍 ExecutorCore.execute called with step: {step}")
        print(f"🔍 Step type: {type(step)}")
        print(f"🔍 Step name: {getattr(step, 'name', 'unknown')}")
        print(f"🔍 Data: {data}")
        telemetry.logfire.debug(
            f"ExecutorCore.execute called for step: {step.name if hasattr(step, 'name') else type(step)}"
        )
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step is ParallelStep: {hasattr(step, 'branches')}")
        """Execute a step with the given data and context."""

        telemetry.logfire.debug("=== EXECUTOR CORE EXECUTE ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(
            f"ExecutorCore.execute called with breach_event: {breach_event is not None}"
        )
        telemetry.logfire.debug(f"ExecutorCore.execute called with limits: {limits}")

        # Generate cache key if caching is enabled
        cache_key = None
        if self._cache_backend is not None and self._enable_cache:
            cache_key = self._cache_key_generator.generate_key(step, data, context, resources)
            telemetry.logfire.debug(f"Generated cache key: {cache_key}")

            # Check cache first
            cached_result = await self._cache_backend.get(cache_key)
            if cached_result is not None:
                telemetry.logfire.debug(f"Cache hit for step: {step.name}")
                # Ensure metadata_ is always a dict
                if cached_result.metadata_ is None:
                    cached_result.metadata_ = {}
                cached_result.metadata_["cache_hit"] = True
                return cached_result
            else:
                telemetry.logfire.debug(f"Cache miss for step: {step.name}")
        else:
            telemetry.logfire.debug(f"Caching disabled for step: {step.name}")

        # Check if this is a complex step that needs special handling
        is_complex = self._is_complex_step(step)
        print(f"🔍 Step {step.name} is_complex: {is_complex}")
        
        if is_complex:
            print(f"🔍 Executing complex step: {step.name}")
            telemetry.logfire.debug(f"Complex step detected: {step.name}")
            return await self._execute_complex_step(
                step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                breach_event,
                context_setter,
                cache_key,
                _fallback_depth,
            )

        print(f"🔍 Executing agent step: {step.name}")
        telemetry.logfire.debug(f"Agent step detected: {step.name}")
        return await self._execute_simple_step(
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
            _fallback_depth,
        )

    def _is_complex_step(self, step: Any) -> bool:
        """Check if step needs complex handling using an object-oriented approach.

        This method uses the `is_complex` property to determine step complexity,
        following Flujo's architectural principles of algebraic closure and
        the Open-Closed Principle. Every step type is a first-class citizen
        in the execution graph, enabling extensibility without core changes.

        The method maintains backward compatibility by preserving existing logic
        for validation steps and plugin steps that don't implement the `is_complex`
        property.

        Args:
            step: The step to check for complexity

        Returns:
            True if the step requires complex handling, False otherwise
        """
        telemetry.logfire.debug("=== IS COMPLEX STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")

        # Use the is_complex property if available (object-oriented approach)
        if getattr(step, "is_complex", False):
            telemetry.logfire.debug(f"Complex step detected via is_complex property: {step.name}")
            return True

        # Check for validation steps (maintain existing logic for backward compatibility)
        if hasattr(step, "meta") and step.meta and step.meta.get("is_validation_step", False):
            telemetry.logfire.debug(f"Validation step detected: {step.name}")
            return True

        # Check for steps with plugins (maintain existing logic for backward compatibility)
        # BUT: If the step has a fallback, treat it as simple to allow fallback logic
        if hasattr(step, "plugins") and step.plugins:
            if hasattr(step, "fallback_step") and step.fallback_step is not None:
                telemetry.logfire.debug(f"Step with plugins and fallback detected: {step.name} - treating as simple")
                return False
            else:
                telemetry.logfire.debug(f"Step with plugins detected: {step.name}")
                return True

        telemetry.logfire.debug(f"Simple step detected: {step.name}")
        return False

    async def _execute_simple_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        telemetry.logfire.debug(f"_execute_simple_step called with step type: {type(step)}, limits: {limits}")
        telemetry.logfire.debug(f"_execute_simple_step step name: {step.name}")
        telemetry.logfire.debug(f"_execute_simple_step is LoopStep: {isinstance(step, LoopStep)}")
        """Execute a simple step with fallback support."""
        # Try to execute the primary step
        primary_result = None
        try:
            primary_result = await self._execute_agent_step(
                step=step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                cache_key=cache_key,
                breach_event=breach_event,
                _fallback_depth=_fallback_depth,
            )
            
            # If primary step succeeded, return the result
            if primary_result.success:
                return primary_result
                
        except Exception as e:
            # Primary step failed with exception
            primary_result = StepResult(
                name=step.name,
                output=None,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=f"Step execution failed: {str(e)}",
                branch_context=None,
                metadata_={},
            )
        
        # Primary step failed, check if we have a fallback
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            # Check for infinite fallback loops
            if _fallback_depth >= self._MAX_FALLBACK_CHAIN_LENGTH:
                raise InfiniteFallbackError(f"Fallback chain too long for step '{step.name}'")
            
            # Execute fallback step
            telemetry.logfire.debug(f"Executing fallback for step '{step.name}'")
            fallback_result = await self._execute_simple_step(
                step=step.fallback_step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                cache_key=cache_key,
                breach_event=breach_event,
                _fallback_depth=_fallback_depth + 1,
            )
            
            # Combine results: use fallback's output and success, but preserve original step name
            # and accumulate metrics from both steps
            combined_result = StepResult(
                name=step.name,  # Preserve original step name
                output=fallback_result.output,
                success=fallback_result.success,
                attempts=primary_result.attempts + fallback_result.attempts,
                latency_s=primary_result.latency_s + fallback_result.latency_s,
                token_counts=primary_result.token_counts + fallback_result.token_counts,
                cost_usd=primary_result.cost_usd + fallback_result.cost_usd,
                feedback=fallback_result.feedback,  # Use fallback's feedback
                branch_context=fallback_result.branch_context,
                metadata_=fallback_result.metadata_.copy(),
                step_history=fallback_result.step_history,
            )
            
            # Add fallback metadata
            combined_result.metadata_["fallback_triggered"] = True
            
            # Combine feedback from both steps if both failed
            if not primary_result.success and not fallback_result.success:
                primary_feedback = primary_result.feedback or ""
                fallback_feedback = fallback_result.feedback or ""
                if primary_feedback and fallback_feedback:
                    combined_result.feedback = f"{primary_feedback}; {fallback_feedback}"
                elif primary_feedback:
                    combined_result.feedback = primary_feedback
                elif fallback_feedback:
                    combined_result.feedback = fallback_feedback
            elif fallback_result.success:
                # Clear feedback on successful fallback
                combined_result.feedback = None
                # Store original error in metadata
                if primary_result.feedback:
                    combined_result.metadata_["original_error"] = primary_result.feedback
            
            return combined_result
        
        # No fallback available, return the failed primary result
        return primary_result

    async def _execute_agent_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """Execute an agent step with comprehensive retry logic and post-processing."""
        import time
        from ...exceptions import MissingAgentError

        # Initialize metadata_ variable to fix the critical issue
        metadata_: Dict[str, Any] = {}
        
        telemetry.logfire.debug("=== EXECUTE AGENT STEP ===")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Data: {data}")
        telemetry.logfire.debug(f"Context: {context}")
        telemetry.logfire.debug(f"Resources: {resources}")
        telemetry.logfire.debug(f"Stream: {stream}")
        telemetry.logfire.debug(f"Fallback depth: {_fallback_depth}")

        # Check for missing agent
        agent = getattr(step, "agent", None)
        if agent is None:
            raise MissingAgentError(f"Step '{step.name}' has no agent")

        # Initialize result
        result = StepResult(
            name=step.name,
            output=None,
            success=False,
            attempts=0,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",
            branch_context=None,
            metadata_=metadata_,
        )

        # Get step configuration
        max_retries = getattr(step, "max_retries", 1)
        if hasattr(step, "config") and step.config:
            max_retries = getattr(step.config, "max_retries", max_retries)

        # Track attempts and timing
        attempts = 0
        start_time = time.monotonic()
        accumulated_feedback = []

        # Retry loop
        while attempts <= max_retries:
            attempts += 1
            result.attempts = attempts
            
            telemetry.logfire.debug(f"Attempt {attempts}/{max_retries + 1} for step: {step.name}")

            try:
                # Prepare agent options
                options = {}
                if hasattr(step, "config") and step.config:
                    for attr in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                        if hasattr(step.config, attr):
                            options[attr] = getattr(step.config, attr)

                # Execute agent using the agent runner
                output = await self._agent_runner.run(
                    agent=agent,
                    payload=data,
                    context=context,
                    resources=resources,
                    options=options,
                    stream=stream,
                    on_chunk=on_chunk,
                    breach_event=breach_event,
                )

                # Success - set result
                result.output = output
                result.success = True
                result.latency_s = time.monotonic() - start_time
                result.feedback = ""
                
                # Extract cost and token information from the output
                from ...cost import extract_usage_metrics
                prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                    raw_output=output, agent=agent, step_name=step.name
                )
                result.cost_usd = cost_usd
                result.token_counts = prompt_tokens + completion_tokens

                # Apply processors if available
                if hasattr(step, "processors") and step.processors:
                    result.output = await self._processor_pipeline.apply_output(
                        step.processors, result.output, context=context
                    )

                # Apply validators if available
                if hasattr(step, "validators") and step.validators:
                    try:
                        await self._validator_runner.validate(
                            step.validators, result.output, context=context
                        )
                    except ValueError as validation_error:
                        # Check if this is a non-strict validation step
                        from ...application.context_manager import _get_validation_flags, _apply_validation_metadata
                        is_validation_step, is_strict = _get_validation_flags(step)
                        
                        if is_validation_step and not is_strict:
                            # Non-strict validation: step succeeds but record failure in metadata
                            _apply_validation_metadata(
                                result,
                                validation_failed=True,
                                is_validation_step=is_validation_step,
                                is_strict=is_strict,
                            )
                            telemetry.logfire.debug(f"Step '{step.name}' validation failed (non-strict): {validation_error}")
                            # Continue with success - don't return here
                        else:
                            # Strict validation or regular step: fail the step
                            result.success = False
                            result.feedback = f"Validation failed: {validation_error}"
                            if is_validation_step and is_strict:
                                # For strict validation steps, drop the output
                                result.output = None
                            else:
                                # For regular steps, keep the output for fallback
                                result.output = output
                            result.latency_s = time.monotonic() - start_time
                            telemetry.logfire.debug(f"Step '{step.name}' failed validation: {validation_error}")
                            return result

                # Apply plugins if available
                if hasattr(step, "plugins") and step.plugins:
                    try:
                        result.output = await self._plugin_runner.run_plugins(
                            step.plugins, result.output, context=context
                        )
                    except Exception as plugin_error:
                        # Plugin failure - DO NOT RETRY AGENT
                        result.success = False
                        result.feedback = f"Plugin failed: {plugin_error}"
                        result.output = output  # Keep the output for fallback
                        result.latency_s = time.monotonic() - start_time
                        telemetry.logfire.debug(f"Step '{step.name}' plugin failed: {plugin_error}")
                        return result

                # Cache successful result
                if self._cache_backend is not None and cache_key is not None:
                    if result.metadata_ is None:
                        result.metadata_ = {}
                    await self._cache_backend.put(cache_key, result, ttl_s=3600)

                telemetry.logfire.debug(f"Step '{step.name}' completed successfully")
                return result

            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                telemetry.logfire.error(f"Step '{step.name}' failed with critical error: {e}")
                raise

            except Exception as e:
                # Handle retryable errors
                error_msg = f"Attempt {attempts} failed: {str(e)}"
                accumulated_feedback.append(error_msg)
                telemetry.logfire.warning(f"Step '{step.name}' attempt {attempts} failed: {e}")

                # Check if we should retry
                if attempts <= max_retries:
                    # Clone payload for retry with accumulated feedback
                    data = self._clone_payload_for_retry(data, accumulated_feedback)
                    continue
                else:
                    # Max retries exceeded
                    result.success = False
                    result.feedback = f"Step execution failed: {str(e)}"
                    result.output = None
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' failed after {attempts} attempts")
                    return result

        # This should never be reached, but just in case
        result.success = False
        result.feedback = "Step execution failed: unexpected error"
        result.output = None
        result.latency_s = time.monotonic() - start_time
        return result

    def _clone_payload_for_retry(self, original_data: Any, accumulated_feedbacks: list[str]) -> Any:
        """Clone payload for retry attempts with accumulated feedback injection.
        
        This method creates a safe copy of the original payload and injects accumulated
        feedback from previous failed attempts. It handles various data types:
        
        - Dict objects: Creates a shallow copy and appends feedback to 'feedback' key
        - Pydantic models: Uses model_copy() for efficient copying and appends to feedback field
        - Dataclasses: Creates a copy using copy.deepcopy() and appends to feedback field
        - Other types: Converts to string and appends feedback as newlines
        
        Args:
            original_data: The original payload to clone
            accumulated_feedbacks: List of error messages from previous attempts
            
        Returns:
            A cloned payload with accumulated feedback injected
        """
        if not accumulated_feedbacks:
            # No feedback to add, return original data unchanged
            return original_data

        feedback_text = "\n".join(accumulated_feedbacks)

        # Handle dict objects (most common case)
        if isinstance(original_data, dict):
            cloned_data = original_data.copy()
            existing_feedback = cloned_data.get("feedback", "")
            cloned_data["feedback"] = (existing_feedback + "\n" + feedback_text).strip()
            return cloned_data
            
        # Handle Pydantic models with efficient model_copy
        elif hasattr(original_data, "model_copy"):
            cloned_data = original_data.model_copy(deep=False)
            if hasattr(cloned_data, "feedback"):
                existing_feedback = getattr(cloned_data, "feedback", "")
                setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
            return cloned_data
            
        # Handle dataclasses and other objects with copy support
        elif hasattr(original_data, "__dict__"):
            import copy
            try:
                cloned_data = copy.deepcopy(original_data)
                if hasattr(cloned_data, "feedback"):
                    existing_feedback = getattr(cloned_data, "feedback", "")
                    setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
                return cloned_data
            except (TypeError, RecursionError):
                # Fall back to string conversion if deep copy fails
                pass
                
        # Handle list/tuple types
        elif isinstance(original_data, (list, tuple)):
            try:
                cloned_data = original_data.copy() if isinstance(original_data, list) else list(original_data)
                # Try to add feedback to the first element if it's a dict
                if cloned_data and isinstance(cloned_data[0], dict):
                    cloned_data[0]["feedback"] = cloned_data[0].get("feedback", "") + "\n" + feedback_text
                return cloned_data
            except (AttributeError, TypeError):
                pass
                
        # Fallback: convert to string and append feedback
        return f"{str(original_data)}\n{feedback_text}"


class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply prompt processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with prompt_processors attribute
        processor_list = processors
        if hasattr(processors, "prompt_processors"):
            processor_list = processors.prompt_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Log error but continue with original data
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Prompt processor failed: {e}")
                except Exception:
                    pass
                processed_data = data

        return processed_data

    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        """Apply output processors sequentially."""
        import inspect

        if not processors:
            return data

        # Handle both list of processors and object with output_processors attribute
        processor_list = processors
        if hasattr(processors, "output_processors"):
            processor_list = processors.output_processors

        if not processor_list:
            return data

        processed_data = data
        for proc in processor_list:
            try:
                fn = getattr(proc, "process", proc)
                if inspect.iscoroutinefunction(fn):
                    try:
                        processed_data = await fn(processed_data, context=context)
                    except TypeError:
                        processed_data = await fn(processed_data)
                else:
                    try:
                        processed_data = fn(processed_data, context=context)
                    except TypeError:
                        processed_data = fn(processed_data)
            except Exception as e:
                # Re-raise processor exceptions to cause step failure
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                # Re-raise the exception to cause step failure
                raise

        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(self, validators: List[Any], data: Any, *, context: Any) -> None:
        """Run validators and raise ValueError on first failure."""
        if not validators:
            return

        for validator in validators:
            try:
                result = await validator.validate(data, context=context)
                if isinstance(result, ValidationResult) and not result.is_valid:
                    # Use feedback field instead of message
                    feedback = result.feedback or "Validation failed"
                    raise ValueError(f"Validation failed: {feedback}")
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                raise ValueError(f"Validator {type(validator).__name__} failed: {e}")


class DefaultPluginRunner:
    """Default plugin runner implementation."""

    async def run_plugins(self, plugins: List[tuple[Any, int]], data: Any, *, context: Any) -> Any:
        """Run plugins in priority order."""
        processed_data = data

        if not plugins:
            return processed_data

        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                from ...domain.plugins import PluginOutcome

                # Call plugin with proper parameter filtering
                plugin_kwargs = {}
                if hasattr(plugin, "validate"):
                    func = plugin.validate
                else:
                    func = plugin

                # Check if plugin accepts context using conservative logic
                # Create a temporary ExecutorCore instance to access the method
                temp_executor: ExecutorCore[Any] = ExecutorCore()
                if temp_executor._should_pass_context_to_plugin(context, func):
                    plugin_kwargs["context"] = context

                # Call plugin
                result = await func(processed_data, **plugin_kwargs)

                # Handle PluginOutcome
                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # NEW: Raise an exception to fail the step
                        raise ValueError(f"Plugin validation failed: {result.feedback}")
                    if result.new_solution is not None:
                        processed_data = result.new_solution
                else:
                    # Plugin returned new data
                    processed_data = result

            except Exception as e:
                # Log error and re-raise to cause step failure (explicit plugin failure handling)
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Plugin {type(plugin).__name__} failed: {e}")
                except Exception:
                    pass
                # Re-raise the exception to cause step failure
                raise e

        return processed_data


class DefaultTelemetry:
    """Default telemetry implementation."""

    def __init__(self) -> None:
        try:
            from ...infra import telemetry

            self._telemetry = telemetry
            self._available = True
        except Exception:
            self._available = False

    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if self._available:
                try:
                    return self._telemetry.logfire.instrument(name)(func)
                except Exception:
                    pass
            return func

        return decorator


# --------------------------------------------------------------------------- #
# ★ Cache Key Generation
# --------------------------------------------------------------------------- #


@dataclass
class CacheKeyComponents:
    """Components for deterministic cache key generation."""

    step_name: str
    step_type: str
    payload_digest: str
    context_digest: Optional[str]
    resource_digest: Optional[str]
    agent_id: Optional[str]


class CacheKeyGenerator:
    """Generates deterministic cache keys from step execution parameters."""

    def __init__(self, serializer: ISerializer, hasher: IHasher):
        self._serializer = serializer
        self._hasher = hasher
        self._seen_hashes: WeakKeyDictionary[Any, str] = WeakKeyDictionary()

    def generate_key(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
    ) -> str:
        """Generate a deterministic cache key."""
        step_name = getattr(step, "name", None)
        if step_name is None:
            # Fallback for objects (like Pipeline) that lack a stable name attribute
            step_name = f"{type(step).__name__}:{id(step)}"

        components = CacheKeyComponents(
            step_name=step_name,
            step_type=type(step).__name__,
            payload_digest=self._hash_obj(data),
            context_digest=self._hash_obj(context) if context is not None else None,
            resource_digest=self._hash_obj(resources) if resources is not None else None,
            agent_id=self._get_agent_id(step),
        )

        # Serialize components for final hash
        serialized = self._serializer.serialize(
            {
                "step_name": components.step_name,
                "step_type": components.step_type,
                "payload_digest": components.payload_digest,
                "context_digest": components.context_digest,
                "resource_digest": components.resource_digest,
                "agent_id": components.agent_id,
            }
        )

        return self._hasher.digest(serialized)

    def _hash_obj(self, obj: Any) -> str:
        """Hash any Python object deterministically."""
        # Fast path for common types
        if obj is None:
            return "null"
        if isinstance(obj, (str, int, float, bool)):
            return self._hasher.digest(str(obj).encode())
        if isinstance(obj, bytes):
            return self._hasher.digest(obj)

        # Check cache first
        try:
            if obj in self._seen_hashes:
                return self._seen_hashes[obj]
        except (TypeError, KeyError):
            pass

        # Handle BaseModel efficiently
        if isinstance(obj, BaseModel):
            try:
                serialized = obj.model_dump_json(sort_keys=True).encode()
                h = self._hasher.digest(serialized)
            except Exception:
                try:
                    data = obj.model_dump()
                    serialized_data = self._serialize_for_hash(data)
                    h = self._hasher.digest(self._serializer.serialize(serialized_data))
                except Exception:
                    h = self._hasher.digest(repr(obj).encode())
        else:
            try:
                h = self._hasher.digest(self._serializer.serialize(obj))
            except (TypeError, ValueError):
                try:
                    serialized = self._serialize_for_hash(obj)
                    h = self._hasher.digest(self._serializer.serialize(serialized))
                except Exception:
                    h = self._hasher.digest(repr(obj).encode())

        # Cache result
        try:
            self._seen_hashes[obj] = h
        except (TypeError, KeyError):
            pass

        return h

    def _serialize_for_hash(self, obj: Any) -> Any:
        """Serialize object for hashing, handling special cases."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_hash(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_hash(item) for item in obj]
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__class__") and "Mock" in obj.__class__.__name__:
            return f"Mock({type(obj).__name__})"
        elif hasattr(obj, "__dict__"):
            try:
                return {k: self._serialize_for_hash(v) for k, v in obj.__dict__.items()}
            except Exception:
                return repr(obj)
        else:
            return obj

    def _get_agent_id(self, step: Any) -> Optional[str]:
        """Get stable agent identifier."""
        agent = getattr(step, "agent", None)
        if agent is None:
            return None

        # Use agent type and configuration for stable identification
        agent_type = f"{type(agent).__module__}.{type(agent).__name__}"
        agent_config = getattr(agent, "config", None)

        if agent_config:
            config_hash = self._hash_obj(agent_config)
            return f"{agent_type}:{config_hash}"
        else:
            return agent_type


# --------------------------------------------------------------------------- #
# ★ Internal Types
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ★ ExecutorCore (Main Implementation)
# --------------------------------------------------------------------------- #




class ExecutorCore(Generic[TContext_w_Scratch]):
    """
    Ultra-optimized step executor with modular, policy-driven architecture.

    This maintains the exact same API as the original UltraStepExecutor
    while providing enhanced performance, reliability, and extensibility.
    """

    # Context variables for tracking fallback relationships and chains
    _fallback_relationships: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
        "fallback_relationships", default={}
    )
    _fallback_chain: contextvars.ContextVar[list[Step[Any, Any]]] = contextvars.ContextVar(
        "fallback_chain", default=[]
    )

    # Cache for fallback relationship loop detection (True if loop detected, False otherwise)
    _fallback_graph_cache: contextvars.ContextVar[Dict[str, bool]] = contextvars.ContextVar(
        "fallback_graph_cache", default={}
    )

    # Maximum length for fallback chains to prevent infinite loops
    _MAX_FALLBACK_CHAIN_LENGTH = 10

    # Maximum iterations for fallback loop detection to prevent infinite loops
    _DEFAULT_MAX_FALLBACK_ITERATIONS = 100

    class _ParallelUsageGovernor:
        """Helper to track and enforce usage limits atomically across parallel branches."""

        def __init__(self, usage_limits: Optional[UsageLimits]) -> None:
            self.usage_limits = usage_limits
            self.lock = asyncio.Lock()
            self.total_cost = 0.0
            self.total_tokens = 0
            self.limit_breached = asyncio.Event()
            self.limit_breach_error: Optional[UsageLimitExceededError] = None
            self._breach_detected = False  # Add explicit flag to prevent race conditions

        def _create_breach_error_message(
            self,
            limit_type: str,
            limit_value: Any,
            current_value: Any,
        ) -> str:
            """Create a breach error message string."""
            if limit_type == "cost":
                # Format cost values to remove trailing zeros (e.g., 0.5 instead of 0.50)
                formatted_limit = format_cost(limit_value)
                formatted_current = format_cost(current_value)
                return f"Cost limit of ${formatted_limit} exceeded. Current cost: ${formatted_current}"
            else:  # token
                return f"Token limit of {limit_value} exceeded. Current tokens: {current_value}"

        async def add_usage(self, cost_delta: float, token_delta: int, result: StepResult) -> bool:
            """Add usage and check for breach. Returns True if breach occurred."""
            try:
                # Add timeout to prevent infinite lock waits
                async with asyncio.timeout(5.0):  # 5 second timeout
                    async with self.lock:
                        # Always update totals for accurate accounting, even after breach
                        self.total_cost += cost_delta
                        self.total_tokens += token_delta

                        # Check for breach only if limits are configured
                        if self.usage_limits is not None:
                            breach_occurred = False

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
                                breach_occurred = True
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
                                breach_occurred = True

                            if breach_occurred:
                                self._breach_detected = True
                                self.limit_breached.set()

                        return self._breach_detected
            except asyncio.TimeoutError:
                # If we can't acquire the lock within timeout, assume breach to be safe
                return True

        def breached(self) -> bool:
            """Check if a limit has been breached."""
            return self._breach_detected or self.limit_breached.is_set()

        def get_error_message(self) -> Optional[str]:
            """Get the error message if a breach occurred."""
            if self.limit_breach_error:
                return self.limit_breach_error.args[0]
            return None

        def get_error(self) -> Optional[UsageLimitExceededError]:
            """Get the error if a breach occurred."""
            return self.limit_breach_error

    def _should_pass_context(
        self,
        spec: Any,
        context: Optional[Any],
        func: Callable[..., Any],
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

    def _should_pass_context_to_plugin(
        self,
        context: Optional[Any],
        func: Callable[..., Any],
    ) -> bool:
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

    def _default_set_final_context(
        self,
        result: "PipelineResult[Any]",
        ctx: Optional[Any],
    ) -> None:
        """Default function to set final context from pipeline result."""
        if ctx is not None and result.final_pipeline_context is not None:
            for (
                field_name,
                field_value,
            ) in result.final_pipeline_context.model_dump().items():
                setattr(ctx, field_name, field_value)

    def _manage_fallback_relationships(
        self,
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
        self.__class__._fallback_graph_cache.set({})

        relationships = self.__class__._fallback_relationships.get()
        relationships_token = self.__class__._fallback_relationships.set(
            {**relationships, step.name: step.fallback_step.name}
        )
        return relationships_token

    def _detect_fallback_loop(self, step: Step[Any, Any], chain: list[Step[Any, Any]]) -> bool:
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
        if len(chain) >= self._MAX_FALLBACK_CHAIN_LENGTH:
            return True

        # Strategy 4: Global relationship loop detection with caching
        relationships = self.__class__._fallback_relationships.get()
        if step.name in relationships:
            # Use cached graph for improved performance
            graph_cache = self.__class__._fallback_graph_cache.get()

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
                max_iterations = self._DEFAULT_MAX_FALLBACK_ITERATIONS
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

    def __init__(
        self,
        *,
        serializer: Optional[ISerializer] = None,
        hasher: Optional[IHasher] = None,
        cache_backend: Optional[ICacheBackend] = None,
        usage_meter: Optional[IUsageMeter] = None,
        agent_runner: Optional[IAgentRunner] = None,
        processor_pipeline: Optional[IProcessorPipeline] = None,
        validator_runner: Optional[IValidatorRunner] = None,
        plugin_runner: Optional[IPluginRunner] = None,
        telemetry: Optional[ITelemetry] = None,
        concurrency_limit: Optional[int] = None,
        enable_cache: bool = True,
    ) -> None:
        # Initialize components with defaults
        self._serializer = serializer or OrjsonSerializer()
        self._hasher = hasher or Blake3Hasher()
        self._cache_backend = cache_backend if enable_cache else None
        self._usage_meter = usage_meter or ThreadSafeMeter()
        self._agent_runner = agent_runner or DefaultAgentRunner()
        self._processor_pipeline = processor_pipeline or DefaultProcessorPipeline()
        self._validator_runner = validator_runner or DefaultValidatorRunner()
        self._plugin_runner = plugin_runner or DefaultPluginRunner()
        self._telemetry = telemetry or DefaultTelemetry()
        self._enable_cache = enable_cache

        # Initialize cache key generator
        self._cache_key_generator = CacheKeyGenerator(self._serializer, self._hasher)

        # Concurrency control
        self._concurrency_limit = concurrency_limit or cpu_count() * 2
        self._concurrency = asyncio.Semaphore(concurrency_limit or cpu_count() * 2)

        # Monitoring task
        self._monitoring_task: Optional[Task[Any]] = None

        # Initialize step history accumulator
        self._step_history_so_far: list[StepResult] = []

    async def execute(
        self,
        frame: Optional[ExecutionFrame[TContext_w_Scratch]] = None,
        *,
        step: Optional[Any] = None,
        data: Optional[Any] = None,
        context: Optional[TContext_w_Scratch] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        result: Optional[StepResult] = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        # Handle both ExecutionFrame and keyword arguments for backward compatibility
        if frame is not None:
            # Extract parameters from the ExecutionFrame
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            stream = frame.stream
            on_chunk = frame.on_chunk
            breach_event = frame.breach_event
            context_setter = frame.context_setter
            result = frame.result
            _fallback_depth = frame._fallback_depth
        elif step is None:
            raise ValueError("Either frame or step must be provided")
        
        print(f"🔍 ExecutorCore.execute called with step: {step}")
        print(f"🔍 Step type: {type(step)}")
        print(f"🔍 Step name: {getattr(step, 'name', 'unknown')}")
        print(f"🔍 Data: {data}")
        telemetry.logfire.debug(
            f"ExecutorCore.execute called for step: {step.name if hasattr(step, 'name') else type(step)}"
        )
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step is ParallelStep: {hasattr(step, 'branches')}")
        """Execute a step with the given data and context."""

        telemetry.logfire.debug("=== EXECUTOR CORE EXECUTE ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(
            f"ExecutorCore.execute called with breach_event: {breach_event is not None}"
        )
        telemetry.logfire.debug(f"ExecutorCore.execute called with limits: {limits}")

        # Generate cache key if caching is enabled
        cache_key = None
        if self._cache_backend is not None and self._enable_cache:
            cache_key = self._cache_key_generator.generate_key(step, data, context, resources)
            telemetry.logfire.debug(f"Generated cache key: {cache_key}")

            # Check cache first
            cached_result = await self._cache_backend.get(cache_key)
            if cached_result is not None:
                telemetry.logfire.debug(f"Cache hit for step: {step.name}")
                # Ensure metadata_ is always a dict
                if cached_result.metadata_ is None:
                    cached_result.metadata_ = {}
                cached_result.metadata_["cache_hit"] = True
                return cached_result
            else:
                telemetry.logfire.debug(f"Cache miss for step: {step.name}")
        else:
            telemetry.logfire.debug(f"Caching disabled for step: {step.name}")

        # Check if this is a complex step that needs special handling
        is_complex = self._is_complex_step(step)
        print(f"🔍 Step {step.name} is_complex: {is_complex}")
        
        if is_complex:
            print(f"🔍 Executing complex step: {step.name}")
            telemetry.logfire.debug(f"Complex step detected: {step.name}")
            return await self._execute_complex_step(
                step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                breach_event,
                context_setter,
                cache_key,
                _fallback_depth,
            )

        print(f"🔍 Executing agent step: {step.name}")
        telemetry.logfire.debug(f"Agent step detected: {step.name}")
        return await self._execute_simple_step(
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
            _fallback_depth,
        )

    def _is_complex_step(self, step: Any) -> bool:
        """Check if step needs complex handling using an object-oriented approach.

        This method uses the `is_complex` property to determine step complexity,
        following Flujo's architectural principles of algebraic closure and
        the Open-Closed Principle. Every step type is a first-class citizen
        in the execution graph, enabling extensibility without core changes.

        The method maintains backward compatibility by preserving existing logic
        for validation steps and plugin steps that don't implement the `is_complex`
        property.

        Args:
            step: The step to check for complexity

        Returns:
            True if the step requires complex handling, False otherwise
        """
        telemetry.logfire.debug("=== IS COMPLEX STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")

        # Use the is_complex property if available (object-oriented approach)
        if getattr(step, "is_complex", False):
            telemetry.logfire.debug(f"Complex step detected via is_complex property: {step.name}")
            return True

        # Check for validation steps (maintain existing logic for backward compatibility)
        if hasattr(step, "meta") and step.meta and step.meta.get("is_validation_step", False):
            telemetry.logfire.debug(f"Validation step detected: {step.name}")
            return True

        # Check for steps with plugins (maintain existing logic for backward compatibility)
        # BUT: If the step has a fallback, treat it as simple to allow fallback logic
        if hasattr(step, "plugins") and step.plugins:
            if hasattr(step, "fallback_step") and step.fallback_step is not None:
                telemetry.logfire.debug(f"Step with plugins and fallback detected: {step.name} - treating as simple")
                return False
            else:
                telemetry.logfire.debug(f"Step with plugins detected: {step.name}")
                return True

        telemetry.logfire.debug(f"Simple step detected: {step.name}")
        return False

    async def _execute_simple_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        telemetry.logfire.debug(f"_execute_simple_step called with step type: {type(step)}, limits: {limits}")
        telemetry.logfire.debug(f"_execute_simple_step step name: {step.name}")
        telemetry.logfire.debug(f"_execute_simple_step is LoopStep: {isinstance(step, LoopStep)}")
        """Execute a simple step with fallback support."""
        # Try to execute the primary step
        primary_result = None
        try:
            primary_result = await self._execute_agent_step(
                step=step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                cache_key=cache_key,
                breach_event=breach_event,
                _fallback_depth=_fallback_depth,
            )
            
            # If primary step succeeded, return the result
            if primary_result.success:
                return primary_result
                
        except Exception as e:
            # Primary step failed with exception
            primary_result = StepResult(
                name=step.name,
                output=None,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=f"Step execution failed: {str(e)}",
                branch_context=None,
                metadata_={},
            )
        
        # Primary step failed, check if we have a fallback
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            # Check for infinite fallback loops
            if _fallback_depth >= self._MAX_FALLBACK_CHAIN_LENGTH:
                raise InfiniteFallbackError(f"Fallback chain too long for step '{step.name}'")
            
            # Execute fallback step
            telemetry.logfire.debug(f"Executing fallback for step '{step.name}'")
            fallback_result = await self._execute_simple_step(
                step=step.fallback_step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                cache_key=cache_key,
                breach_event=breach_event,
                _fallback_depth=_fallback_depth + 1,
            )
            
            # Combine results: use fallback's output and success, but preserve original step name
            # and accumulate metrics from both steps
            combined_result = StepResult(
                name=step.name,  # Preserve original step name
                output=fallback_result.output,
                success=fallback_result.success,
                attempts=primary_result.attempts + fallback_result.attempts,
                latency_s=primary_result.latency_s + fallback_result.latency_s,
                token_counts=primary_result.token_counts + fallback_result.token_counts,
                cost_usd=primary_result.cost_usd + fallback_result.cost_usd,
                feedback=fallback_result.feedback,  # Use fallback's feedback
                branch_context=fallback_result.branch_context,
                metadata_=fallback_result.metadata_.copy(),
                step_history=fallback_result.step_history,
            )
            
            # Add fallback metadata
            combined_result.metadata_["fallback_triggered"] = True
            
            # Combine feedback from both steps if both failed
            if not primary_result.success and not fallback_result.success:
                primary_feedback = primary_result.feedback or ""
                fallback_feedback = fallback_result.feedback or ""
                if primary_feedback and fallback_feedback:
                    combined_result.feedback = f"{primary_feedback}; {fallback_feedback}"
                elif primary_feedback:
                    combined_result.feedback = primary_feedback
                elif fallback_feedback:
                    combined_result.feedback = fallback_feedback
            elif fallback_result.success:
                # Clear feedback on successful fallback
                combined_result.feedback = None
                # Store original error in metadata
                if primary_result.feedback:
                    combined_result.metadata_["original_error"] = primary_result.feedback
            
            return combined_result
        
        # No fallback available, return the failed primary result
        return primary_result

    async def _execute_agent_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """Execute an agent step with comprehensive retry logic and post-processing."""
        import time
        from ...exceptions import MissingAgentError

        # Initialize metadata_ variable to fix the critical issue
        metadata_: Dict[str, Any] = {}
        
        telemetry.logfire.debug("=== EXECUTE AGENT STEP ===")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Data: {data}")
        telemetry.logfire.debug(f"Context: {context}")
        telemetry.logfire.debug(f"Resources: {resources}")
        telemetry.logfire.debug(f"Stream: {stream}")
        telemetry.logfire.debug(f"Fallback depth: {_fallback_depth}")

        # Check for missing agent
        agent = getattr(step, "agent", None)
        if agent is None:
            raise MissingAgentError(f"Step '{step.name}' has no agent")

        # Initialize result
        result = StepResult(
            name=step.name,
            output=None,
            success=False,
            attempts=0,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback="",
            branch_context=None,
            metadata_=metadata_,
        )

        # Get step configuration
        max_retries = getattr(step, "max_retries", 1)
        if hasattr(step, "config") and step.config:
            max_retries = getattr(step.config, "max_retries", max_retries)

        # Track attempts and timing
        attempts = 0
        start_time = time.monotonic()
        accumulated_feedback = []

        # Retry loop
        while attempts <= max_retries:
            attempts += 1
            result.attempts = attempts
            
            telemetry.logfire.debug(f"Attempt {attempts}/{max_retries + 1} for step: {step.name}")

            try:
                # Prepare agent options
                options = {}
                if hasattr(step, "config") and step.config:
                    for attr in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                        if hasattr(step.config, attr):
                            options[attr] = getattr(step.config, attr)

                # Execute agent using the agent runner
                output = await self._agent_runner.run(
                    agent=agent,
                    payload=data,
                    context=context,
                    resources=resources,
                    options=options,
                    stream=stream,
                    on_chunk=on_chunk,
                    breach_event=breach_event,
                )

                # Success - set result
                result.output = output
                result.success = True
                result.latency_s = time.monotonic() - start_time
                result.feedback = ""
                
                # Extract cost and token information from the output
                from ...cost import extract_usage_metrics
                prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                    raw_output=output, agent=agent, step_name=step.name
                )
                result.cost_usd = cost_usd
                result.token_counts = prompt_tokens + completion_tokens

                # Apply processors if available
                if hasattr(step, "processors") and step.processors:
                    result.output = await self._processor_pipeline.apply_output(
                        step.processors, result.output, context=context
                    )

                # Apply validators if available
                if hasattr(step, "validators") and step.validators:
                    try:
                        await self._validator_runner.validate(
                            step.validators, result.output, context=context
                        )
                    except ValueError as validation_error:
                        # Check if this is a non-strict validation step
                        from ...application.context_manager import _get_validation_flags, _apply_validation_metadata
                        is_validation_step, is_strict = _get_validation_flags(step)
                        
                        if is_validation_step and not is_strict:
                            # Non-strict validation: step succeeds but record failure in metadata
                            _apply_validation_metadata(
                                result,
                                validation_failed=True,
                                is_validation_step=is_validation_step,
                                is_strict=is_strict,
                            )
                            telemetry.logfire.debug(f"Step '{step.name}' validation failed (non-strict): {validation_error}")
                            # Continue with success - don't return here
                        else:
                            # Strict validation or regular step: fail the step
                            result.success = False
                            result.feedback = f"Validation failed: {validation_error}"
                            if is_validation_step and is_strict:
                                # For strict validation steps, drop the output
                                result.output = None
                            else:
                                # For regular steps, keep the output for fallback
                                result.output = output
                            result.latency_s = time.monotonic() - start_time
                            telemetry.logfire.debug(f"Step '{step.name}' failed validation: {validation_error}")
                            return result

                # Apply plugins if available
                if hasattr(step, "plugins") and step.plugins:
                    try:
                        result.output = await self._plugin_runner.run_plugins(
                            step.plugins, result.output, context=context
                        )
                    except Exception as plugin_error:
                        # Plugin failure - DO NOT RETRY AGENT
                        result.success = False
                        result.feedback = f"Plugin failed: {plugin_error}"
                        result.output = output  # Keep the output for fallback
                        result.latency_s = time.monotonic() - start_time
                        telemetry.logfire.debug(f"Step '{step.name}' plugin failed: {plugin_error}")
                        return result

                # Cache successful result
                if self._cache_backend is not None and cache_key is not None:
                    if result.metadata_ is None:
                        result.metadata_ = {}
                    await self._cache_backend.put(cache_key, result, ttl_s=3600)

                telemetry.logfire.debug(f"Step '{step.name}' completed successfully")
                return result

            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                telemetry.logfire.error(f"Step '{step.name}' failed with critical error: {e}")
                raise

            except Exception as e:
                # Handle retryable errors
                error_msg = f"Attempt {attempts} failed: {str(e)}"
                accumulated_feedback.append(error_msg)
                telemetry.logfire.warning(f"Step '{step.name}' attempt {attempts} failed: {e}")

                # Check if we should retry
                if attempts <= max_retries:
                    # Clone payload for retry with accumulated feedback
                    data = self._clone_payload_for_retry(data, accumulated_feedback)
                    continue
                else:
                    # Max retries exceeded
                    result.success = False
                    result.feedback = f"Step execution failed: {str(e)}"
                    result.output = None
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' failed after {attempts} attempts")
                    return result

        # This should never be reached, but just in case
        result.success = False
        result.feedback = "Step execution failed: unexpected error"
        result.output = None
        result.latency_s = time.monotonic() - start_time
        return result

    def _clone_payload_for_retry(self, original_data: Any, accumulated_feedbacks: list[str]) -> Any:
        """Clone payload for retry attempts with accumulated feedback injection.
        
        This method creates a safe copy of the original payload and injects accumulated
        feedback from previous failed attempts. It handles various data types:
        
        - Dict objects: Creates a shallow copy and appends feedback to 'feedback' key
        - Pydantic models: Uses model_copy() for efficient copying and appends to feedback field
        - Dataclasses: Creates a copy using copy.deepcopy() and appends to feedback field
        - Other types: Converts to string and appends feedback as newlines
        
        Args:
            original_data: The original payload to clone
            accumulated_feedbacks: List of error messages from previous attempts
            
        Returns:
            A cloned payload with accumulated feedback injected
        """
        if not accumulated_feedbacks:
            # No feedback to add, return original data unchanged
            return original_data

        feedback_text = "\n".join(accumulated_feedbacks)

        # Handle dict objects (most common case)
        if isinstance(original_data, dict):
            cloned_data = original_data.copy()
            existing_feedback = cloned_data.get("feedback", "")
            cloned_data["feedback"] = (existing_feedback + "\n" + feedback_text).strip()
            return cloned_data
            
        # Handle Pydantic models with efficient model_copy
        elif hasattr(original_data, "model_copy"):
            cloned_data = original_data.model_copy(deep=False)
            if hasattr(cloned_data, "feedback"):
                existing_feedback = getattr(cloned_data, "feedback", "")
                setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
            return cloned_data
            
        # Handle dataclasses and other objects with copy support
        elif hasattr(original_data, "__dict__"):
            import copy
            try:
                cloned_data = copy.deepcopy(original_data)
                if hasattr(cloned_data, "feedback"):
                    existing_feedback = getattr(cloned_data, "feedback", "")
                    setattr(cloned_data, "feedback", (existing_feedback + "\n" + feedback_text).strip())
                return cloned_data
            except (TypeError, RecursionError):
                # Fall back to string conversion if deep copy fails
                pass
                
        # Handle list/tuple types
        elif isinstance(original_data, (list, tuple)):
            try:
                cloned_data = original_data.copy() if isinstance(original_data, list) else list(original_data)
                # Try to add feedback to the first element if it's a dict
                if cloned_data and isinstance(cloned_data[0], dict):
                    cloned_data[0]["feedback"] = cloned_data[0].get("feedback", "") + "\n" + feedback_text
                return cloned_data
            except (AttributeError, TypeError):
                pass
                
        # Fallback: convert to string and append feedback
        return f"{str(original_data)}\n{feedback_text}"

    async def _execute_complex_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        cache_key: Optional[str] = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        print("ENTERED _execute_complex_step")
        """Execute complex steps using step logic helpers."""


        if context_setter is None:
            context_setter = self._default_set_final_context

        # Create recursive step executor
        async def step_executor(
            s: Any,
            d: Any,
            c: Optional[Any],
            r: Optional[Any],
            breach_event: Optional[Any] = None,
            **extra_kwargs: Any,
        ) -> StepResult:
            """Recursive step executor that forwards execution to :py:meth:`ExecutorCore.execute`.

            Accepts arbitrary keyword arguments for forward-compatibility (e.g. ``usage_limits``,
            ``context_setter``, ``stream``) so that helper utilities can evolve without breaking
            this interface. Unknown kwargs are either forwarded to :py:meth:`execute` when they
            have an equivalent parameter name or ignored when they are only relevant to the
            higher-level logic.
            """
            # Map commonly-used extra kwargs to the modern execute() signature
            _limits = extra_kwargs.get("usage_limits", limits)
            _stream = extra_kwargs.get("stream", stream)
            _on_chunk = extra_kwargs.get("on_chunk", on_chunk)
            _context_setter = extra_kwargs.get("context_setter", context_setter)

            # Create ExecutionFrame for the recursive call
            frame = ExecutionFrame(
                step=s,
                data=d,
                context=c,
                resources=r,
                limits=_limits,
                stream=_stream,
                on_chunk=_on_chunk,
                breach_event=breach_event,
                context_setter=_context_setter,
            )
            return await self.execute(frame)

        # Handle specific step types
        telemetry.logfire.debug("=== EXECUTE COMPLEX STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step is LoopStep: {isinstance(step, LoopStep)}")
        telemetry.logfire.debug(f"LoopStep import: {LoopStep}")
        telemetry.logfire.debug(f"Step is CacheStep: {isinstance(step, CacheStep)}")
        telemetry.logfire.debug(f"Step is ConditionalStep: {isinstance(step, ConditionalStep)}")
        telemetry.logfire.debug(f"Step is DynamicParallelRouterStep: {isinstance(step, DynamicParallelRouterStep)}")
        telemetry.logfire.debug(f"Step is ParallelStep: {isinstance(step, ParallelStep)}")
        telemetry.logfire.debug(f"Step is HITLStep: {isinstance(step, HumanInTheLoopStep)}")
        telemetry.logfire.debug(f"Step is HumanInTheLoopStep: {isinstance(step, HumanInTheLoopStep)}")

        if isinstance(step, CacheStep):
            telemetry.logfire.debug("Handling CacheStep")
            result = await self._handle_cache_step(
                step,
                data,
                context,
                resources,
                limits,
                breach_event,
                context_setter,
                step_executor,
            )
        elif isinstance(step, LoopStep):
            telemetry.logfire.debug("Handling LoopStep")
            telemetry.logfire.debug(f"LoopStep limits: {limits}")
            telemetry.logfire.debug(f"LoopStep usage_meter: {self._usage_meter}")
            telemetry.logfire.debug(f"LoopStep step: {step}")
            result = await self._handle_loop_step(
                step,
                data,
                context,
                resources,
                limits,
                context_setter,
                _fallback_depth,
            )
        elif isinstance(step, ConditionalStep):
            telemetry.logfire.debug("Handling ConditionalStep")
            result = await self._handle_conditional_step(
                step,
                data,
                context,
                resources,
                limits,
                context_setter,
                _fallback_depth,
            )
        elif isinstance(step, DynamicParallelRouterStep):
            telemetry.logfire.debug("Handling DynamicParallelRouterStep")
            result = await self._handle_dynamic_router_step(
                step,
                data,
                context,
                resources,
                limits,
                context_setter,
            )
        elif isinstance(step, ParallelStep):
            telemetry.logfire.debug("Handling ParallelStep")
            telemetry.logfire.debug(
                f"Calling _handle_parallel_step with breach_event: {breach_event is not None}"
            )
            result = await self._handle_parallel_step(
                step,
                data,
                context,
                resources,
                limits,
                breach_event,
                context_setter,
                step_executor,
            )
        elif isinstance(step, HumanInTheLoopStep):
            telemetry.logfire.debug("Handling HumanInTheLoopStep")
            result = await self._handle_hitl_step(
                step,
                data,
                context,
                resources,
                limits,
                breach_event,
                context_setter,
            )
        else:
            telemetry.logfire.debug("Falling back to agent step execution")
            # Fall back to agent step execution using the unified _execute_agent_step method
            try:
                result = await self._execute_agent_step(
                    step=step,
                    data=data,
                    context=context,
                    resources=resources,
                    limits=limits,
                    stream=stream,
                    on_chunk=on_chunk,
                    cache_key=cache_key,
                    breach_event=breach_event,
                )
            except MissingAgentError as e:
                # Handle missing agent error gracefully
                result = StepResult(name=step.name)
                result.success = False
                result.feedback = str(e)
                result.output = None
                telemetry.logfire.warning(f"Step '{step.name}' failed due to missing agent: {e}")
            except (
                InfiniteFallbackError,
                InfiniteRedirectError,
                PausedException,
                RuntimeError,
            ) as e:
                # Let critical exceptions propagate to maintain test expectations
                telemetry.logfire.error(f"Step '{step.name}' failed with critical error: {e}")
                raise
            except Exception as e:
                # Handle other errors gracefully
                result = StepResult(name=step.name)
                result.success = False
                result.feedback = f"Step execution failed: {str(e)}"
                result.output = None
                telemetry.logfire.error(f"Step '{step.name}' failed with error: {e}")

        # Cache successful result for complex steps
        if result.success and self._cache_backend is not None and cache_key is not None:
            if result.metadata_ is None:
                result.metadata_ = {}
            await self._cache_backend.put(cache_key, result, ttl_s=3600)

        return result

    async def _handle_parallel_step(
        self,
        parallel_step: ParallelStep[Any],
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[
            Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
        ] = None,
    ) -> StepResult:
        telemetry.logfire.debug(f"=== HANDLING PARALLEL STEP === {parallel_step.name}")
        telemetry.logfire.debug(f"Parallel step branches: {list(parallel_step.branches.keys())}")
        """Handle ParallelStep execution using simplified, deadlock-free architecture."""

        # Initialize result
        result = StepResult(name=parallel_step.name)
        result.metadata_ = {}

        telemetry.logfire.debug(f"_handle_parallel_step called for step: {parallel_step.name}")

        # Check for empty branches
        if not parallel_step.branches:
            result.success = False
            result.feedback = "Parallel step has no branches to execute"
            result.output = {}
            return result

        # Create usage governor for parallel execution
        usage_governor = self._ParallelUsageGovernor(limits)

        # Create breach event for immediate cancellation signaling
        if breach_event is None and limits is not None:
            breach_event = asyncio.Event()

        # --- Wrap Pipeline branches as Step objects to maintain algebraic closure ---
        wrapped_branches: Dict[str, Step[Any, Any]] = {}
        for key, branch_pipe in parallel_step.branches.items():
            # Check if this is a Pipeline object by importing the Pipeline class
            from ...domain.dsl.pipeline import Pipeline

            is_pipeline = isinstance(branch_pipe, Pipeline)
            if is_pipeline:
                # This is a Pipeline object - wrap it as a Step
                wrapped_branches[key] = _PipelineStepAdapter(
                    pipeline=branch_pipe, name=f"{parallel_step.name}_{key}"
                )
            else:
                # This is already a Step object - use as-is
                wrapped_branches[key] = branch_pipe

        # Use provided step executor or fall back to self.execute
        if step_executor is None:

            async def step_executor(
                s: Any,
                d: Any,
                c: Optional[Any],
                r: Optional[Any],
                breach_event: Optional[Any] = None,
                **extra_kwargs: Any,
            ) -> StepResult:
                # Use the wrapped branch if it exists in wrapped_branches
                # Find the key for this step
                step_key = None
                for key, branch in wrapped_branches.items():
                    if branch is s:
                        step_key = key
                        break

                # Use the wrapped branch if found, otherwise use the original
                step_to_execute = wrapped_branches.get(step_key, s) if step_key else s

                # Create ExecutionFrame for the recursive call
                frame = ExecutionFrame(
                    step=step_to_execute,
                    data=d,
                    context=c,
                    resources=r,
                    limits=limits,
                    stream=False,  # Default for parallel branches
                    on_chunk=None,  # Default for parallel branches
                    breach_event=breach_event,
                    context_setter=context_setter or (lambda result, ctx: None),  # Use provided or default
                )
                return await self.execute(frame)

        # Simplified branch execution without complex locking
        async def run_branch(key: str, branch_pipe: Any) -> tuple[str, StepResult]:
            """Execute a single branch with simplified logic."""
            try:
                # Isolate context for this branch
                if context is not None:
                    branch_context = copy.deepcopy(context)
                else:
                    from flujo.domain.models import PipelineContext
                    branch_context = PipelineContext(initial_prompt=str(data))

                if not hasattr(branch_pipe, "name"):
                    object.__setattr__(branch_pipe, "name", f"parallel_branch_{key}")

                current_data = data
                total_latency = 0.0
                total_cost = 0.0
                total_tokens = 0
                all_successful = True
                step_outputs: List[StepResult] = []
                # Execute the branch using step_executor
                step_result: Optional[StepResult] = await step_executor(
                    branch_pipe, current_data, branch_context, resources, breach_event
                )

                # Capture the final state of branch_context
                final_branch_context = (
                    copy.deepcopy(branch_context) if branch_context is not None else None
                )

                # Preserve the original StepResult exactly; just attach branch_context
                cloned: StepResult = (
                    step_result.model_copy(deep=True) if step_result else StepResult(name="unknown")
                )
                cloned.branch_context = final_branch_context

                # Feed usage metrics to the governor
                cost_delta = getattr(cloned, "cost_usd", 0.0)
                token_delta = getattr(cloned, "token_counts", 0)
                telemetry.logfire.debug(
                    f"Adding usage to governor: cost={cost_delta}, tokens={token_delta}"
                )

                breach_occurred = await usage_governor.add_usage(cost_delta=cost_delta, token_delta=token_delta, result=cloned)

                telemetry.logfire.debug(f"Breach occurred: {breach_occurred}")

                # If a breach occurred, raise the error immediately
                if breach_occurred:
                    breach_error = usage_governor.get_error()
                    telemetry.logfire.debug(f"Breach error: {breach_error}")
                    if breach_error:
                        # Add current result to shared step history and update error
                        self._step_history_so_far.append(cloned)
                        breach_error.result.step_history = self._step_history_so_far
                        raise breach_error

                return key, cloned
            except asyncio.CancelledError:
                return key, StepResult(
                    name=getattr(branch_pipe, "name", str(key)),
                    output=None,
                    success=False,
                    branch_context=None,
                )

        # Execute all branches concurrently with proper concurrency control
        try:
            # Compute timeout from step config or use default
            timeout_seconds = (
                getattr(parallel_step.config, "timeout", 30)
                if hasattr(parallel_step, "config")
                else 30
            )

            # Use semaphore to limit concurrent executions
            semaphore = asyncio.Semaphore(self._concurrency_limit or 10)

            async def run_branch_with_semaphore(
                key: str, branch_pipe: Any
            ) -> tuple[str, StepResult]:
                async with semaphore:
                    return await run_branch(key, branch_pipe)

            # Create tasks for all branches with concurrency control
            telemetry.logfire.debug(f"Creating {len(wrapped_branches)} parallel tasks")
            tasks = [
                asyncio.create_task(
                    run_branch_with_semaphore(key, branch_pipe), name=f"branch_{key}"
                )
                for key, branch_pipe in wrapped_branches.items()
            ]

            # Replace gather with early-exit wait for proactive cancellation
            branch_results = {}
            outputs = {}

            while tasks:
                # Wait for first completion
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Process completed tasks
                for task in done:
                    try:
                        res_key, step_res = await task
                        branch_results[res_key] = step_res
                        if step_res.success:
                            outputs[res_key] = step_res.output
                    except UsageLimitExceededError as e:
                        # Propagate usage limit errors to the main execution
                        telemetry.logfire.debug(
                            f"Usage limit exceeded in task {task.get_name()}: {str(e)}"
                        )
                        raise e
                    except Exception as e:
                        # Handle other task exceptions
                        key = (
                            task.get_name().split("_", 1)[1]
                            if "_" in task.get_name() else "unknown"
                        )
                        telemetry.logfire.debug(f"Task failed: {key}, error: {str(e)}")
                        branch_results[key] = StepResult(
                            name=key, success=False, feedback=f"Branch execution failed: {str(e)}"
                        )

                # Check for usage limit breach and cancel remaining tasks
                telemetry.logfire.debug(f"Checking for breach: {usage_governor.breached()}")
                if usage_governor.breached():
                    telemetry.logfire.debug("Breach detected, cancelling remaining tasks")
                    for task in pending:
                        task.cancel()

                    # Harvest whatever finished before cancellation
                    for task in pending:
                        try:
                            res_key, step_res = await task
                            branch_results[res_key] = step_res
                        except asyncio.CancelledError:
                            key = (
                                task.get_name().split("_", 1)[1]
                                if "_" in task.get_name() else "unknown"
                            )
                            branch_results[key] = StepResult(
                                name=key,
                                success=False,
                                feedback="Cancelled due to usage-limit breach",
                            )
                    break

                # Update tasks list to remaining pending tasks
                tasks = list(pending)

        except asyncio.TimeoutError:
            # Cancel all tasks and create timeout results
            for task in tasks:
                if not task.done():
                    task.cancel()

            branch_results = {}
            for key in wrapped_branches.keys():
                branch_results[key] = StepResult(
                    name=key, success=False, feedback="Branch execution timed out"
                )

        # Check for usage limit breach
        if usage_governor.breached():
            # Get the actual error that was raised during the breach
            breach_error = usage_governor.get_error()
            if breach_error:
                # Update the error's result with the final history
                final_history = list(branch_results.values())
                breach_error.result.step_history = final_history
                raise breach_error
            else:
                # If no specific error, create a generic one
                raise UsageLimitExceededError(
                    "Parallel execution exceeded usage limits.",
                    PipelineResult(
                        step_history=list(branch_results.values()),
                        total_cost_usd=usage_governor.total_cost,
                    ),
                )

        # Aggregate results
        all_successful = all(r.success for r in branch_results.values())
        result.success = all_successful
        result.output = outputs
        result.step_history = list(branch_results.values())

        # Calculate total cost and tokens from successful branches
        total_cost = sum(r.cost_usd for r in branch_results.values() if r.success)
        total_tokens = sum(
            r.token_counts for r in branch_results.values() if r.success
        )
        result.cost_usd = total_cost
        result.token_counts = total_tokens

        # Merge context updates from all successful branches
        if context is not None:
            for branch_result in branch_results.values():
                if branch_result.success and branch_result.branch_context is not None:
                    # Import the safe_merge_context_updates function
                    from ...utils.context import safe_merge_context_updates
                    safe_merge_context_updates(context, branch_result.branch_context)

        # Set final context if provided
        if context_setter and context:
            # Create a PipelineResult instead of PipelineContext for the context_setter
            pipeline_result: PipelineResult[Any] = PipelineResult(
                step_history=result.step_history,
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
            )
            context_setter(pipeline_result, context)

        # Update overall step history
        self._step_history_so_far.extend(result.step_history)
        
        # Return the merged context in the branch_context field
        result.branch_context = context

        return result

    async def _handle_loop_step(
        self,
        loop_step: LoopStep[Any],
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """Handle loop step execution."""
        import time
        from ...domain.dsl.pipeline import Pipeline

        telemetry.logfire.debug("=== HANDLE LOOP STEP ===")
        telemetry.logfire.debug(f"Loop step name: {loop_step.name}")

        # Initialize result
        result = StepResult(
            name=loop_step.name,
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
        cumulative_cost = 0.0
        cumulative_tokens = 0
        iterations = 0
        max_iterations = getattr(loop_step, "max_loops", 10)
        current_data = data

        try:
            telemetry.logfire.debug(f"Starting LoopStep: max_iterations={max_iterations}, limits={limits}")
            
            # Initialize mutable context for loop iterations
            current_context = context
            if current_context is None:
                from flujo.domain.models import PipelineContext
                current_context = PipelineContext(initial_prompt=str(current_data))
            
            # Apply initial input mapper if provided
            if hasattr(loop_step, "initial_input_to_loop_body_mapper") and loop_step.initial_input_to_loop_body_mapper:
                try:
                    current_data = loop_step.initial_input_to_loop_body_mapper(data, context)
                except Exception as e:
                    result.success = False
                    result.feedback = f"Initial input mapper failed: {str(e)}"
                    result.latency_s = time.monotonic() - start_time
                    result.attempts = 0  # No iterations attempted
                    telemetry.logfire.error(f"Error in initial input mapper for LoopStep '{loop_step.name}': {str(e)}")
                    return result
            
            # Track loop termination reason
            loop_exit_reason = None
            last_body_error = None
            
            # Main loop
            while iterations < max_iterations:
                iterations += 1
                telemetry.logfire.info(f"LoopStep '{loop_step.name}': Starting Iteration {iterations}/{max_iterations}")
                telemetry.logfire.debug(f"Starting iteration {iterations}, current_data={current_data}, current_context.val={getattr(current_context, 'val', 'N/A')}")

                # Apply iteration input mapper if provided (for subsequent iterations)
                if iterations > 1 and hasattr(loop_step, "iteration_input_mapper") and loop_step.iteration_input_mapper:
                    try:
                        # Use raw_output for iteration input mapper, not current_data (which may be mapped)
                        iteration_input = loop_step.iteration_input_mapper(raw_output, current_context, iterations)
                        current_data = iteration_input
                        telemetry.logfire.debug(f"After iteration input mapper: current_data={current_data}")
                    except Exception as e:
                        result.success = False
                        result.feedback = f"Iteration input mapper failed: {str(e)}"
                        result.latency_s = time.monotonic() - start_time
                        telemetry.logfire.error(f"Error in iteration input mapper for LoopStep '{loop_step.name}': {str(e)}")
                        return result

                # Execute the loop body using the executor's own methods
                if isinstance(loop_step.loop_body_pipeline, Pipeline):
                    # Execute pipeline branch by executing each step in the pipeline
                    current_body_data = current_data
                    
                    # Isolate context for body execution
                    if current_context is not None:
                        body_context = copy.deepcopy(current_context)
                    else:
                        from flujo.domain.models import PipelineContext
                        body_context = PipelineContext(initial_prompt=str(current_body_data))
                    
                    # Execute each step in the pipeline
                    all_successful = True
                    final_output = None
                    total_cost = 0.0
                    total_tokens = 0
                    body_error_message = None
                    
                    for step in loop_step.loop_body_pipeline.steps:
                        with telemetry.logfire.span(step.name) as step_span:
                            step_result = await self.execute_step(
                                step=step,
                                data=current_body_data,
                                context=body_context,
                                resources=resources,
                                limits=limits,
                                stream=False,
                                on_chunk=None,
                                breach_event=None,
                            )
                        
                        total_cost += step_result.cost_usd
                        total_tokens += step_result.token_counts
                        
                        if not step_result.success:
                            all_successful = False
                            body_error_message = step_result.feedback
                            last_body_error = body_error_message
                            # Don't break here - continue to capture context updates
                        
                        # Use output as input for next step (even if failed)
                        current_body_data = step_result.output
                    
                    # Capture the final state of body_context
                    final_body_context = (
                        copy.deepcopy(body_context) if body_context is not None else None
                    )
                    
                    # Merge body context back into current context
                    if final_body_context is not None and current_context is not None:
                        safe_merge_context_updates(current_context, final_body_context)
                    
                    body_result = StepResult(
                        name=loop_step.name,
                        output=current_body_data,  # Use raw output for exit condition check
                        success=all_successful,
                        attempts=1,
                        latency_s=time.monotonic() - start_time,
                        token_counts=total_tokens,
                        cost_usd=total_cost,
                        feedback="Loop body executed successfully" if all_successful else f"Loop body failed: {body_error_message}",
                        branch_context=current_context,
                        metadata_={},
                    )
                else:
                    # Execute as a regular step
                    body_result = await self.execute_step(
                        step=loop_step.loop_body_pipeline,
                        data=current_data,
                        context=current_context,
                        resources=resources,
                        limits=limits,
                        stream=False,
                        on_chunk=None,
                        breach_event=None,
                    )
                    
                    # Update current_context with branch_context from the result
                    if body_result.branch_context is not None:
                        current_context = body_result.branch_context

                # Update cumulative costs (even if body failed)
                cumulative_cost += body_result.cost_usd
                cumulative_tokens += body_result.token_counts

                # Store raw output for exit condition check
                raw_output = body_result.output

                # Update usage meter with actual costs from this iteration
                if self._usage_meter is not None:
                    await self._usage_meter.add(
                        cost_usd=body_result.cost_usd,
                        prompt_tokens=body_result.token_counts,
                        completion_tokens=0  # Assuming all tokens are prompt tokens for simplicity
                    )

                telemetry.logfire.debug(f"After usage meter update")

                telemetry.logfire.debug(f"About to check exit condition")

                # Check exit condition BEFORE output mapper (exit condition should use raw output)
                if hasattr(loop_step, "exit_condition_callable") and loop_step.exit_condition_callable:
                    try:
                        telemetry.logfire.debug(f"Checking exit condition: output={raw_output}, context.val={getattr(current_context, 'val', 'N/A')}")
                        should_exit = loop_step.exit_condition_callable(raw_output, current_context)
                        telemetry.logfire.debug(f"Exit condition result: {should_exit}")
                        if should_exit:
                            loop_exit_reason = "condition"
                            # Use the current iteration's output as the final result
                            current_data = raw_output
                            # Apply output mapper if present
                            if hasattr(loop_step, "loop_output_mapper") and loop_step.loop_output_mapper:
                                try:
                                    current_data = loop_step.loop_output_mapper(raw_output, current_context)
                                    telemetry.logfire.debug(f"Final output mapper applied: {current_data}")
                                except Exception as e:
                                    result.success = False
                                    result.feedback = f"Loop output mapper failed (final): {str(e)}"
                                    result.latency_s = time.monotonic() - start_time
                                    telemetry.logfire.error(f"Error in loop output mapper for LoopStep '{loop_step.name}' (final): {str(e)}")
                                    return result
                            break
                    except Exception as e:
                        telemetry.logfire.warning(f"Exit condition evaluation failed: {e}")
                        # Exit condition failed - fail the loop immediately
                        result.success = False
                        result.feedback = f"Exit condition failed: {str(e)}"
                        result.output = raw_output
                        result.cost_usd = cumulative_cost
                        result.token_counts = cumulative_tokens
                        result.latency_s = time.monotonic() - start_time
                        result.metadata_["iterations"] = iterations
                        result.attempts = iterations
                        result.branch_context = current_context
                        return result

                # Apply output mapper if provided (after exit condition check, for next iteration)
                if hasattr(loop_step, "loop_output_mapper") and loop_step.loop_output_mapper:
                    try:
                        current_data = loop_step.loop_output_mapper(raw_output, current_context)
                        telemetry.logfire.debug(f"After output mapper in loop: current_context.val={getattr(current_context, 'val', 'N/A')}")
                    except Exception as e:
                        result.success = False
                        result.feedback = f"Loop output mapper failed: {str(e)}"
                        result.latency_s = time.monotonic() - start_time
                        telemetry.logfire.error(f"Error in loop output mapper for LoopStep '{loop_step.name}': {str(e)}")
                        return result
                else:
                    # If no output mapper, use raw output for next iteration
                    current_data = raw_output

            # After the loop, set result fields
            # Determine success based on exit reason
            if loop_exit_reason == "condition":
                result.success = True
                result.feedback = "Loop exited by condition"
            elif iterations >= max_iterations:
                result.success = False
                result.feedback = f"Loop terminated due to max_loops ({max_iterations})"
            else:
                # Loop failed due to other reasons
                result.success = False
                if last_body_error:
                    result.feedback = f"Loop failed: {last_body_error}"
                else:
                    result.feedback = "Loop failed for unknown reason"
            
            result.output = current_data
            result.cost_usd = cumulative_cost
            result.token_counts = cumulative_tokens
            result.latency_s = time.monotonic() - start_time
            result.metadata_["iterations"] = iterations
            result.attempts = iterations
            result.branch_context = current_context
            telemetry.logfire.debug(f"Final context after loop: val={getattr(current_context, 'val', 'N/A')}")
            return result
        except UsageLimitExceededError as e:
            # Re-raise UsageLimitExceededError to preserve the specific exception type
            raise e
        except Exception as e:
            result.success = False
            result.feedback = f"Loop step failed: {str(e)}"
            result.latency_s = time.monotonic() - start_time
            telemetry.logfire.error(f"Error in LoopStep '{loop_step.name}': {str(e)}")

        # After the loop, set result.output to the last output value
        result.output = current_data
        # Apply output mapper one more time if present to ensure context modifications are preserved
        if hasattr(loop_step, "loop_output_mapper") and loop_step.loop_output_mapper:
            try:
                result.output = loop_step.loop_output_mapper(result.output, current_context)
                telemetry.logfire.debug(f"After final output mapper: current_context.val={getattr(current_context, 'val', 'N/A')}")
            except Exception as e:
                result.success = False
                result.feedback = f"Loop output mapper failed (final): {str(e)}"
                result.latency_s = time.monotonic() - start_time
                telemetry.logfire.error(f"Error in loop output mapper for LoopStep '{loop_step.name}' (final): {str(e)}")
                return result
        telemetry.logfire.debug(f"Final context after loop: val={getattr(current_context, 'val', 'N/A')}")

        return result

    async def _handle_conditional_step(
        self,
        conditional_step: ConditionalStep[Any],
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        """Handle conditional step execution."""
        import time
        from ...domain.dsl.pipeline import Pipeline

        telemetry.logfire.debug("=== HANDLE CONDITIONAL STEP ===")
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

        try:
            # Execute condition
            branch_key = conditional_step.condition_callable(data, context)

            # Determine which branch to execute
            branch_to_execute = None
            if branch_key in conditional_step.branches:
                branch_to_execute = conditional_step.branches[branch_key]
                result.metadata_["executed_branch_key"] = branch_key
                telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}'")
                telemetry.logfire.info(f"Executing branch for key '{branch_key}'")
                # Set span attribute for tracing
                with telemetry.logfire.span(f"branch_{branch_key}") as span:
                    span.set_attribute("executed_branch_key", branch_key)
            elif conditional_step.default_branch_pipeline is not None:
                branch_to_execute = conditional_step.default_branch_pipeline
                result.metadata_["executed_branch_key"] = "default"
                telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}', using default branch")
                telemetry.logfire.info("Executing default branch")
                # Set span attribute for tracing
                with telemetry.logfire.span("branch_default") as span:
                    span.set_attribute("executed_branch_key", "default")
            else:
                telemetry.logfire.warn(f"No branch matches condition '{branch_key}' and no default branch provided")

            if branch_to_execute:
                # Execute the selected branch using the executor's own methods
                if isinstance(branch_to_execute, Pipeline):
                    # Execute pipeline branch by executing each step in the pipeline
                    branch_data = data
                    
                    # Apply input mapper if provided (on main context)
                    if conditional_step.branch_input_mapper:
                        branch_data = conditional_step.branch_input_mapper(data, context)
                    
                    # Isolate context for branch execution
                    if context is not None:
                        branch_context = copy.deepcopy(context)
                    else:
                        from flujo.domain.models import PipelineContext
                        branch_context = PipelineContext(initial_prompt=str(branch_data))
                    
                    # Execute each step in the pipeline
                    current_data = branch_data
                    total_cost = 0.0
                    total_tokens = 0
                    all_successful = True
                    step_results = []
                    branch_error_message = None
                    
                    for step in branch_to_execute.steps:
                        with telemetry.logfire.span(step.name) as step_span:
                            step_result = await self.execute_step(
                                step=step,
                                data=current_data,
                                context=branch_context,
                                resources=resources,
                                limits=limits,
                                stream=False,
                                on_chunk=None,
                                breach_event=None,
                            )
                        
                        step_results.append(step_result)
                        total_cost += step_result.cost_usd
                        total_tokens += step_result.token_counts
                        
                        if not step_result.success:
                            all_successful = False
                            branch_error_message = step_result.feedback
                            break
                        
                        # Use output as input for next step
                        current_data = step_result.output
                    
                    # Apply output mapper if provided (on main context)
                    final_output = current_data
                    if conditional_step.branch_output_mapper:
                        final_output = conditional_step.branch_output_mapper(current_data, branch_key, context)
                    
                    # Capture the final state of branch_context
                    final_branch_context = (
                        copy.deepcopy(branch_context) if branch_context is not None else None
                    )
                    
                    # Merge branch context back into main context (regardless of success/failure)
                    # But only if no mappers are used, since mappers modify the main context directly
                    if (final_branch_context is not None and context is not None and 
                        conditional_step.branch_input_mapper is None and 
                        conditional_step.branch_output_mapper is None):
                        safe_merge_context_updates(context, final_branch_context)
                    
                    result.success = all_successful
                    result.output = final_output
                    result.cost_usd = total_cost
                    result.token_counts = total_tokens
                    result.latency_s = time.monotonic() - start_time
                    result.metadata_["executed_branch_key"] = branch_key
                    result.branch_context = final_branch_context
                    
                    if all_successful:
                        result.feedback = f"Branch '{branch_key}' executed successfully"
                    else:
                        result.feedback = f"Failure in branch '{branch_key}': {branch_error_message}"
                        
                else:
                    # Execute as a regular step
                    branch_result = await self._execute_simple_step(
                        step=branch_to_execute,
                        data=data,
                        context=context,
                        resources=resources,
                        limits=limits,
                        stream=False,
                        on_chunk=None,
                        cache_key=None,  # No caching for conditional branches
                        breach_event=None,
                        _fallback_depth=_fallback_depth,
                    )

                    result.success = branch_result.success
                    result.output = branch_result.output
                    result.feedback = branch_result.feedback
                    result.cost_usd = branch_result.cost_usd
                    result.token_counts = branch_result.token_counts
                    result.latency_s = time.monotonic() - start_time
                    result.metadata_.update(branch_result.metadata_ or {})
                    result.branch_context = branch_result.branch_context
            else:
                # No branch to execute and no default branch
                result.success = False
                result.output = data
                result.latency_s = time.monotonic() - start_time
                result.feedback = f"No branch matches condition '{branch_key}' and no default branch provided"

        except Exception as e:
            result.success = False
            result.feedback = f"Error executing conditional logic or branch: {str(e)}"
            result.latency_s = time.monotonic() - start_time
            telemetry.logfire.error(f"Error in conditional step '{conditional_step.name}': {str(e)}")

        return result

    async def _handle_cache_step(
        self,
        cache_step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[
            Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
        ] = None,
    ) -> StepResult:
        """Handle cache step execution."""
        import time

        telemetry.logfire.debug("=== HANDLE CACHE STEP ===")
        telemetry.logfire.debug(f"Cache step name: {cache_step.name}")

        # Initialize result
        result = StepResult(
            name=cache_step.name,
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

        try:
            # Check cache first
            cache_key = None
            if self._cache_backend is not None:
                cache_key = self._cache_key_generator.generate_key(cache_step, data, context, resources)
                cached_result = await self._cache_backend.get(cache_key)
                if cached_result is not None:
                    result = cached_result
                    result.metadata_["cache_hit"] = True
                    result.latency_s = time.monotonic() - start_time
                    return result

            # Cache miss - execute the step
            if step_executor:
                result = await step_executor(
                    cache_step.wrapped_step,
                    data,
                    context,
                    resources,
                    breach_event,
                )
            else:
                # Create ExecutionFrame for the recursive call
                frame = ExecutionFrame(
                    step=cache_step.wrapped_step,
                    data=data,
                    context=context,
                    resources=resources,
                    limits=limits,
                    stream=False,  # Default for cache steps
                    on_chunk=None,  # Default for cache steps
                    breach_event=breach_event,
                    context_setter=context_setter or (lambda result, ctx: None),  # Use provided or default
                )
                result = await self.execute(frame)

            # Cache successful result
            if result.success and self._cache_backend is not None and cache_key is not None:
                if result.metadata_ is None:
                    result.metadata_ = {}
                result.metadata_["cache_hit"] = False
                await self._cache_backend.put(cache_key, result, ttl_s=3600)

            result.latency_s = time.monotonic() - start_time

        except Exception as e:
            result.success = False
            result.feedback = f"Cache step failed: {str(e)}"
            result.latency_s = time.monotonic() - start_time

        return result

    async def _handle_hitl_step(
        self,
        hitl_step: HumanInTheLoopStep,
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[
            Callable[[Any, Any, Optional[Any], Optional[Any], Optional[Any]], Awaitable[StepResult]]
        ] = None,
    ) -> StepResult:
        """Handle Human-in-the-Loop step execution."""
        import time
        from ...exceptions import PausedException

        telemetry.logfire.debug("=== HANDLE HITL STEP ===")
        telemetry.logfire.debug(f"HITL step name: {hitl_step.name}")

        # Initialize result
        result = StepResult(
            name=hitl_step.name,
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

        # Update context scratchpad if available
        if isinstance(context, PipelineContext):
            try:
                context.scratchpad["status"] = "paused"
                context.scratchpad["hitl_message"] = hitl_step.message_for_user or str(data)
                context.scratchpad["hitl_data"] = data
            except Exception as e:
                telemetry.logfire.error(f"Failed to update context scratchpad: {e}")

        # HITL steps pause execution for human input
        # The actual human input handling is done by the orchestrator
        # For now, we'll just pause the execution
        raise PausedException(
            f"Human-in-the-Loop step '{hitl_step.name}' requires human input"
        )

    async def _handle_dynamic_router_step(
        self,
        router_step: DynamicParallelRouterStep[Any],
        data: Any,
        context: Optional[TContext_w_Scratch],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> StepResult:
        """Handle dynamic router step execution."""
        import time

        telemetry.logfire.debug("=== HANDLE DYNAMIC ROUTER STEP ===")
        telemetry.logfire.debug(f"Dynamic router step name: {router_step.name}")

        # Initialize result
        result = StepResult(
            name=router_step.name,
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

        try:
            # Execute router agent to determine which branches to execute
            router_frame = ExecutionFrame(
                step=router_step.router_agent,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                stream=False,  # Default for router
                on_chunk=None,  # Default for router
                breach_event=None,  # Default for router
                context_setter=lambda result, ctx: None,  # Default for router
            )
            router_result = await self.execute(router_frame)

            if not router_result.success:
                result.success = False
                result.feedback = f"Router agent failed: {router_result.feedback}"
                result.latency_s = time.monotonic() - start_time
                return result

            # Determine which branches to execute based on router output
            selected_branches = router_result.output
            if not isinstance(selected_branches, (list, tuple)):
                selected_branches = [selected_branches]

            # Execute selected branches in parallel
            branch_results = {}
            for branch_key in selected_branches:
                if branch_key in router_step.branches:
                    branch_pipeline = router_step.branches[branch_key]
                    # Wrap the pipeline in a step adapter for the ExecutionFrame
                    from ...domain.dsl.pipeline import Pipeline
                    if isinstance(branch_pipeline, Pipeline):
                        # Create a step adapter for the pipeline
                        step_adapter = _PipelineStepAdapter(branch_pipeline, f"branch_{branch_key}")
                    else:
                        step_adapter = branch_pipeline
                    
                    branch_frame = ExecutionFrame(
                        step=step_adapter,
                        data=data,
                        context=context,
                        resources=resources,
                        limits=limits,
                        stream=False,  # Default for branches
                        on_chunk=None,  # Default for branches
                        breach_event=None,  # Default for branches
                        context_setter=lambda result, ctx: None,  # Default for branches
                    )
                    branch_result = await self.execute(branch_frame)
                    branch_results[branch_key] = branch_result

            # Combine results
            result.success = all(r.success for r in branch_results.values())
            result.output = branch_results
            result.metadata_["executed_branches"] = list(branch_results.keys())
            result.latency_s = time.monotonic() - start_time

        except Exception as e:
            result.success = False
            result.feedback = f"Dynamic router step failed: {str(e)}"
            result.latency_s = time.monotonic() - start_time

        return result



    async def execute_step(
        self,
        step: Any,
        data: Any,
        *,
        context: Optional[TContext_w_Scratch] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        **kwargs: Any,
    ) -> StepResult:
        """Alias for execute method to maintain backward compatibility."""
        # Create ExecutionFrame for the backward compatibility call
        frame = ExecutionFrame(
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            breach_event=breach_event,
            context_setter=lambda result, ctx: None,  # Default for backward compatibility
        )
        return await self.execute(frame)

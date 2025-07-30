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
import time
import hashlib
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
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
)
from weakref import WeakKeyDictionary

from ...domain.dsl.step import HumanInTheLoopStep
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.models import BaseModel, StepResult, UsageLimits, PipelineResult
from ...domain.validation import ValidationResult
from ...exceptions import (
    UsageLimitExceededError,
    PausedException,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PricingNotConfiguredError,
    ContextInheritanceError,
)
from ...steps.cache_step import CacheStep
from ...utils.performance import time_perf_ns, time_perf_ns_to_seconds
from ...cost import extract_usage_metrics
from ...infra import telemetry

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
    async def guard(self, limits: UsageLimits) -> None:
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
            s = self._json.dumps(obj, sort_keys=True, separators=(",", ":"))
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
    _store: OrderedDict[str, tuple[StepResult, float]] = field(
        init=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl_s < 0:
            raise ValueError("ttl must be non-negative")

    async def get(self, key: str) -> Optional[StepResult]:
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
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # Remove oldest

        self._store[key] = (value.model_copy(deep=True), time.monotonic())

    async def clear(self) -> None:
        self._store.clear()


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

    async def guard(self, limits: UsageLimits) -> None:
        async with self._lock:
            # Use approximate comparison for floating point
            if (
                limits.total_cost_usd_limit is not None
                and self.total_cost_usd > limits.total_cost_usd_limit + 1e-10
            ):
                raise UsageLimitExceededError(
                    f"Cost limit of ${limits.total_cost_usd_limit} exceeded (current: ${self.total_cost_usd})",
                    PipelineResult(step_history=[], total_cost_usd=self.total_cost_usd),
                )

            total_tokens = self.prompt_tokens + self.completion_tokens
            if limits.total_tokens_limit is not None and total_tokens > limits.total_tokens_limit:
                raise UsageLimitExceededError(
                    f"Token limit of {limits.total_tokens_limit} exceeded (current: {total_tokens})",
                    PipelineResult(step_history=[], total_cost_usd=self.total_cost_usd),
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
    ) -> Any:
        """Run agent with proper parameter filtering and fallback strategies."""
        import inspect
        from unittest.mock import Mock, MagicMock, AsyncMock
        from ..context_manager import _accepts_param, _should_pass_context
        from ...signature_tools import analyze_signature

        if agent is None:
            raise RuntimeError("Agent is None")

        # Step 1: Extract the target agent (handle wrapped agents)
        # Follow the same pattern as step_logic.py
        target_agent = getattr(agent, "_agent", agent)

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
                if _should_pass_context(spec, context, executable_func):
                    filtered_kwargs["context"] = context

                # Add resources if the function accepts it
                if resources is not None and _accepts_param(executable_func, "resources"):
                    filtered_kwargs["resources"] = resources

                # Add other options based on function signature
                for key, value in options.items():
                    if value is not None and _accepts_param(executable_func, key):
                        filtered_kwargs[key] = value

            except Exception:
                # If signature analysis fails, try basic parameter passing
                filtered_kwargs.update(options)
                if context is not None:
                    filtered_kwargs["context"] = context
                if resources is not None:
                    filtered_kwargs["resources"] = resources
        else:
            # For mocks, pass all parameters
            filtered_kwargs.update(options)
            if context is not None:
                filtered_kwargs["context"] = context
            if resources is not None:
                filtered_kwargs["resources"] = resources

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
                # Log error but continue with original output
                try:
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                processed_data = data

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


def _should_pass_context_to_plugin(context: Optional[Any], func: Callable[..., Any]) -> bool:
    """Determine if context should be passed to a plugin based on signature analysis.

    This is more conservative than _accepts_param - it only passes context
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
                if _should_pass_context_to_plugin(context, func):
                    plugin_kwargs["context"] = context

                # Call plugin
                result = await func(processed_data, **plugin_kwargs)

                # Handle PluginOutcome
                if isinstance(result, PluginOutcome):
                    if not result.success:
                        # Plugin validation failed - this should cause the step to fail
                        raise ValueError(f"Plugin validation failed: {result.feedback}")
                    if result.new_solution is not None:
                        processed_data = result.new_solution
                else:
                    # Plugin returned new data
                    processed_data = result

            except Exception as e:
                # Log error and re-raise to cause step failure
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
        elif hasattr(obj, "isoformat"):  # datetime objects
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

TContext = TypeVar("TContext", bound=BaseModel)


class ExecutorCore(Generic[TContext]):
    """
    Modular, policy-driven step executor with deterministic behavior.

    This is the core implementation that orchestrates all concerns through
    dependency injection. Each component is replaceable via interfaces.
    """

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
    ):
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
        self._concurrency = asyncio.Semaphore(concurrency_limit or cpu_count() * 2)

    async def execute(
        self,
        step: Any,
        data: Any,
        *,
        context: Optional[TContext] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        result: Optional[Any] = None,  # For backward compatibility
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a step with the given data and context."""

        telemetry.logfire.debug("=== EXECUTOR CORE EXECUTE ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")

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
        if self._is_complex_step(step):
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
            )

        telemetry.logfire.debug(f"Simple step detected: {step.name}")
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
        )

    def _is_complex_step(self, step: Any) -> bool:
        """Check if step needs complex handling."""
        telemetry.logfire.debug("=== IS COMPLEX STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")

        # Check for specific step types
        if isinstance(
            step,
            (
                CacheStep,
                LoopStep,
                ConditionalStep,
                DynamicParallelRouterStep,
                ParallelStep,
                HumanInTheLoopStep,
            ),
        ):
            if isinstance(step, ParallelStep):
                telemetry.logfire.debug(f"ParallelStep detected: {step.name}")
            elif isinstance(step, DynamicParallelRouterStep):
                telemetry.logfire.debug(f"DynamicParallelRouterStep detected: {step.name}")
            telemetry.logfire.debug(f"Complex step detected: {step.name}")
            return True

        # Check for validation steps
        if hasattr(step, "meta") and step.meta and step.meta.get("is_validation_step", False):
            telemetry.logfire.debug(f"Validation step detected: {step.name}")
            return True

        # Check for steps with fallbacks
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            telemetry.logfire.debug(f"Step with fallback detected: {step.name}")
            return True

        # Check for steps with plugins (plugins can have redirects, feedback, etc.)
        if hasattr(step, "plugins") and step.plugins:
            telemetry.logfire.debug(f"Step with plugins detected: {step.name}")
            return True

        telemetry.logfire.debug(f"Simple step detected: {step.name}")
        return False

    async def _execute_complex_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        breach_event: Optional[Any],
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]],
        cache_key: Optional[str] = None,
    ) -> StepResult:
        """Execute complex steps using step logic helpers."""

        # Import step logic helpers
        from .step_logic import (
            _handle_cache_step,
            _handle_loop_step,
            _handle_conditional_step,
            _handle_dynamic_router_step,
            _handle_parallel_step,
            _handle_hitl_step,
            _run_step_logic,
            _default_set_final_context,
        )

        if context_setter is None:
            context_setter = _default_set_final_context

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

            return await self.execute(
                s,
                d,
                context=c,
                resources=r,
                limits=_limits,
                stream=_stream,
                on_chunk=_on_chunk,
                breach_event=breach_event,
                context_setter=_context_setter,
            )

        # Handle specific step types
        telemetry.logfire.debug("=== EXECUTE COMPLEX STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")

        if isinstance(step, CacheStep):
            telemetry.logfire.debug("Handling CacheStep")
            result = await _handle_cache_step(step, data, context, resources, step_executor)
        elif isinstance(step, LoopStep):
            telemetry.logfire.debug("Handling LoopStep")
            result = await _handle_loop_step(
                step,
                data,
                context,
                resources,
                step_executor,
                context_model_defined=True,
                usage_limits=limits,
                context_setter=context_setter,
            )
        elif isinstance(step, ConditionalStep):
            telemetry.logfire.debug("Handling ConditionalStep")
            result = await _handle_conditional_step(
                step,
                data,
                context,
                resources,
                step_executor,
                context_model_defined=True,
                usage_limits=limits,
                context_setter=context_setter,
            )
        elif isinstance(step, DynamicParallelRouterStep):
            telemetry.logfire.debug("Handling DynamicParallelRouterStep")
            result = await _handle_dynamic_router_step(
                step,
                data,
                context,
                resources,
                step_executor,
                context_model_defined=True,
                usage_limits=limits,
                context_setter=context_setter,
            )
        elif isinstance(step, ParallelStep):
            telemetry.logfire.debug("Handling ParallelStep")
            result = await _handle_parallel_step(
                step,
                data,
                context,
                resources,
                step_executor,
                context_model_defined=True,
                usage_limits=limits,
                context_setter=context_setter,
                breach_event=breach_event,
            )
        elif isinstance(step, HumanInTheLoopStep):
            telemetry.logfire.debug("Handling HumanInTheLoopStep")
            result = await _handle_hitl_step(step, data, context)
        else:
            telemetry.logfire.debug("Falling back to general step logic")
            # Fall back to general step logic
            result = await _run_step_logic(
                step=step,
                data=data,
                context=context,
                resources=resources,
                step_executor=step_executor,
                context_model_defined=True,
                usage_limits=limits,
                context_setter=context_setter,
                stream=stream,
                on_chunk=on_chunk,
            )

        # Cache successful result for complex steps
        if result.success and self._cache_backend is not None and cache_key is not None:
            if result.metadata_ is None:
                result.metadata_ = {}
            await self._cache_backend.put(cache_key, result, ttl_s=3600)

        return result

    async def _execute_step_logic(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
    ) -> StepResult:
        """Execute a single step with retry logic and usage tracking."""

        # Check for missing agent
        if not hasattr(step, "agent") or step.agent is None:
            from ...exceptions import MissingAgentError

            raise MissingAgentError(f"Step '{step.name}' has no agent configured")

        # Check usage limits before execution
        if limits:
            await self._usage_meter.guard(limits)

        last_exception: Exception | None = None
        accumulated_feedbacks: list[str] = []

        # Get agent from step
        agent = step.agent

        for attempt in range(1, step.config.max_retries + 1):
            # Start timing
            start_time = time_perf_ns()

            # Initialize cost variables
            cost_usd = 0.0
            prompt_tokens = 0
            completion_tokens = 0

            try:
                # Add accumulated feedback to data
                if accumulated_feedbacks:
                    feedback_text = "\n".join(accumulated_feedbacks)
                    if isinstance(data, dict):
                        data["feedback"] = data.get("feedback", "") + "\n" + feedback_text
                    else:
                        data = f"{str(data)}\n{feedback_text}"

                # Initialize processed_output to avoid UnboundLocalError
                processed_output = None

                # Apply prompt processors
                processed_data = await self._processor_pipeline.apply_prompt(
                    getattr(step.processors, "prompt_processors", []) if step.processors else [],
                    data,
                    context=context,
                )

                # Run agent
                raw_output = await self._agent_runner.run(
                    agent,
                    processed_data,
                    context=context,
                    resources=resources,
                    options={"temperature": step.config.temperature},
                    stream=stream,
                    on_chunk=on_chunk,
                )

                # Check for Mock objects in output (not the agent itself)
                from unittest.mock import Mock, MagicMock, AsyncMock

                if isinstance(raw_output, (Mock, MagicMock, AsyncMock)):
                    raise TypeError("returned a Mock object")

                # Apply output processors
                processed_output = await self._processor_pipeline.apply_output(
                    getattr(step.processors, "output_processors", []) if step.processors else [],
                    raw_output,
                    context=context,
                )

                # Run validators
                if step.validators:
                    await self._validator_runner.validate(
                        step.validators, processed_output, context=context
                    )

                # Persist validation results if requested
                if (
                    hasattr(step, "persist_validation_results_to")
                    and step.persist_validation_results_to
                    and context is not None
                ):
                    if hasattr(context, step.persist_validation_results_to):
                        history_list = getattr(context, step.persist_validation_results_to)
                        if isinstance(history_list, list):
                            # Note: validation_results is not available in this context
                            # The validation results are handled by the validator_runner
                            pass

                # Run plugins
                processed_output = await self._plugin_runner.run_plugins(
                    step.plugins, processed_output, context=context
                )

                # Extract usage metrics
                try:
                    prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                        raw_output, agent, step.name
                    )
                except PricingNotConfiguredError:
                    # Re-raise PricingNotConfiguredError for strict mode failures
                    raise
                except Exception:
                    # Fallback to zero cost if extraction fails
                    prompt_tokens, completion_tokens, cost_usd = 0, 0, 0.0

                # Track usage
                await self._usage_meter.add(cost_usd, prompt_tokens, completion_tokens)

                # Calculate latency
                latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_time)

                # Create result
                result = StepResult(
                    name=step.name,
                    output=getattr(processed_output, "output", processed_output),
                    success=True,
                    attempts=attempt,
                    latency_s=latency_s,
                    cost_usd=cost_usd,
                    token_counts=prompt_tokens + completion_tokens,
                    feedback=None,
                )

                # Cache successful result
                if self._cache_backend is not None and cache_key is not None:
                    if result.metadata_ is None:
                        result.metadata_ = {}
                    await self._cache_backend.put(cache_key, result, ttl_s=3600)

                return result

            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise critical exceptions immediately
                raise e
            except (
                TypeError,
                UsageLimitExceededError,
                PricingNotConfiguredError,
            ) as exc:
                # Don't retry these critical exceptions
                if isinstance(exc, TypeError) and "Mock object" in str(exc):
                    raise
                if isinstance(exc, (UsageLimitExceededError, PricingNotConfiguredError)):
                    raise
                last_exception = exc
                continue

            except ValueError as exc:
                # Check if this is a plugin validation failure
                if "Plugin validation failed" in str(exc):
                    # Extract feedback from the error message
                    feedback = str(exc).replace("Plugin validation failed: ", "")
                    accumulated_feedbacks.append(feedback)

                    # Add feedback to data for next retry
                    if feedback:
                        if isinstance(data, dict):
                            data["feedback"] = data.get("feedback", "") + "\n" + feedback
                        else:
                            data = f"{str(data)}\n{feedback}"

                    # Plugin validation failures should be retryable
                    last_exception = exc
                    continue
                else:
                    # Other ValueError exceptions should be retryable
                    last_exception = exc
                    continue

            except ContextInheritanceError:
                raise
            except Exception as exc:
                # Treat all other exceptions as retryable
                last_exception = exc
                continue

        # All retries failed
        latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_time)

        # Re-raise critical exceptions
        if isinstance(
            last_exception,
            (PausedException, InfiniteFallbackError, InfiniteRedirectError),
        ):
            raise last_exception

        # For validation failures, preserve the output
        preserved_output = None
        if isinstance(last_exception, ValueError) and "Validation failed" in str(last_exception):
            # Preserve the output that was generated before validation failed
            preserved_output = getattr(processed_output, "output", processed_output)

        # Return failed result
        result = StepResult(
            name=step.name,
            output=preserved_output,
            success=False,
            attempts=step.config.max_retries,
            feedback=f"Agent execution failed with {type(last_exception).__name__}: {last_exception}",
            latency_s=latency_s,
        )

        # Persist feedback if requested
        if (
            hasattr(step, "persist_feedback_to_context")
            and step.persist_feedback_to_context
            and context is not None
        ):
            if hasattr(context, step.persist_feedback_to_context):
                history_list = getattr(context, step.persist_feedback_to_context)
                if isinstance(history_list, list) and result.feedback:
                    history_list.append(result.feedback)

        return result

    async def _execute_simple_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        breach_event: Optional[Any],
    ) -> StepResult:
        """
        ✅ NEW: This method contains the migrated retry loop for simple steps.
        This is the clean, new implementation that replaces the legacy _run_step_logic
        for simple steps only.
        """
        # Check for missing agent
        if not hasattr(step, "agent") or step.agent is None:
            from ...exceptions import MissingAgentError

            raise MissingAgentError(f"Step '{step.name}' has no agent configured")

        # ✅ COMPATIBILITY: Call usage meter guard for direct ExecutorCore usage
        # This maintains backward compatibility while allowing ExecutionManager
        # to handle centralized enforcement when used through the manager.
        if limits and self._usage_meter:
            await self._usage_meter.guard(limits)

        last_exception: Exception | None = None
        accumulated_feedbacks: list[str] = []

        # Get agent from step
        agent = step.agent

        for attempt in range(1, step.config.max_retries + 1):
            # Start timing
            start_time = time_perf_ns()

            # Initialize cost variables
            cost_usd = 0.0
            prompt_tokens = 0
            completion_tokens = 0

            try:
                # Add accumulated feedback to data
                if accumulated_feedbacks:
                    feedback_text = "\n".join(accumulated_feedbacks)
                    if isinstance(data, dict):
                        data["feedback"] = data.get("feedback", "") + "\n" + feedback_text
                    else:
                        data = f"{str(data)}\n{feedback_text}"

                # Initialize processed_output to avoid UnboundLocalError
                processed_output = None

                # Apply prompt processors
                processed_data = await self._processor_pipeline.apply_prompt(
                    getattr(step.processors, "prompt_processors", []) if step.processors else [],
                    data,
                    context=context,
                )

                # Run agent
                raw_output = await self._agent_runner.run(
                    agent,
                    processed_data,
                    context=context,
                    resources=resources,
                    options={"temperature": step.config.temperature},
                    stream=stream,
                    on_chunk=on_chunk,
                )

                # Check for Mock objects in output (not the agent itself)
                from unittest.mock import Mock, MagicMock, AsyncMock

                if isinstance(raw_output, (Mock, MagicMock, AsyncMock)):
                    raise TypeError("returned a Mock object")

                # Apply output processors
                processed_output = await self._processor_pipeline.apply_output(
                    getattr(step.processors, "output_processors", []) if step.processors else [],
                    raw_output,
                    context=context,
                )

                # Run validators
                if step.validators:
                    await self._validator_runner.validate(
                        step.validators, processed_output, context=context
                    )

                # Persist validation results if requested
                if (
                    hasattr(step, "persist_validation_results_to")
                    and step.persist_validation_results_to
                    and context is not None
                ):
                    if hasattr(context, step.persist_validation_results_to):
                        history_list = getattr(context, step.persist_validation_results_to)
                        if isinstance(history_list, list):
                            # Note: validation_results is not available in this context
                            # The validation results are handled by the validator_runner
                            pass

                # Run plugins
                processed_output = await self._plugin_runner.run_plugins(
                    step.plugins, processed_output, context=context
                )

                # Extract usage metrics
                try:
                    prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                        raw_output, agent, step.name
                    )
                except PricingNotConfiguredError:
                    # Re-raise PricingNotConfiguredError for strict mode failures
                    raise
                except Exception:
                    # Fallback to zero cost if extraction fails
                    prompt_tokens, completion_tokens, cost_usd = 0, 0, 0.0

                # Track usage
                await self._usage_meter.add(cost_usd, prompt_tokens, completion_tokens)

                # Calculate latency
                latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_time)

                # Create result
                result = StepResult(
                    name=step.name,
                    output=getattr(processed_output, "output", processed_output),
                    success=True,
                    attempts=attempt,
                    latency_s=latency_s,
                    cost_usd=cost_usd,
                    token_counts=prompt_tokens + completion_tokens,
                    feedback=None,
                )

                # Cache successful result
                if self._cache_backend is not None and cache_key is not None:
                    if result.metadata_ is None:
                        result.metadata_ = {}
                    await self._cache_backend.put(cache_key, result, ttl_s=3600)

                return result

            except (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ) as e:
                # Re-raise control flow and critical exceptions immediately
                raise e
            except (
                TypeError,
                UsageLimitExceededError,
                PricingNotConfiguredError,
            ) as exc:
                # Don't retry these critical exceptions
                if isinstance(exc, TypeError) and "Mock object" in str(exc):
                    raise
                if isinstance(exc, (UsageLimitExceededError, PricingNotConfiguredError)):
                    raise
                last_exception = exc
                continue

            except ValueError as exc:
                # Check if this is a plugin validation failure
                if "Plugin validation failed" in str(exc):
                    # Extract feedback from the error message
                    feedback = str(exc).replace("Plugin validation failed: ", "")
                    accumulated_feedbacks.append(feedback)

                    # Add feedback to data for next retry
                    if feedback:
                        if isinstance(data, dict):
                            data["feedback"] = data.get("feedback", "") + "\n" + feedback
                        else:
                            data = f"{str(data)}\n{feedback}"

                    # Plugin validation failures should be retryable
                    last_exception = exc
                    continue
                else:
                    # Other ValueError exceptions should be retryable
                    last_exception = exc
                    continue

            except Exception as exc:
                # Treat all other exceptions as retryable
                last_exception = exc
                continue

        # All retries failed
        latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_time)

        # Re-raise critical exceptions
        if isinstance(
            last_exception,
            (
                PausedException,
                InfiniteFallbackError,
                InfiniteRedirectError,
                ContextInheritanceError,
            ),
        ):
            raise last_exception

        # For validation failures, preserve the output
        preserved_output = None
        if isinstance(last_exception, ValueError) and "Validation failed" in str(last_exception):
            # Preserve the output that was generated before validation failed
            preserved_output = getattr(processed_output, "output", processed_output)

        # Return failed result
        result = StepResult(
            name=step.name,
            output=preserved_output,
            success=False,
            attempts=step.config.max_retries,
            feedback=f"Agent execution failed with {type(last_exception).__name__}: {last_exception}",
            latency_s=latency_s,
        )

        # Persist feedback if requested
        if (
            hasattr(step, "persist_feedback_to_context")
            and step.persist_feedback_to_context
            and context is not None
        ):
            if hasattr(context, step.persist_feedback_to_context):
                history_list = getattr(context, step.persist_feedback_to_context)
                if isinstance(history_list, list) and result.feedback:
                    history_list.append(result.feedback)

        return result

    # Utility methods for inspection
    async def clear_cache(self) -> None:
        """Clear all cached results."""
        if self._cache_backend is not None:
            await self._cache_backend.clear()

    async def get_usage_snapshot(self) -> tuple[float, int, int]:
        """Get current usage metrics."""
        return await self._usage_meter.snapshot()


def traced(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Simple tracing decorator."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        try:
            from ...infra import telemetry

            return telemetry.logfire.instrument(name)(func)
        except Exception:
            return func

    return decorator


# --------------------------------------------------------------------------- #
# ★ Backward Compatible Wrapper
# --------------------------------------------------------------------------- #


class UltraStepExecutor(Generic[TContext]):
    """
    Backward compatible wrapper for ExecutorCore.

    This maintains the exact same API as the original UltraStepExecutor
    while using the new modular architecture underneath.
    """

    def __init__(
        self,
        *,
        enable_cache: bool = True,
        cache_size: int = 1_024,
        cache_ttl: int = 3_600,
        concurrency_limit: Optional[int] = None,
    ) -> None:
        # Validate parameters (backward compatibility)
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
        if concurrency_limit is not None and concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be positive if specified")

        # Create cache backend if enabled
        cache_backend = None
        if enable_cache:
            cache_backend = InMemoryLRUBackend(max_size=cache_size, ttl_s=cache_ttl)

        # Initialize the core executor
        self._core: ExecutorCore[Any] = ExecutorCore(
            cache_backend=cache_backend,
            concurrency_limit=concurrency_limit,
            enable_cache=enable_cache,
        )

        # Backward compatibility attributes
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._enable_cache = enable_cache

    async def _execute_complex_step(self, step: Any, data: Any, **kwargs: Any) -> StepResult:
        """Backward compatibility wrapper for _execute_complex_step."""
        # Provide default values for backward compatibility
        context = kwargs.get("context", None)
        resources = kwargs.get("resources", None)
        limits = kwargs.get("limits", None)
        stream = kwargs.get("stream", False)
        on_chunk = kwargs.get("on_chunk", None)
        breach_event = kwargs.get("breach_event", None)
        context_setter = kwargs.get("context_setter", None)

        return await self._core._execute_complex_step(
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            breach_event,
            context_setter,
        )

    async def execute_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        usage_limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        breach_event: Optional[Any] = None,
        result: Optional[Any] = None,  # Preserved for backward compatibility
        context_setter: Optional[Callable[["PipelineResult[Any]", Optional[Any]], None]] = None,
    ) -> StepResult:
        """Execute a single step (backward compatible API)."""
        return await self._core.execute(
            step,
            data,
            context=context,
            resources=resources,
            limits=usage_limits,
            stream=stream,
            on_chunk=on_chunk,
            breach_event=breach_event,
            result=result,
            context_setter=context_setter,
        )

    @cached_property
    def cache(self) -> Optional[Any]:
        """Expose cache for inspection (backward compatibility)."""
        if not self._enable_cache:
            return None
        # Create a wrapper that uses the same backend as the executor
        cache_wrapper = _LRUCache(max_size=self._cache_size, ttl=self._cache_ttl)
        # Replace the backend with the one from the executor
        cache_wrapper._backend = self._core._cache_backend  # type: ignore
        return cache_wrapper

    def _cache_key(self, frame: Any) -> str:
        """Generate cache key for a frame (backward compatibility)."""
        return self._core._cache_key_generator.generate_key(
            frame.step, frame.data, frame.context, frame.resources
        )

    def _hash_obj(self, obj: Any) -> str:
        """Hash any Python object deterministically (backward compatibility)."""
        return self._core._cache_key_generator._hash_obj(obj)

    def clear_cache(self) -> None:
        """Clear cache (backward compatibility)."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the task
            loop.create_task(self._core.clear_cache())
        except RuntimeError:
            # No event loop running, create a new one
            if self._core._cache_backend is not None:
                asyncio.run(self._core.clear_cache())


# Simple backward-compatibility object for cache key generation
class _Frame:
    """Minimal frame object for backward compatibility with cache tests."""

    def __init__(
        self,
        step: Any = None,
        data: Any = None,
        context: Any = None,
        resources: Any = None,
    ) -> None:
        self.step = step
        self.data = data
        self.context = context
        self.resources = resources


# --------------------------------------------------------------------------- #
# ★ Backward Compatibility Aliases
# --------------------------------------------------------------------------- #


class _LRUCache:
    """Backward compatibility wrapper for the old _LRUCache interface."""

    def __init__(self, max_size: int = 1024, ttl: int = 3600) -> None:
        # Adapt old parameter names to new ones
        self.max_size = max_size
        self.ttl = ttl
        self._backend = InMemoryLRUBackend(max_size=max_size, ttl_s=ttl)

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl < 0:
            raise ValueError("ttl must be non-negative")

    @property
    def _store(self) -> OrderedDict[str, tuple[StepResult, float]]:
        """Expose the store for backward compatibility."""
        return self._backend._store

    def get(self, key: str) -> Optional[StepResult]:
        """Get item from cache (sync version)."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Use run_until_complete to wait for the result
            return loop.run_until_complete(self._backend.get(key))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self._backend.get(key))

    def set(self, key: str, val: StepResult) -> None:
        """Set item in cache (sync version)."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Use run_until_complete to wait for the result
            loop.run_until_complete(self._backend.put(key, val, ttl_s=self._backend.ttl_s))
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(self._backend.put(key, val, ttl_s=self._backend.ttl_s))

    def clear(self) -> None:
        """Clear all cache items (sync version)."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Use run_until_complete to wait for the result
            loop.run_until_complete(self._backend.clear())
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(self._backend.clear())


class _UsageTracker:
    """Backward compatibility wrapper for the old _UsageTracker interface."""

    def __init__(self) -> None:
        self._meter = ThreadSafeMeter()

    @property
    def total_cost(self) -> float:
        """Get total cost (sync version)."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._meter.snapshot())
            cost, _, _ = task.result() if task.done() else asyncio.run(self._meter.snapshot())
            return cost
        except RuntimeError:
            cost, _, _ = asyncio.run(self._meter.snapshot())
            return cost

    @property
    def total_tokens(self) -> int:
        """Get total tokens (sync version)."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._meter.snapshot())
            _, prompt_tokens, completion_tokens = (
                task.result() if task.done() else asyncio.run(self._meter.snapshot())
            )
            return prompt_tokens + completion_tokens
        except RuntimeError:
            _, prompt_tokens, completion_tokens = asyncio.run(self._meter.snapshot())
            return prompt_tokens + completion_tokens

    async def get_current_totals(self) -> tuple[float, int]:
        """Get current totals (cost, tokens) - async version."""
        cost, prompt_tokens, completion_tokens = await self._meter.snapshot()
        return cost, prompt_tokens + completion_tokens

    def get_current_totals_sync(self) -> tuple[float, int]:
        """Get current totals (cost, tokens) - sync wrapper. Do NOT call from async code."""
        import asyncio

        return asyncio.run(self.get_current_totals())

    async def add(self, cost: float, tokens: int) -> None:
        """Add usage (async version)."""
        # Split tokens evenly between prompt and completion for compatibility
        prompt_tokens = tokens // 2
        completion_tokens = tokens - prompt_tokens
        await self._meter.add(cost, prompt_tokens, completion_tokens)

    async def guard(self, lim: UsageLimits, result: Optional["PipelineResult[Any]"] = None) -> None:
        """Guard against usage limits."""
        await self._meter.guard(lim)


# --------------------------------------------------------------------------- #
# ★ Public API
# --------------------------------------------------------------------------- #

__all__ = [
    # Core interfaces
    "ISerializer",
    "IHasher",
    "ICacheBackend",
    "IUsageMeter",
    "IAgentRunner",
    "IProcessorPipeline",
    "IValidatorRunner",
    "IPluginRunner",
    "ITelemetry",
    # Default implementations
    "OrjsonSerializer",
    "Blake3Hasher",
    "InMemoryLRUBackend",
    "ThreadSafeMeter",
    "DefaultAgentRunner",
    "DefaultProcessorPipeline",
    "DefaultValidatorRunner",
    "DefaultPluginRunner",
    "DefaultTelemetry",
    # Core executor
    "ExecutorCore",
    # Backward compatible wrapper
    "UltraStepExecutor",
    # Internal classes for testing
    # "_Frame", # Removed
    # Legacy aliases for backward compatibility
    "_LRUCache",
    "_UsageTracker",
]

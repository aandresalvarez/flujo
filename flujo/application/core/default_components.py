from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

from ...domain.models import PipelineResult, StepResult, UsageLimits
from ...domain.validation import ValidationResult
from ...exceptions import (
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PausedException,
    UsageLimitExceededError,
)
from ...infra import telemetry
from ...signature_tools import analyze_signature
from .context_manager import _accepts_param

if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, Mock  # pragma: no cover
else:  # pragma: no cover - mock types only used for isinstance checks in tests
    try:
        from unittest.mock import AsyncMock, MagicMock, Mock  # type: ignore
    except Exception:

        class Mock:  # minimal runtime fallbacks
            pass

        class MagicMock(Mock):
            pass

        class AsyncMock(Mock):
            pass


# -----------------------------
# Serialization / Hashing
# -----------------------------
class OrjsonSerializer:
    """Fast JSON serializer using orjson if available, unified with flujo.utils.serialization."""

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
        from flujo.utils.serialization import safe_serialize

        serialized_obj = safe_serialize(obj, mode="default")
        if self._use_orjson:
            return self._orjson.dumps(serialized_obj, option=self._orjson.OPT_SORT_KEYS)
        else:
            s = self._json.dumps(serialized_obj, sort_keys=True, separators=(",", ":"))
            return s.encode("utf-8")

    def deserialize(self, blob: bytes) -> Any:
        from flujo.utils.serialization import safe_deserialize

        if self._use_orjson:
            raw_data = self._orjson.loads(blob)
        else:
            raw_data = self._json.loads(blob.decode("utf-8"))
        return safe_deserialize(raw_data)


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


class DefaultCacheKeyGenerator:
    """Default cache key generator implementation."""

    def __init__(self, hasher: Any = None):
        self._hasher = hasher or Blake3Hasher()

    def generate_key(self, step: Any, data: Any, context: Any, resources: Any) -> str:
        step_name = getattr(step, "name", str(type(step).__name__))
        data_str = str(data) if data is not None else ""
        key_bytes = f"{step_name}:{data_str}".encode("utf-8")
        digest = self._hasher.digest(key_bytes)
        # Ensure return type is exactly str for static typing
        return str(digest)


# -----------------------------
# Caching / Usage
# -----------------------------
@dataclass
class _LRUCache:
    """LRU cache implementation with TTL support."""

    max_size: int = 1024
    ttl: int = 3600
    _store: OrderedDict[str, tuple[StepResult, float]] = field(
        init=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl < 0:
            raise ValueError("ttl must be non-negative")

    def set(self, key: str, value: StepResult) -> None:
        current_time = time.monotonic()
        while len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = (value, current_time)
        self._store.move_to_end(key)

    def get(self, key: str) -> Optional[StepResult]:
        if key not in self._store:
            return None
        value, timestamp = self._store[key]
        current_time = time.monotonic()
        if self.ttl > 0 and current_time - timestamp > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def clear(self) -> None:
        self._store.clear()


@dataclass
class InMemoryLRUBackend:
    """O(1) LRU cache with TTL support, async interface."""

    max_size: int = 1024
    ttl_s: int = 3600
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _store: OrderedDict[str, tuple[StepResult, float, int]] = field(
        init=False, default_factory=OrderedDict
    )

    async def get(self, key: str) -> Optional[StepResult]:
        async with self._lock:
            if key not in self._store:
                return None
            result, timestamp, access_count = self._store[key]
            current_time = time.monotonic()
            if current_time - timestamp > self.ttl_s:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            self._store[key] = (result, timestamp, access_count + 1)
            return result.model_copy(deep=True)

    async def put(self, key: str, value: StepResult, ttl_s: int) -> None:
        async with self._lock:
            current_time = time.monotonic()
            while len(self._store) >= self.max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, current_time, 0)
            self._store.move_to_end(key)

    async def clear(self) -> None:
        async with self._lock:
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

    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None) -> None:
        async with self._lock:
            if (
                limits.total_cost_usd_limit is not None
                and isinstance(limits.total_cost_usd_limit, (int, float))
                and self.total_cost_usd - limits.total_cost_usd_limit > 1e-9
            ):
                msg = (
                    f"Cost limit of ${limits.total_cost_usd_limit} exceeded "
                    f"(current: ${self.total_cost_usd})"
                )
                raise UsageLimitExceededError(
                    msg,
                    PipelineResult(
                        step_history=step_history or [], total_cost_usd=self.total_cost_usd
                    ),
                )

            total_tokens = self.prompt_tokens + self.completion_tokens
            if (
                limits.total_tokens_limit is not None
                and isinstance(limits.total_tokens_limit, (int, float))
                and total_tokens - limits.total_tokens_limit > 0
            ):
                msg = (
                    f"Token limit of {limits.total_tokens_limit} exceeded (current: {total_tokens})"
                )
                raise UsageLimitExceededError(
                    msg,
                    PipelineResult(
                        step_history=step_history or [], total_cost_usd=self.total_cost_usd
                    ),
                )

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._lock:
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens


# -----------------------------
# Runners
# -----------------------------
class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any:
        import inspect

        processor_list = (
            processors.prompt_processors if hasattr(processors, "prompt_processors") else processors
        )
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
                try:
                    telemetry.logfire.error(f"Prompt processor failed: {e}")
                except Exception:
                    pass
                raise e

        return processed_data

    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any:
        import inspect

        processor_list = (
            processors.output_processors if hasattr(processors, "output_processors") else processors
        )
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
                try:
                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                raise e

        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(
        self, validators: List[Any], data: Any, *, context: Any
    ) -> List[ValidationResult]:
        if not validators:
            return []

        validation_results: List[ValidationResult] = []
        for validator in validators:
            try:
                # Support both validator objects with .validate and bare callables
                validate_fn = getattr(validator, "validate", None) or validator
                # Prefer passing context when accepted; fall back to data-only
                try:
                    result = await validate_fn(data, context=context)
                except TypeError:
                    result = await validate_fn(data)
                if isinstance(result, ValidationResult):
                    validation_results.append(result)
                elif hasattr(result, "is_valid"):
                    feedback = getattr(result, "feedback", None)
                    if hasattr(feedback, "_mock_name"):
                        feedback = None

                    validator_name = getattr(validator, "name", None)
                    if hasattr(validator_name, "_mock_name") or validator_name is None:
                        validator_name = type(validator).__name__

                    validation_results.append(
                        ValidationResult(
                            is_valid=result.is_valid,
                            feedback=feedback,
                            validator_name=validator_name,
                        )
                    )
                else:
                    feedback_msg = (
                        f"Validator {type(validator).__name__} returned invalid result type"
                    )
                    validation_results.append(
                        ValidationResult(
                            is_valid=False,
                            feedback=feedback_msg,
                            validator_name=type(validator).__name__,
                        )
                    )
            except Exception as e:
                validation_results.append(
                    ValidationResult(
                        is_valid=False,
                        feedback=f"Validator {type(validator).__name__} failed: {e}",
                        validator_name=type(validator).__name__,
                    )
                )

        return validation_results


def _should_pass_context_to_plugin(context: Optional[Any], func: Callable[..., Any]) -> bool:
    if context is None:
        return False
    import inspect

    sig = inspect.signature(func)
    return any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context"
        for p in sig.parameters.values()
    )


def _should_pass_resources_to_plugin(resources: Optional[Any], func: Callable[..., Any]) -> bool:
    if resources is None:
        return False
    import inspect

    sig = inspect.signature(func)
    return any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "resources"
        for p in sig.parameters.values()
    )


class DefaultPluginRunner:
    """Default plugin runner implementation."""

    async def run_plugins(
        self,
        plugins: List[tuple[Any, int]],
        data: Any,
        *,
        context: Any,
        resources: Optional[Any] = None,
    ) -> Any:
        from ...domain.plugins import PluginOutcome

        processed_data = data
        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                plugin_kwargs: Dict[str, Any] = {}
                if _should_pass_context_to_plugin(context, plugin.validate):
                    plugin_kwargs["context"] = context
                if _should_pass_resources_to_plugin(resources, plugin.validate):
                    plugin_kwargs["resources"] = resources

                result = await plugin.validate(processed_data, **plugin_kwargs)

                if isinstance(result, PluginOutcome):
                    if not result.success:
                        return result
                    if result.new_solution is not None:
                        processed_data = result.new_solution
                    continue
                else:
                    processed_data = result

            except Exception as e:
                plugin_name = getattr(plugin, "name", type(plugin).__name__)
                telemetry.logfire.error(f"Plugin {plugin_name} failed: {e}")
                raise ValueError(f"Plugin {plugin_name} failed: {e}")

        return processed_data


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
        import inspect
        from ...application.core.context_manager import _should_pass_context

        if agent is None:
            raise RuntimeError("Agent is None")

        target_agent = getattr(agent, "_agent", agent)

        executable_func = None
        if stream:
            if hasattr(agent, "stream"):
                executable_func = getattr(agent, "stream")
            elif hasattr(target_agent, "stream"):
                executable_func = getattr(target_agent, "stream")
            elif hasattr(agent, "run"):
                executable_func = getattr(agent, "run")
            elif hasattr(target_agent, "run"):
                executable_func = getattr(target_agent, "run")
            elif callable(target_agent):
                executable_func = target_agent
            else:
                raise RuntimeError(f"Agent {type(agent).__name__} has no executable method")
        else:
            if hasattr(agent, "run"):
                executable_func = getattr(agent, "run")
            elif hasattr(target_agent, "run"):
                executable_func = getattr(target_agent, "run")
            elif callable(target_agent):
                executable_func = target_agent
            else:
                raise RuntimeError(f"Agent {type(agent).__name__} has no executable method")

        filtered_kwargs: Dict[str, Any] = {}

        if isinstance(executable_func, (Mock, MagicMock, AsyncMock)):
            filtered_kwargs.update(options)
            if context is not None:
                filtered_kwargs["context"] = context
            if resources is not None:
                filtered_kwargs["resources"] = resources
            if breach_event is not None:
                filtered_kwargs["breach_event"] = breach_event
        else:
            try:
                spec = analyze_signature(executable_func)
                if _should_pass_context(spec, context, executable_func):
                    filtered_kwargs["context"] = context
                if resources is not None and _accepts_param(executable_func, "resources"):
                    filtered_kwargs["resources"] = resources
                for key, value in options.items():
                    if value is not None and _accepts_param(executable_func, key):
                        filtered_kwargs[key] = value
                if breach_event is not None and _accepts_param(executable_func, "breach_event"):
                    filtered_kwargs["breach_event"] = breach_event
            except Exception:
                filtered_kwargs.update(options)
                if context is not None:
                    filtered_kwargs["context"] = context
                if resources is not None:
                    filtered_kwargs["resources"] = resources
                if breach_event is not None:
                    filtered_kwargs["breach_event"] = breach_event

        try:
            if stream:
                if inspect.isasyncgenfunction(executable_func):
                    async_generator = executable_func(payload, **filtered_kwargs)
                    chunks = []
                    async for chunk in async_generator:
                        chunks.append(chunk)
                        if on_chunk is not None:
                            await on_chunk(chunk)
                    if chunks:
                        if all(isinstance(chunk, str) for chunk in chunks):
                            return "".join(chunks)
                        elif all(isinstance(chunk, bytes) for chunk in chunks):
                            return b"".join(chunks)
                        else:
                            return str(chunks)
                    else:
                        return "" if on_chunk is None else chunks
                elif inspect.iscoroutinefunction(executable_func):
                    result = await executable_func(payload, **filtered_kwargs)
                    if hasattr(result, "__aiter__"):
                        chunks = []
                        async for chunk in result:
                            chunks.append(chunk)
                            if on_chunk is not None:
                                await on_chunk(chunk)
                        if chunks:
                            if all(isinstance(chunk, str) for chunk in chunks):
                                return "".join(chunks)
                            elif all(isinstance(chunk, bytes) for chunk in chunks):
                                return b"".join(chunks)
                            else:
                                return str(chunks)
                        else:
                            return "" if on_chunk is None else chunks
                    else:
                        if on_chunk is not None:
                            await on_chunk(result)
                        return result
                else:
                    result = executable_func(payload, **filtered_kwargs)
                    if on_chunk is not None:
                        await on_chunk(result)
                    return result
            else:
                if inspect.iscoroutinefunction(executable_func):
                    return await executable_func(payload, **filtered_kwargs)
                else:
                    result = executable_func(payload, **filtered_kwargs)
                    if inspect.iscoroutine(result):
                        return await result
                    return result
        except (
            PausedException,
            InfiniteFallbackError,
            InfiniteRedirectError,
            ContextInheritanceError,
        ) as e:
            raise e


# -----------------------------
# Telemetry
# -----------------------------
class DefaultTelemetry:
    """Default telemetry implementation."""

    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


__all__ = [
    # Serialization / hashing
    "OrjsonSerializer",
    "Blake3Hasher",
    "DefaultCacheKeyGenerator",
    # Caching / usage
    "_LRUCache",
    "InMemoryLRUBackend",
    "ThreadSafeMeter",
    # Runners
    "DefaultProcessorPipeline",
    "DefaultValidatorRunner",
    "DefaultPluginRunner",
    "DefaultAgentRunner",
    # Telemetry
    "DefaultTelemetry",
]

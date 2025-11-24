from __future__ import annotations

import asyncio
import hashlib
import time
import importlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Protocol

from ...domain.models import StepResult, UsageLimits
from ...domain.validation import ValidationResult
from ...exceptions import (
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PausedException,
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
            blob: bytes = self._orjson.dumps(serialized_obj, option=self._orjson.OPT_SORT_KEYS)
            return blob
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
            # hexdigest() returns str, but typing of _blake3 is dynamic; coerce explicitly
            return str(self._blake3.blake3(data).hexdigest())
        else:
            return hashlib.blake2b(data, digest_size=32).hexdigest()


class _HasherProtocol(Protocol):
    def digest(self, data: bytes) -> str: ...


class DefaultCacheKeyGenerator:
    """Default cache key generator implementation."""

    def __init__(self, hasher: _HasherProtocol | None = None):
        # Ensure the hasher provides a digest(bytes) -> str
        self._hasher: _HasherProtocol = hasher or Blake3Hasher()

    # -----------------------------
    # Helper methods (readability & testability)
    # -----------------------------
    def _get_agent(self, step: Any) -> Any:
        return getattr(step, "agent", None)

    def _get_agent_type(self, agent: Any) -> Optional[str]:
        if agent is None:
            return None
        try:
            return type(agent).__name__
        except Exception:
            return None

    def _get_agent_model_id(self, agent: Any) -> Optional[str]:
        if agent is None:
            return None
        try:
            return getattr(agent, "model_id", None)
        except Exception:
            return None

    def _get_agent_system_prompt(self, agent: Any) -> Optional[Any]:
        if agent is None:
            return None
        try:
            system_prompt = getattr(agent, "system_prompt", None)
            if system_prompt is None and hasattr(agent, "_system_prompt"):
                system_prompt = getattr(agent, "_system_prompt")
            return system_prompt
        except Exception:
            return None

    def _hash_text_sha256(self, text: Any) -> Optional[str]:
        try:
            import hashlib as _hashlib

            if text is None:
                return None
            return _hashlib.sha256(str(text).encode()).hexdigest()
        except Exception:
            return None

    def _get_processor_names(self, step: Any, attribute_name: str) -> List[str]:
        try:
            processors_obj = getattr(step, "processors", None)
            if processors_obj is None:
                return []
            candidate_list = getattr(processors_obj, attribute_name, [])
            if not isinstance(candidate_list, list):
                return []
            return [type(p).__name__ for p in candidate_list]
        except Exception:
            return []

    def _get_validator_names(self, step: Any) -> List[str]:
        try:
            validators = getattr(step, "validators", [])
            if not isinstance(validators, list):
                return []
            return [type(v).__name__ for v in validators]
        except Exception:
            return []

    def _build_step_section(self, step: Any) -> Dict[str, Any]:
        agent = self._get_agent(step)
        agent_type = self._get_agent_type(agent)
        model_id = self._get_agent_model_id(agent)
        system_prompt = self._get_agent_system_prompt(agent)
        system_prompt_hash = self._hash_text_sha256(system_prompt)

        return {
            "name": getattr(step, "name", str(type(step).__name__)),
            "agent": {
                "type": agent_type,
                "model_id": model_id,
                "system_prompt_sha256": system_prompt_hash,
            },
            "config": {
                "max_retries": getattr(getattr(step, "config", None), "max_retries", None),
                "timeout_s": getattr(getattr(step, "config", None), "timeout_s", None),
                "temperature": getattr(getattr(step, "config", None), "temperature", None),
            },
            "processors": {
                "prompt_processors": self._get_processor_names(step, "prompt_processors"),
                "output_processors": self._get_processor_names(step, "output_processors"),
            },
            "validators": self._get_validator_names(step),
        }

    def _build_payload(self, step: Any, data: Any, context: Any, resources: Any) -> Dict[str, Any]:
        from flujo.utils.serialization import safe_serialize as _safe_serialize

        return {
            "step": self._build_step_section(step),
            "data": _safe_serialize(data, mode="cache"),
            "context": _safe_serialize(context, mode="cache"),
            "resources": _safe_serialize(resources, mode="cache"),
        }

    def generate_key(self, step: Any, data: Any, context: Any, resources: Any) -> str:
        """Generate a collision-resistant cache key.

        Incorporates agent identity (type, model_id, system_prompt hash),
        step configuration, processors/validators, and a stable serialization of
        inputs to prevent collisions across logically distinct agents/steps.
        """
        try:
            # Only delegate to cache_step generator for real DSL Step objects
            from flujo.domain.dsl.step import Step as _DSLStep

            if isinstance(step, _DSLStep):
                from flujo.steps.cache_step import _generate_cache_key as _gen_alt

                key: str | None = _gen_alt(step, data, context, resources)
                if key is not None:
                    return key
                # If _gen_alt returns None, fall through to local generator
        except Exception:
            # Fall back to local generator on any error (including mocks)
            pass

        # Fallback: local robust key generation mirroring cache_step
        import hashlib as _hashlib
        import json as _json

        payload = self._build_payload(step, data, context, resources)
        try:
            blob = _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            return _hashlib.sha256(blob).hexdigest()
        except Exception:
            # Robust fallback: incorporate serialized inputs to preserve uniqueness
            step_name = getattr(step, "name", str(type(step).__name__))
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                data_repr = _safe_serialize(data, mode="cache")
            except Exception:
                data_repr = str(data)
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                ctx_repr = _safe_serialize(context, mode="cache")
            except Exception:
                ctx_repr = str(context)
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                res_repr = _safe_serialize(resources, mode="cache")
            except Exception:
                res_repr = str(resources)
            seed = _json.dumps(
                {
                    "name": step_name,
                    "data": data_repr,
                    "context": ctx_repr,
                    "resources": res_repr,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            return _hashlib.sha256(seed).hexdigest()


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
        # Compatibility no-op: enforcement now happens via proactive quota reservations.
        # This shim is retained for legacy callers/tests that still invoke guard().
        return None

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
                if isinstance(processed_data, str) and processed_data.isdigit():
                    try:
                        processed_data = int(processed_data)
                    except Exception:
                        pass
                if isinstance(proc, dict) and proc.get("type") == "callable":
                    fn = proc.get("callable")
                else:
                    fn = getattr(proc, "process", proc)
                if fn is None:
                    continue
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
                prior_data = processed_data
                if isinstance(proc, dict) and proc.get("type") == "callable":
                    fn = proc.get("callable")
                else:
                    fn = getattr(proc, "process", proc)
                if fn is None:
                    continue
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
                # If a processor returns a callable, eagerly invoke it with the latest data when possible.
                if callable(processed_data) and not inspect.iscoroutinefunction(processed_data):
                    try:
                        sig = inspect.signature(processed_data)
                        params = sig.parameters
                        if len(params) == 0:
                            processed_data = processed_data()
                        elif len(params) == 1:
                            processed_data = processed_data(prior_data)
                    except Exception:
                        # Leave callable as-is if invocation heuristics fail
                        pass
                try:
                    if isinstance(processed_data, dict) and "iteration" in processed_data:
                        ctr_val = None
                        try:
                            ctr = getattr(context, "counter", None) if context is not None else None
                            if isinstance(ctr, (int, float)):
                                ctr_val = int(ctr)
                            elif isinstance(ctr, str) and ctr.lstrip("-").isdigit():
                                ctr_val = int(ctr)
                        except Exception:
                            ctr_val = None
                        if ctr_val is not None:
                            processed_data["iteration"] = ctr_val + 1
                except Exception:
                    pass
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
    ) -> Any:
        import inspect
        from ...application.core.context_manager import _should_pass_context
        from flujo.domain.interfaces import get_skill_resolver

        if agent is None:
            raise RuntimeError("Agent is None")

        target_agent = getattr(agent, "_agent", agent)

        # Resolve string/dict agent specs via the skill registry or import path
        try:
            if isinstance(agent, str):
                reg = get_skill_resolver()
                entry = reg.get(agent) if reg is not None else None
                if entry is not None:
                    factory = entry.get("factory")
                    target_agent = factory() if callable(factory) else factory
                else:
                    module_path, _, attr = agent.partition(":")
                    mod = importlib.import_module(module_path)
                    target_agent = getattr(mod, attr) if attr else mod
            elif isinstance(agent, dict):
                skill_id = agent.get("id") or agent.get("path")
                params = agent.get("params", {}) if isinstance(agent, dict) else {}
                if skill_id:
                    reg = get_skill_resolver()
                    entry = reg.get(skill_id) if reg is not None else None
                    if entry is not None:
                        factory = entry.get("factory")
                        target_agent = factory(**params) if callable(factory) else factory
                    else:
                        mod_path, _, attr = skill_id.partition(":")
                        mod = importlib.import_module(mod_path)
                        obj = getattr(mod, attr) if attr else mod
                        target_agent = obj(**params) if callable(obj) else obj
        except Exception:
            target_agent = getattr(agent, "_agent", agent)

        # Minimal built-in fallbacks for dict specs when registry/import fails
        if isinstance(target_agent, dict) and isinstance(target_agent.get("id"), str):
            agent_id = target_agent.get("id")

            def _passthrough_fn(x: Any, **_k: Any) -> Any:
                return x

            def _stringify_fn(x: Any, **_k: Any) -> str:
                return str(x)

            if agent_id == "flujo.builtins.passthrough":
                target_agent = _passthrough_fn
            elif agent_id == "flujo.builtins.stringify":
                target_agent = _stringify_fn

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
            # Avoid passing mock contexts to mock agent functions to minimize overhead
            if context is not None:
                try:
                    from unittest.mock import Mock as _M

                    if not isinstance(context, _M):
                        filtered_kwargs["context"] = context
                except Exception:
                    filtered_kwargs["context"] = context
            if resources is not None:
                filtered_kwargs["resources"] = resources
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
            except Exception:
                filtered_kwargs.update(options)
                if context is not None:
                    filtered_kwargs["context"] = context
                if resources is not None:
                    filtered_kwargs["resources"] = resources

        try:
            if stream:
                # Case 1: async generator function
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
                        if all(isinstance(chunk, bytes) for chunk in chunks):
                            return b"".join(chunks)
                        return str(chunks)
                    return "" if on_chunk is None else chunks

                # Case 2: coroutine function that returns an async iterator
                if inspect.iscoroutinefunction(executable_func):
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
                            if all(isinstance(chunk, bytes) for chunk in chunks):
                                return b"".join(chunks)
                            return str(chunks)
                        return "" if on_chunk is None else chunks
                    # Not an iterator: treat as single result
                    if on_chunk is not None:
                        await on_chunk(result)
                    return result

                # Case 3: regular callable returning an async iterator/generator
                result = executable_func(payload, **filtered_kwargs)
                if hasattr(result, "__aiter__"):
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)
                        if on_chunk is not None:
                            await on_chunk(chunk)
                    if chunks:
                        if all(isinstance(chunk, str) for chunk in chunks):
                            return "".join(chunks)
                        if all(isinstance(chunk, bytes) for chunk in chunks):
                            return b"".join(chunks)
                        return str(chunks)
                    return "" if on_chunk is None else chunks
                # Fallback: single value passthrough
                if on_chunk is not None:
                    await on_chunk(result)
                return result

            # Non-streaming execution
            if inspect.iscoroutinefunction(executable_func):
                _res = await executable_func(payload, **filtered_kwargs)
            else:
                _res = executable_func(payload, **filtered_kwargs)
                if inspect.iscoroutine(_res):
                    _res = await _res

            # Detect mock objects in agent outputs
            # Mock detection: don't swallow exceptions; only guard imports
            def _is_mock(obj: Any) -> bool:
                try:
                    from unittest.mock import Mock as _M, MagicMock as _MM

                    try:
                        from unittest.mock import AsyncMock as _AM

                        if isinstance(obj, (_M, _MM, _AM)):
                            return True
                    except Exception:
                        if isinstance(obj, (_M, _MM)):
                            return True
                except Exception:
                    pass
                return bool(getattr(obj, "_is_mock", False) or hasattr(obj, "assert_called"))

            if _is_mock(_res):
                from ...exceptions import MockDetectionError as _MDE

                raise _MDE(f"Agent {type(agent).__name__} returned a Mock object")
            return _res
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

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        pass


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

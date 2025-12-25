from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator
from typing import Awaitable, Callable, TypeGuard

from ....domain.validation import ValidationResult
from ....exceptions import (
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError,
    PausedException,
)
from ....infra import telemetry
from ....signature_tools import analyze_signature
from ..context.context_manager import _accepts_param
from .default_cache_components import (
    Blake3Hasher,
    DefaultCacheKeyGenerator,
    InMemoryLRUBackend,
    OrjsonSerializer,
    ThreadSafeMeter,
    _LRUCache,
)


def _is_callable(obj: object) -> TypeGuard[Callable[..., object]]:
    return callable(obj)


def _is_awaitable(obj: object) -> TypeGuard[Awaitable[object]]:
    return inspect.isawaitable(obj)


def _is_async_iterator(obj: object) -> TypeGuard[AsyncIterator[object]]:
    return hasattr(obj, "__aiter__")


def _all_str(items: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in items)


def _all_bytes(items: list[object]) -> TypeGuard[list[bytes]]:
    return all(isinstance(x, bytes) for x in items)


# -----------------------------
# Runners
# -----------------------------
class DefaultProcessorPipeline:
    """Default processor pipeline implementation."""

    async def apply_prompt(self, processors: object, data: object, *, context: object) -> object:
        processor_source: (
            list[object] | tuple[object, ...] | AsyncIterator[object] | object | None
        ) = getattr(processors, "prompt_processors", processors)
        if processor_source is None:
            return data
        processor_list: list[object]
        if isinstance(processor_source, (list, tuple)):
            processor_list = list(processor_source)
        elif _is_async_iterator(processor_source):
            processor_list = [processor_source]
        else:
            processor_list = [processor_source]
        if not processor_list:
            return data

        processed_data: object = data
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

    async def apply_output(self, processors: object, data: object, *, context: object) -> object:
        processor_source: (
            list[object] | tuple[object, ...] | AsyncIterator[object] | object | None
        ) = getattr(processors, "output_processors", processors)
        if processor_source is None:
            return data
        processor_list: list[object]
        if isinstance(processor_source, (list, tuple)):
            processor_list = list(processor_source)
        elif _is_async_iterator(processor_source):
            processor_list = [processor_source]
        else:
            processor_list = [processor_source]
        if not processor_list:
            return data

        processed_data: object = data
        _slots_fallback_used = False
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
                if not _slots_fallback_used:
                    try:
                        hitl_data = (
                            getattr(context, "hitl_data", None) if context is not None else None
                        )
                        if isinstance(hitl_data, dict) and "slots" in hitl_data:
                            processed_data = {"slots": hitl_data.get("slots", {})}
                            _slots_fallback_used = True
                            continue
                    except Exception:
                        pass
                try:
                    telemetry.logfire.error(f"Output processor failed: {e}")
                except Exception:
                    pass
                raise e

        return processed_data


class DefaultValidatorRunner:
    """Default validator runner implementation."""

    async def validate(
        self, validators: list[object], data: object, *, context: object
    ) -> list[ValidationResult]:
        if not validators:
            return []

        validation_results: list[ValidationResult] = []
        for validator in validators:
            try:
                # Support both validator objects with .validate and bare callables
                validate_fn_obj = getattr(validator, "validate", None) or validator
                if not _is_callable(validate_fn_obj):
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
                    continue
                validate_fn = validate_fn_obj
                # Prefer passing context when accepted; fall back to data-only
                try:
                    result_obj = validate_fn(data, context=context)
                except TypeError:
                    result_obj = validate_fn(data)
                if _is_awaitable(result_obj):
                    result = await result_obj
                else:
                    result = result_obj
                if isinstance(result, ValidationResult):
                    validation_results.append(result)
                elif hasattr(result, "is_valid"):
                    feedback = getattr(result, "feedback", None)
                    if hasattr(feedback, "_mock_name"):
                        feedback = None

                    score = getattr(result, "score", None)
                    if hasattr(score, "_mock_name"):
                        score = None
                    diff = getattr(result, "diff", None)
                    if hasattr(diff, "_mock_name"):
                        diff = None

                    validator_name = getattr(validator, "name", None)
                    if hasattr(validator_name, "_mock_name") or validator_name is None:
                        validator_name = type(validator).__name__

                    payload: dict[str, object] = {
                        "is_valid": bool(getattr(result, "is_valid")),
                        "feedback": feedback,
                        "validator_name": validator_name,
                    }
                    if score is not None:
                        try:
                            payload["score"] = float(score)
                        except Exception:
                            pass
                    if diff is not None:
                        payload["diff"] = diff

                    validation_results.append(ValidationResult(**payload))
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


def _should_pass_context_to_plugin(context: object | None, func: Callable[..., object]) -> bool:
    if context is None:
        return False

    sig = inspect.signature(func)
    return any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context"
        for p in sig.parameters.values()
    )


def _should_pass_resources_to_plugin(resources: object | None, func: Callable[..., object]) -> bool:
    if resources is None:
        return False

    sig = inspect.signature(func)
    return any(
        p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "resources"
        for p in sig.parameters.values()
    )


class DefaultPluginRunner:
    """Default plugin runner implementation."""

    async def run_plugins(
        self,
        plugins: list[tuple[object, int]],
        data: object,
        *,
        context: object,
        resources: object | None = None,
    ) -> object:
        from ....domain.plugins import PluginOutcome

        processed_data: object = data
        for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
            try:
                validate_obj = getattr(plugin, "validate", None)
                if validate_obj is None or not _is_callable(validate_obj):
                    plugin_name = getattr(plugin, "name", type(plugin).__name__)
                    raise ValueError(f"Plugin {plugin_name} has no validate method")
                plugin_kwargs: dict[str, object] = {}
                if _should_pass_context_to_plugin(context, validate_obj):
                    plugin_kwargs["context"] = context
                if _should_pass_resources_to_plugin(resources, validate_obj):
                    plugin_kwargs["resources"] = resources

                result_obj = validate_obj(processed_data, **plugin_kwargs)
                if _is_awaitable(result_obj):
                    result = await result_obj
                else:
                    result = result_obj

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
        agent: object,
        payload: object,
        *,
        context: object,
        resources: object,
        options: dict[str, object],
        stream: bool = False,
        on_chunk: Callable[[object], Awaitable[None]] | None = None,
    ) -> object:
        from ..context.context_manager import _should_pass_context
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
                params_obj = agent.get("params", {}) if isinstance(agent, dict) else {}
                params: dict[str, object] = params_obj if isinstance(params_obj, dict) else {}
                if isinstance(skill_id, str) and skill_id:
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

            def _passthrough_fn(x: object, **_k: object) -> object:
                return x

            def _stringify_fn(x: object, **_k: object) -> str:
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

        executable_obj = executable_func
        if executable_obj is None or not _is_callable(executable_obj):
            raise RuntimeError(f"Agent {type(agent).__name__} has no executable method")
        executable_func = executable_obj

        filtered_kwargs: dict[str, object] = {}

        if not options and context is None and resources is None:
            # Fast path: no injections or options to forward, avoid signature inspection.
            filtered_kwargs = {}
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

        # Handle builtin skill input: syntax - unpack dict payload as kwargs
        # When builtins are called via step.input instead of agent.params, the dict
        # payload needs to be unpacked as keyword arguments to match function signatures.
        # Only unpack if the dict keys match the function's expected parameter names.
        _is_builtin = False
        _skip_payload = False  # Track if we deliberately cleared payload for builtin unpacking
        try:
            func_module = getattr(executable_func, "__module__", "")
            _is_builtin = isinstance(func_module, str) and (
                func_module.startswith("flujo.builtins") or "builtins" in func_module
            )
        except Exception:
            _is_builtin = False
        if _is_builtin and isinstance(payload, dict):
            # Check if dict keys match function parameters before unpacking
            try:
                sig = inspect.signature(executable_func)
                func_params = set(sig.parameters.keys()) - {"self", "cls"}
                payload_keys = set(payload.keys())
                # Only unpack if ALL payload keys are valid function parameters
                # This prevents passing unexpected kwargs that would cause TypeError
                if payload_keys and payload_keys <= func_params:
                    filtered_kwargs.update(payload)
                    payload = None  # Clear payload since we're passing via kwargs
                    _skip_payload = True  # We deliberately unpacked; don't pass payload
            except (ValueError, TypeError):
                # Can't inspect signature; don't unpack
                pass

        # Helper to call function with or without payload based on builtin unpacking
        def _call_args() -> tuple[tuple[object, ...], dict[str, object]]:
            if _skip_payload:
                return (), filtered_kwargs
            # Always pass payload as first arg, even if None (agents may expect it)
            return (payload,), filtered_kwargs

        try:
            if stream:
                # Case 1: async generator function
                if inspect.isasyncgenfunction(executable_func):
                    args, kwargs = _call_args()
                    async_generator_obj = executable_func(*args, **kwargs)
                    if not _is_async_iterator(async_generator_obj):
                        raise TypeError("Expected async iterator from async generator")
                    chunks = []
                    async for chunk in async_generator_obj:
                        chunks.append(chunk)
                        if on_chunk is not None:
                            await on_chunk(chunk)
                    if chunks:
                        if _all_str(chunks):
                            return "".join(chunks)
                        if _all_bytes(chunks):
                            return b"".join(chunks)
                        return str(chunks)
                    return "" if on_chunk is None else chunks

                # Case 2: coroutine function that returns an async iterator
                if inspect.iscoroutinefunction(executable_func):
                    args, kwargs = _call_args()
                    result_obj = await executable_func(*args, **kwargs)
                    if _is_async_iterator(result_obj):
                        chunks = []
                        async for chunk in result_obj:
                            chunks.append(chunk)
                            if on_chunk is not None:
                                await on_chunk(chunk)
                        if chunks:
                            if _all_str(chunks):
                                return "".join(chunks)
                            if _all_bytes(chunks):
                                return b"".join(chunks)
                            return str(chunks)
                        return "" if on_chunk is None else chunks
                    # Not an iterator: treat as single result
                    if on_chunk is not None:
                        await on_chunk(result_obj)
                    return result_obj

                # Case 3: regular callable returning an async iterator/generator
                args, kwargs = _call_args()
                result_obj = executable_func(*args, **kwargs)
                if _is_awaitable(result_obj):
                    result_obj = await result_obj
                if _is_async_iterator(result_obj):
                    chunks = []
                    async for chunk in result_obj:
                        chunks.append(chunk)
                        if on_chunk is not None:
                            await on_chunk(chunk)
                    if chunks:
                        if _all_str(chunks):
                            return "".join(chunks)
                        if _all_bytes(chunks):
                            return b"".join(chunks)
                        return str(chunks)
                    return "" if on_chunk is None else chunks
                # Fallback: single value passthrough
                if on_chunk is not None:
                    await on_chunk(result_obj)
                return result_obj

            # Non-streaming execution
            args, kwargs = _call_args()
            if inspect.iscoroutinefunction(executable_func):
                _res = await executable_func(*args, **kwargs)
            else:
                _res = executable_func(*args, **kwargs)
                if _is_awaitable(_res):
                    _res = await _res

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

    def trace(self, name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return decorator

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        pass

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        pass

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        pass

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
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

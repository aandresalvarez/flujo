from __future__ import annotations

from typing import Any, Optional, TypeVar, Set
import hashlib
import pickle  # nosec B403 - Used for fallback serialization of complex objects in cache keys
import json
from pydantic import Field

from flujo.domain.dsl import Step
from flujo.domain.models import PipelineContext
from flujo.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def cached(
        cls,
        wrapped_step: Step[Any, Any],
        cache_backend: Optional[CacheBackend] = None,
    ) -> "CacheStep[Any, Any]":
        """Create a CacheStep that wraps the given step with caching."""
        return cls(
            name=wrapped_step.name,
            wrapped_step=wrapped_step,
            cache_backend=cache_backend or InMemoryCache(),
        )


def _serialize_for_key(obj: Any, visited: Optional[Set[int]] = None, _is_root: bool = True) -> Any:
    """Best-effort conversion of arbitrary objects to cacheable structures for cache keys."""
    from flujo.utils.serialization import lookup_custom_serializer

    if obj is None:
        return None

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        # At the root, return a dict with all fields as placeholders for models/dicts
        if _is_root and hasattr(obj, "model_dump"):
            field_names = []
            try:
                d = obj.model_dump(mode="json")
                if "run_id" in d:
                    d.pop("run_id", None)
                field_names = list(d.keys())
            except Exception:
                if hasattr(obj, "__fields__"):
                    field_names = list(obj.__fields__.keys())
            # Special handling for Step: set 'agent' field to agent's class name
            if isinstance(obj, Step):
                step_result = {k: "<Step circular>" for k in field_names}
                if "agent" in field_names:
                    original_agent = getattr(obj, "agent", None)
                    step_result["agent"] = (
                        type(original_agent).__name__ if original_agent else "<unknown>"
                    )
                return step_result
            if isinstance(obj, PipelineContext):
                pipeline_ctx_placeholder = {k: "<PipelineContext circular>" for k in field_names}
                return pipeline_ctx_placeholder
            # Use actual class name for generic placeholder
            generic_placeholder = {k: f"<{obj.__class__.__name__} circular>" for k in field_names}
            return generic_placeholder
        if _is_root and isinstance(obj, dict):
            dict_placeholder = {k: "<dict circular>" for k in obj.keys()}
            return dict_placeholder
        # For fields, use a string placeholder
        if hasattr(obj, "model_dump"):
            # Special handling for Step: set 'agent' field to agent's class name
            if isinstance(obj, Step):
                original_agent = getattr(obj, "agent", None)
                if original_agent is not None:
                    return {"agent": type(original_agent).__name__}
            return f"<{obj.__class__.__name__} circular>"
        if isinstance(obj, dict):
            return "<dict circular>"
        if isinstance(obj, (list, tuple, set, frozenset)):
            return "<list circular>"
        # For custom types, use the custom serializer if available
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer is not None:
            try:
                return _serialize_for_key(custom_serializer(obj), visited, _is_root=False)
            except Exception:
                pass
        return "<circular>"

    visited.add(obj_id)
    try:
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer is not None:
            try:
                return _serialize_for_key(custom_serializer(obj), visited, _is_root=False)
            except Exception:
                pass

        if hasattr(obj, "model_dump") and isinstance(obj, PipelineContext):
            try:
                d = obj.model_dump(mode="json")
                d.pop("run_id", None)
                return {k: _serialize_for_key(v, visited, _is_root=False) for k, v in d.items()}
            except (ValueError, RecursionError):
                field_names = []
                if hasattr(obj, "__annotations__"):
                    field_names = list(obj.__annotations__.keys())
                elif hasattr(obj, "__fields__"):
                    field_names = list(obj.__fields__.keys())
                return {k: "<PipelineContext circular>" for k in field_names}
        if hasattr(obj, "model_dump") and isinstance(obj, Step):
            try:
                d = obj.model_dump(mode="json")
                if "run_id" in d:
                    d.pop("run_id", None)
                field_names = list(d.keys())
            except Exception:
                if hasattr(obj, "__fields__"):
                    field_names = list(obj.__fields__.keys())
                result_step_circular = {k: "<Step circular>" for k in field_names}
                if "agent" in field_names:
                    original_agent = getattr(obj, "agent", None)
                    result_step_circular["agent"] = (
                        type(original_agent).__name__ if original_agent else "<unknown>"
                    )
                return result_step_circular
        if hasattr(obj, "model_dump"):
            try:
                d = obj.model_dump(mode="json")
                if "run_id" in d:
                    d.pop("run_id", None)
                result_dict = {}
                for k, v in d.items():
                    if isinstance(v, (list, tuple)):
                        result_dict[k] = _serialize_list_for_key(list(v), visited)
                    elif isinstance(v, (set, frozenset)):
                        result_dict[k] = _serialize_list_for_key(
                            _sort_set_deterministically(v, visited), visited
                        )
                    else:
                        result_dict[k] = _serialize_for_key(v, visited, _is_root=False)
                return result_dict
            except (ValueError, RecursionError):
                if _is_root:
                    field_names = []
                    if hasattr(obj, "__annotations__"):
                        field_names = list(obj.__annotations__.keys())
                    elif hasattr(obj, "__fields__"):
                        field_names = list(obj.__fields__.keys())
                    return {k: f"<{obj.__class__.__name__} circular>" for k in field_names}
                return f"<{obj.__class__.__name__} circular>"
        if isinstance(obj, dict):
            d = dict(obj)
            if "run_id" in d and "initial_prompt" in d:
                d.pop("run_id", None)
            result: dict[Any, Any] = {}
            for k, v in d.items():
                # Always check for custom serializer for every value
                custom_serializer_v = lookup_custom_serializer(v)
                if custom_serializer_v is not None:
                    try:
                        result[k] = _serialize_for_key(
                            custom_serializer_v(v), visited, _is_root=False
                        )
                        continue
                    except Exception:
                        pass
                if hasattr(v, "model_dump"):
                    try:
                        v_dict = v.model_dump(mode="json")
                        if "run_id" in v_dict:
                            v_dict.pop("run_id", None)
                        result[k] = {
                            kk: _serialize_for_key(vv, visited, _is_root=False)
                            for kk, vv in v_dict.items()
                        }
                    except (ValueError, RecursionError):
                        field_names = []
                        if hasattr(v, "__annotations__"):
                            field_names = list(v.__annotations__.keys())
                        elif hasattr(v, "__fields__"):
                            field_names = list(v.__fields__.keys())
                        result[k] = {kk: f"<{v.__class__.__name__} circular>" for kk in field_names}
                elif isinstance(v, (list, tuple)):
                    result[k] = _serialize_list_for_key(list(v), visited)
                elif isinstance(v, (set, frozenset)):
                    result[k] = _serialize_list_for_key(
                        _sort_set_deterministically(v, visited), visited
                    )
                elif isinstance(v, dict):
                    result[k] = {
                        kk: _serialize_for_key(vv, visited, _is_root=False) for kk, vv in v.items()
                    }
                else:
                    result[k] = _serialize_for_key(v, visited, _is_root=False)
            return result
        if isinstance(obj, (list, tuple)):
            return _serialize_list_for_key(list(obj), visited)
        if isinstance(obj, (set, frozenset)):
            return _serialize_list_for_key(_sort_set_deterministically(obj, visited), visited)
        if callable(obj):
            return f"<callable {getattr(obj, '__name__', repr(obj))}>"
        return obj
    except Exception:
        return f"<unserializable: {type(obj).__name__}>"
    finally:
        # Remove object from visited set to prevent incorrect circular reference detection
        # in subsequent serialization paths within the same call
        visited.discard(obj_id)


def _sort_set_deterministically(
    obj_set: set[Any] | frozenset[Any], visited: Optional[Set[int]] = None
) -> list[Any]:
    """Sort a set or frozenset deterministically for cache key generation."""
    if visited is None:
        visited = set()

    try:
        # Try to sort by a stable representation
        return sorted(obj_set, key=lambda x: _get_stable_repr(x, visited))
    except (TypeError, ValueError):
        # Fallback: convert to string representation and sort
        return sorted(obj_set, key=lambda x: str(x))


def _get_stable_repr(obj: Any, visited: Optional[Set[int]] = None) -> str:
    """Get a stable string representation for sorting objects.

    This function includes circular reference detection to prevent infinite recursion
    and uses deterministic representations for complex objects.
    """
    if obj is None:
        return "None"

    # Initialize visited set for circular reference detection
    if visited is None:
        visited = set()

    # Get object id for circular reference detection
    obj_id = id(obj)
    if obj_id in visited:
        return f"<{type(obj).__name__} circular>"

    # Add current object to visited set
    visited.add(obj_id)

    try:
        if isinstance(obj, (int, float, str, bool)):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return f"[{','.join(_get_stable_repr(x, visited) for x in obj)}]"
        if isinstance(obj, dict):
            items = sorted((str(k), _get_stable_repr(v, visited)) for k, v in obj.items())
            return f"{{{','.join(f'{k}:{v}' for k, v in items)}}}"
        if isinstance(obj, (set, frozenset)):
            return f"{{{','.join(sorted(_get_stable_repr(x, visited) for x in obj))}}}"

        # For BaseModel objects, use a deterministic representation
        if hasattr(obj, "model_dump"):
            try:
                d = obj.model_dump(mode="json")
                # Remove run_id for consistency with other serialization functions
                if "run_id" in d:
                    d.pop("run_id", None)
                items = sorted((str(k), _get_stable_repr(v, visited)) for k, v in d.items())
                return f"{{{','.join(f'{k}:{v}' for k, v in items)}}}"
            except (ValueError, RecursionError):
                return f"<{type(obj).__name__} circular>"

        # For callable objects, use module and qualname for determinism
        if callable(obj):
            module = getattr(obj, "__module__", "<unknown>")
            qualname = getattr(obj, "__qualname__", repr(obj))
            return f"{module}.{qualname}"

        # For other objects, use type name and a hash of the object's content
        # This is more deterministic than id(obj) but still handles complex objects
        try:
            # Try to get a hash of the object's content
            if hasattr(obj, "__hash__") and obj.__hash__ is not None:
                return f"{type(obj).__name__}:{hash(obj)}"
            else:
                # For unhashable objects, use a hash of their string representation
                obj_repr = repr(obj)
                return f"{type(obj).__name__}:{hash(obj_repr)}"
        except Exception:
            # Final fallback: use type name only
            return f"{type(obj).__name__}"
    finally:
        # Remove current object from visited set when done
        visited.discard(obj_id)


def _serialize_list_for_key(obj_list: list[Any], visited: Optional[Set[int]] = None) -> list[Any]:
    if visited is None:
        visited = set()
    result_list: list[Any] = []
    for v in obj_list:
        obj_id = id(v)
        if obj_id in visited:
            if hasattr(v, "model_dump"):
                result_list.append(f"<{v.__class__.__name__} circular>")
            elif isinstance(v, dict):
                result_list.append("<dict circular>")
            elif isinstance(v, (list, tuple, set, frozenset)):
                result_list.append("<list circular>")
            else:
                result_list.append("<circular>")
            continue
        visited.add(obj_id)
        try:
            if hasattr(v, "model_dump"):
                d = v.model_dump(mode="json")
                if "run_id" in d:
                    d.pop("run_id", None)
                result_list.append(
                    {k: _serialize_for_key(val, visited, _is_root=False) for k, val in d.items()}
                )
            elif isinstance(v, dict):
                result_list.append(
                    {k: _serialize_for_key(val, visited, _is_root=False) for k, val in v.items()}
                )
            elif isinstance(v, (list, tuple)):
                result_list.append(_serialize_list_for_key(list(v), visited))
            elif isinstance(v, (set, frozenset)):
                result_list.append(
                    _serialize_list_for_key(_sort_set_deterministically(v, visited), visited)
                )
            else:
                from flujo.utils.serialization import lookup_custom_serializer

                custom_serializer = lookup_custom_serializer(v)
                if custom_serializer is not None:
                    try:
                        result_list.append(
                            _serialize_for_key(custom_serializer(v), visited, _is_root=False)
                        )
                        continue
                    except Exception:
                        pass
                result_list.append(_serialize_for_key(v, visited, _is_root=False))
        finally:
            visited.discard(obj_id)
    return result_list


def _create_step_fingerprint(step: Step[Any, Any]) -> dict[str, Any]:
    """Create a stable fingerprint for a step based on essential properties only.

    This function extracts only the deterministic, essential properties of a step
    that should affect cache key generation, avoiding object identity and
    internal state that may change between runs.
    """
    fingerprint = {
        "name": step.name,
        "agent_type": type(step.agent).__name__ if step.agent is not None else None,
        "config": {
            "max_retries": step.config.max_retries,
            "timeout_s": step.config.timeout_s,
            "temperature": step.config.temperature,
        },
        "plugins": [(type(plugin).__name__, priority) for plugin, priority in step.plugins],
        "validators": [type(validator).__name__ for validator in step.validators],
        "processors": {
            "prompt_processors": [
                type(proc).__name__ for proc in step.processors.prompt_processors
            ],
            "output_processors": [
                type(proc).__name__ for proc in step.processors.output_processors
            ],
        },
        "updates_context": step.updates_context,
        "persist_feedback_to_context": step.persist_feedback_to_context,
        "persist_validation_results_to": step.persist_validation_results_to,
    }
    return fingerprint


def _generate_cache_key(
    step: Step[Any, Any],
    data: Any,
    context: Any | None = None,
    resources: Any | None = None,
) -> Optional[str]:
    """Return a stable cache key for the step definition and input."""
    # Use stable step fingerprint instead of full step serialization
    step_fingerprint = _create_step_fingerprint(step)

    payload = {
        "step": step_fingerprint,
        "data": _serialize_for_key(data),
        "context": _serialize_for_key(context),
        "resources": _serialize_for_key(resources),
    }
    try:
        serialized = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(serialized).hexdigest()
    except Exception:
        try:
            serialized = pickle.dumps(payload)  # nosec B403 - Fallback serialization for cache keys
            digest = hashlib.sha256(serialized).hexdigest()
        except Exception:
            return None
    return f"{step.name}:{digest}"

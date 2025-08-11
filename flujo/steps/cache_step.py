from __future__ import annotations

from typing import Any, Optional, TypeVar, Set
import hashlib
import json
import logging
from pydantic import Field
from flujo.utils.hash import stable_digest

from flujo.domain.dsl import Step
from flujo.infra.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_complex(self) -> bool:
        # ✅ Override to mark as complex.
        return True

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


def _serialize_for_cache_key(
    obj: Any, visited: Optional[Set[int]] = None, _is_root: bool = True
) -> Any:
    """Best-effort conversion of arbitrary objects to cacheable structures for cache keys using unified serialization."""
    from flujo.utils.serialization import lookup_custom_serializer
    from flujo.domain.dsl import Step

    if obj is None:
        return None

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        # Special-case for Step agent field
        if isinstance(obj, Step):
            original_agent = getattr(obj, "agent", None)
            if original_agent is not None:
                return type(original_agent).__name__
            return "<Step circular>"
        # Special-case for Node (and similar): return placeholder string directly
        if obj.__class__.__name__ == "Node":
            return f"<{obj.__class__.__name__} circular>"
        if hasattr(obj, "model_dump"):
            return f"<{obj.__class__.__name__} circular>"
        if isinstance(obj, dict):
            return "<dict circular>"
        if isinstance(obj, (list, tuple, set, frozenset)):
            return "<list circular>"
        return "<circular>"

    # Handle primitives first - don't add to visited set
    if isinstance(obj, (int, float, str, bool)):
        return obj

    visited.add(obj_id)
    try:
        # Check for custom serializers first
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer is not None:
            try:
                return _serialize_for_cache_key(custom_serializer(obj), visited, _is_root=False)
            except Exception:
                pass

        # Handle models with model_dump
        if hasattr(obj, "model_dump"):
            try:
                d = obj.model_dump(mode="cache")
                if "run_id" in d:
                    d.pop("run_id", None)

                # Special-case for Step agent field
                if isinstance(obj, Step) and "agent" in d:
                    original_agent = getattr(obj, "agent", None)
                    d["agent"] = type(original_agent).__name__ if original_agent else "<unknown>"

                result_dict = {}
                for k in sorted(d.keys(), key=str):
                    v = d[k]
                    result_dict[k] = _serialize_for_cache_key(v, visited, _is_root=False)
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

        # Handle dictionaries
        if isinstance(obj, dict):
            d = dict(obj)
            if "run_id" in d and "initial_prompt" in d:
                d.pop("run_id", None)
            result = {}
            for k in sorted(d.keys(), key=str):
                v = d[k]
                # Check for custom serializers for values
                custom_serializer_v = lookup_custom_serializer(v)
                if custom_serializer_v is not None:
                    try:
                        result[k] = _serialize_for_cache_key(
                            custom_serializer_v(v), visited, _is_root=False
                        )
                        continue
                    except Exception:
                        pass

                if hasattr(v, "model_dump"):
                    try:
                        v_dict = v.model_dump(mode="cache")
                        if "run_id" in v_dict:
                            v_dict.pop("run_id", None)
                        result[k] = {
                            kk: _serialize_for_cache_key(v_dict[kk], visited, _is_root=False)
                            for kk in sorted(v_dict.keys(), key=str)
                        }
                    except (ValueError, RecursionError):
                        # Store a fallback value to simplify error handling
                        fallback_value = f"<unserializable: {type(v).__name__}>"
                        result[k] = fallback_value
                    except Exception:
                        # For other exceptions during model_dump, also use unserializable
                        fallback_value = f"<unserializable: {type(v).__name__}>"
                        result[k] = fallback_value
                else:
                    result[k] = _serialize_for_cache_key(v, visited, _is_root=False)
            return result

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return _serialize_list_for_key(list(obj), visited)

        # Handle sets
        if isinstance(obj, (set, frozenset)):
            return _serialize_list_for_key(_sort_set_deterministically(obj, visited), visited)

        # Handle callables
        if callable(obj):
            return f"<callable {getattr(obj, '__name__', repr(obj))}>"

        # For other types, try to get a stable representation
        try:
            if hasattr(obj, "__hash__") and obj.__hash__ is not None:
                try:
                    hash(obj)
                except Exception:
                    return f"<unserializable: {type(obj).__name__}>"
            return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"

    finally:
        visited.discard(obj_id)


def _sort_set_deterministically(
    obj_set: set[Any] | frozenset[Any], visited: Optional[Set[int]] = None
) -> list[Any]:
    """Sort a set or frozenset deterministically for cache key generation using unified serialization."""
    from flujo.utils.serialization import safe_serialize

    if visited is None:
        visited = set()

    try:
        # Try to sort by a stable representation using unified serialization
        return sorted(obj_set, key=lambda x: str(safe_serialize(x, mode="cache")))
    except (TypeError, ValueError):
        # Fallback: convert to string representation and sort
        return sorted(obj_set, key=lambda x: str(x))


def _get_sorted_keys(dictionary: dict[Any, Any]) -> list[Any]:
    """Get sorted keys from a dictionary for deterministic cache key generation."""
    return sorted(dictionary.keys(), key=str)


def _get_stable_repr(obj: Any, visited: Optional[Set[int]] = None) -> str:
    """Get a stable string representation for sorting objects."""
    if obj is None:
        return "None"

    # Initialize visited set for circular reference detection
    if visited is None:
        visited = set()

    # Get object id for circular reference detection
    obj_id = id(obj)
    if obj_id in visited:
        return f"<{type(obj).__name__} circular>"

    # Handle primitives without adding to visited
    if isinstance(obj, (int, float, str, bool)):
        return str(obj)

    # Add current object to visited set
    visited.add(obj_id)

    try:
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
                d = obj.model_dump(mode="cache")
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
    """Serialize list for cache key."""
    from flujo.utils.serialization import lookup_custom_serializer

    if visited is None:
        visited = set()

    result_list: list[Any] = []
    for v in obj_list:
        # Only check for circular references for container/object types
        if isinstance(v, (int, float, str, bool, type(None))):
            result_list.append(v)
            continue

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
            # Check for custom serializers first
            custom_serializer = lookup_custom_serializer(v)
            if custom_serializer is not None:
                try:
                    result_list.append(
                        _serialize_for_cache_key(custom_serializer(v), visited, _is_root=False)
                    )
                    continue
                except Exception:
                    pass

            if hasattr(v, "model_dump"):
                try:
                    d = v.model_dump(mode="cache")
                    if "run_id" in d:
                        d.pop("run_id", None)
                    result_list.append(
                        {
                            k: _serialize_for_cache_key(d[k], visited, _is_root=False)
                            for k in sorted(d.keys(), key=str)
                        }
                    )
                except Exception:
                    # For exceptions during model_dump, use unserializable fallback
                    result_list.append(f"<unserializable: {type(v).__name__}>")
            elif isinstance(v, dict):
                result_list.append(
                    {
                        k: _serialize_for_cache_key(v[k], visited, _is_root=False)
                        for k in sorted(v.keys(), key=str)
                    }
                )
            elif isinstance(v, (list, tuple)):
                result_list.append(_serialize_list_for_key(list(v), visited))
            elif isinstance(v, (set, frozenset)):
                result_list.append(
                    _serialize_list_for_key(_sort_set_deterministically(v, visited), visited)
                )
            else:
                result_list.append(_serialize_for_cache_key(v, visited, _is_root=False))

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
        "data": _serialize_for_cache_key(data),
        "context": _serialize_for_cache_key(context),
        "resources": _serialize_for_cache_key(resources),
    }
    try:
        serialized = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(serialized).hexdigest()
    except (TypeError, ValueError):
        # Handle unhashable types more gracefully
        try:
            # Try to create a more robust serialization
            safe_payload = {}
            for key, value in payload.items():
                try:
                    safe_payload[key] = _serialize_for_cache_key(value)
                except Exception:
                    # If serialization fails for a specific field, use a fallback
                    safe_payload[key] = f"<unserializable_{key}: {type(value).__name__}>"

            serialized = json.dumps(safe_payload, sort_keys=True).encode()
            digest = hashlib.sha256(serialized).hexdigest()
        except Exception:
            # Final fallback: use stable hashing helper (pickle-free)
            try:
                digest = stable_digest(payload)
            except Exception as e:
                logging.debug("Stable digest fallback failed: %s", e)
                return None
    return f"{step.name}:{digest}"

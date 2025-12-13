"""State serialization utilities with hashing and caching.

This module isolates low-level serialization/deserialization, hashing, and
change-detection concerns from StateManager to improve separation of concerns
and testability.
"""

from __future__ import annotations

import hashlib
import json
import itertools
from typing import Generic, Optional, Type, TypeVar

from flujo.domain.models import BaseModel, PipelineContext, StepResult
from flujo.type_definitions.common import JSONObject
from flujo.state.backends.base import _serialize_for_json

ContextT = TypeVar("ContextT", bound=BaseModel)


class StateSerializer(Generic[ContextT]):
    """Handles context and step-history serialization with change detection."""

    def __init__(self) -> None:
        # Cache for serialization results to avoid redundant work
        self._serialization_cache: dict[str, JSONObject] = {}
        self._context_hash_cache: dict[str, str] = {}

    # -------------------------- Hashing and cache --------------------------

    def compute_context_hash(self, context: Optional[ContextT]) -> str:
        if context is None:
            return "none"

        # Remove auto-generated fields that shouldn't affect change detection
        fields_to_exclude = {
            "run_id",
            "created_at",
            "updated_at",
            "pipeline_id",
            "pipeline_name",
            "pipeline_version",
        }

        data_items: list[tuple[str, object]] = []
        raw_mapping: dict[str, object] | None = None

        # Optimization: Track size estimation during extraction to avoid double iteration
        estimated_size = 0
        is_large = False

        if isinstance(context, BaseModel):
            # Fast path for Pydantic models
            for name in type(context).model_fields:
                if name in fields_to_exclude:
                    continue
                val = getattr(context, name)
                data_items.append((name, val))
                # Heuristic: check if we have many fields
                estimated_size += 1
        else:
            # Fallback for dicts or other types
            try:
                raw_mapping = (
                    context.model_dump() if hasattr(context, "model_dump") else dict(context)
                )
                for k, v in raw_mapping.items():
                    if k not in fields_to_exclude:
                        data_items.append((k, v))
                        estimated_size += 1
            except Exception as e:
                # CRITICAL: Do not swallow control flow exceptions
                from flujo.exceptions import ControlFlowError

                if isinstance(e, ControlFlowError):
                    raise
                return "error_hashing_context"

        # Heuristic: Check if context is "large" without expensive string conversion
        # Check 1: Too many top-level fields
        if estimated_size > 10:
            is_large = True
        else:
            # Check 2: Check for large container values (only if few fields)
            for _, v in data_items:
                if isinstance(v, (list, dict, str, bytes, tuple, set)):
                    # Check length of container directly, avoid str(v) which is O(N)
                    if len(v) > 100:
                        is_large = True
                        break

        def _fingerprint_scalar(value: object) -> str:
            type_name = type(value).__name__

            if isinstance(value, (list, tuple, dict, set)):
                return f"{type_name}:{len(value)}"

            if isinstance(value, str):
                return f"str:{len(value)}:{value[:20]}"

            if isinstance(value, bytes):
                return f"bytes:{len(value)}:{value[:8].hex()}"

            if isinstance(value, (int, float, bool)) or value is None:
                return f"{type_name}:{repr(value)}"

            try:
                rep = repr(value)
            except Exception:
                rep = ""
            rep = rep[:20] if rep else f"<{type_name}>"
            return f"{type_name}:{rep}"

        def _fingerprint_container(value: object) -> str:
            if isinstance(value, (str, bytes)):
                return _fingerprint_scalar(value)

            if isinstance(value, dict):
                # Optimization: Avoid sorting all keys if dict is huge
                if len(value) > 20:
                    # Just fingerprint size for huge dicts to be fast
                    return f"dict:{len(value)}"

                sample_keys = sorted(value.keys())[:5]
                samples: list[str] = []
                for key in sample_keys:
                    try:
                        samples.append(f"{key}:{_fingerprint_scalar(value[key])}")
                    except Exception:
                        samples.append(f"{key}:<unavailable>")
                return f"{len(value)}:{'|'.join(samples)}"

            if isinstance(value, (list, tuple)):
                # Optimization: Just take first 5 items
                sample_items = list(itertools.islice(value, 5))
                sample_fingerprints = ",".join(_fingerprint_scalar(item) for item in sample_items)
                return f"{len(value)}:{sample_fingerprints}"

            if isinstance(value, set):
                # Sets are unordered, so we must sort to be deterministic, but limit to 5
                # Optimization: If set is huge, just use size
                if len(value) > 20:
                    return f"set:{len(value)}"
                sample_items = sorted((_fingerprint_scalar(item) for item in value))[:5]
                return f"{len(value)}:{','.join(sample_items)}"

            return _fingerprint_scalar(value)

        if is_large:
            # For large contexts, use a simpler hash to avoid expensive JSON serialization
            hash_input = []
            # Sort by key to ensure deterministic order
            data_items.sort(key=lambda x: x[0])
            for key, value in data_items:
                if isinstance(value, (str, list, dict, bytes, tuple, set)):
                    val_repr = _fingerprint_container(value)
                else:
                    val_repr = _fingerprint_scalar(value)
                hash_input.append(f"{key}:{type(value).__name__}:{val_repr}")
            context_str = "|".join(hash_input)
            return hashlib.md5(context_str.encode()).hexdigest()

        if raw_mapping is None:
            # Use cache-friendly serialization path only for small contexts to avoid overhead
            raw_mapping = context.model_dump() if hasattr(context, "model_dump") else {}
        filtered_data = {k: v for k, v in raw_mapping.items() if k not in fields_to_exclude}

        normalized = _serialize_for_json(filtered_data)
        context_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

        return hashlib.md5(context_str.encode()).hexdigest()

    def should_serialize_context(self, context: Optional[ContextT], run_id: str) -> bool:
        if context is None:
            return False
        current_hash = self.compute_context_hash(context)
        cached_hash = self._context_hash_cache.get(run_id)
        if cached_hash != current_hash:
            self._context_hash_cache[run_id] = current_hash
            return True
        return False

    def _create_cache_key(self, run_id: str, context_hash: str) -> str:
        return f"{run_id}|{context_hash}"

    def _cache_get_by_hash(self, run_id: str, context_hash: str) -> Optional[JSONObject]:
        return self._serialization_cache.get(self._create_cache_key(run_id, context_hash))

    def _cache_put_by_hash(self, run_id: str, context_hash: str, serialized: JSONObject) -> None:
        if len(self._serialization_cache) >= 100:
            self._serialization_cache.pop(next(iter(self._serialization_cache)))
        self._serialization_cache[self._create_cache_key(run_id, context_hash)] = serialized

    def get_cached_serialization(
        self, context: Optional[ContextT], run_id: str
    ) -> Optional[JSONObject]:
        if context is None:
            return None
        context_hash = self.compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)
        return self._serialization_cache.get(cache_key)

    def cache_serialization(
        self, context: Optional[ContextT], run_id: str, serialized: JSONObject
    ) -> None:
        if context is None:
            return
        context_hash = self.compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)
        # Limit cache size to prevent memory leaks (simple FIFO)
        if len(self._serialization_cache) >= 100:
            self._serialization_cache.pop(next(iter(self._serialization_cache)))
        self._serialization_cache[cache_key] = serialized

    def clear_cache(self, run_id: Optional[str] = None) -> None:
        """Clear serialization cache globally or for a specific run_id."""
        if run_id is None:
            self._serialization_cache.clear()
            self._context_hash_cache.clear()
            return
        # Remove entries matching run_id
        keys_to_remove = []
        prefix = f"{run_id}|"
        for key in list(self._serialization_cache.keys()):
            if key.startswith(prefix):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._serialization_cache.pop(key, None)
        self._context_hash_cache.pop(run_id, None)

    # ---------------------------- Serialization ----------------------------

    def serialize_context_full(self, context: ContextT) -> JSONObject:
        # Use Pydantic dump for comprehensive state
        data = context.model_dump()
        return data if isinstance(data, dict) else {}

    def serialize_context_minimal(self, context: ContextT) -> JSONObject:
        # Minimal set used for optimized persistence paths
        data: JSONObject = {
            "initial_prompt": getattr(context, "initial_prompt", ""),
            "pipeline_id": getattr(context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(context, "pipeline_version", "latest"),
            "run_id": getattr(context, "run_id", ""),
        }
        # Optionally include common fields if present
        for field_name in [
            "total_steps",
            "error_message",
            "status",
            "current_step",
            "last_error",
            "metadata",
            "created_at",
            "updated_at",
        ]:
            if hasattr(context, field_name):
                data[field_name] = getattr(context, field_name, None)
        return data

    def serialize_context_for_state(
        self, context: Optional[ContextT], run_id: str
    ) -> Optional[JSONObject]:
        if context is None:
            return None
        # Compute hash once and check caches
        current_hash = self.compute_context_hash(context)
        cached_hash = self._context_hash_cache.get(run_id)

        # If we already have a cached full serialization for this (run, hash), reuse it
        cached_full = self._cache_get_by_hash(run_id, current_hash)
        if cached_full is not None:
            return cached_full

        # Hash not seen for this run yet, or it changed -> produce FULL serialization and cache it
        if cached_hash != current_hash:
            self._context_hash_cache[run_id] = current_hash
            full = self.serialize_context_full(context)
            self._cache_put_by_hash(run_id, current_hash, full)
            return full

        # Hash unchanged but no cached full exists (e.g., hash primed elsewhere): return minimal
        return self.serialize_context_minimal(context)

    def serialize_step_history_full(
        self, step_history: Optional[list[StepResult]]
    ) -> list[JSONObject]:
        out: list[JSONObject] = []
        if not step_history:
            return out
        for step_result in step_history:
            try:
                out.append(step_result.model_dump())
            except Exception:
                continue
        return out

    def serialize_step_history_minimal(
        self, step_history: Optional[list[StepResult]]
    ) -> list[JSONObject]:
        """Serialize only the most recent step to avoid quadratic growth."""
        if not step_history:
            return []
        latest = step_history[-1]
        try:
            return [
                {
                    "name": latest.name,
                    "output": latest.output,
                    "success": latest.success,
                    "cost_usd": latest.cost_usd,
                    "token_counts": latest.token_counts,
                    "attempts": latest.attempts,
                    "latency_s": latest.latency_s,
                    "feedback": latest.feedback,
                }
            ]
        except Exception:
            return []

    # -------------------------- Deserialization ---------------------------

    def deserialize_context(
        self, data: object, context_model: Optional[Type[ContextT]] = None
    ) -> Optional[ContextT]:
        if data is None:
            return None
        try:
            if context_model is not None:
                return context_model.model_validate(data)
            # Fallback to PipelineContext when no specific model provided
            return PipelineContext.model_validate(data)  # type: ignore[return-value]
        except Exception:
            return None

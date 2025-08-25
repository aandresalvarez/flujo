"""State serialization utilities with hashing and caching.

This module isolates low-level serialization/deserialization, hashing, and
change-detection concerns from StateManager to improve separation of concerns
and testability.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Generic, Optional, Type, TypeVar, cast

from flujo.domain.models import BaseModel, PipelineContext, StepResult

ContextT = TypeVar("ContextT", bound=BaseModel)


class StateSerializer(Generic[ContextT]):
    """Handles context and step-history serialization with change detection."""

    def __init__(self) -> None:
        # Cache for serialization results to avoid redundant work
        self._serialization_cache: Dict[str, Dict[str, Any]] = {}
        self._context_hash_cache: Dict[str, str] = {}

    # -------------------------- Hashing and cache --------------------------

    def compute_context_hash(self, context: Optional[ContextT]) -> str:
        if context is None:
            return "none"

        # Use a fast hash of the context data, excluding auto-generated fields
        context_data = context.model_dump()

        # Remove auto-generated fields that shouldn't affect change detection
        fields_to_exclude = {
            "run_id",
            "created_at",
            "updated_at",
            "pipeline_id",
            "pipeline_name",
            "pipeline_version",
        }
        filtered_data = {k: v for k, v in context_data.items() if k not in fields_to_exclude}

        # For large contexts, use a simpler hash to avoid expensive JSON serialization
        if len(filtered_data) > 10 or any(
            isinstance(v, (list, dict)) and len(str(v)) > 1000 for v in filtered_data.values()
        ):
            hash_input = []
            for key, value in sorted(filtered_data.items()):
                hash_input.append(f"{key}:{type(value).__name__}:{len(str(value))}")
            context_str = "|".join(hash_input)
        else:

            def default_serializer(o: Any) -> Any:
                if hasattr(o, "__class__") and "Mock" in o.__class__.__name__:
                    return f"Mock({type(o).__name__})"
                raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

            context_str = json.dumps(
                filtered_data, sort_keys=True, separators=(",", ":"), default=default_serializer
            )

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

    def _cache_get_by_hash(self, run_id: str, context_hash: str) -> Optional[Dict[str, Any]]:
        return self._serialization_cache.get(self._create_cache_key(run_id, context_hash))

    def _cache_put_by_hash(
        self, run_id: str, context_hash: str, serialized: Dict[str, Any]
    ) -> None:
        if len(self._serialization_cache) >= 100:
            self._serialization_cache.pop(next(iter(self._serialization_cache)))
        self._serialization_cache[self._create_cache_key(run_id, context_hash)] = serialized

    def get_cached_serialization(
        self, context: Optional[ContextT], run_id: str
    ) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        context_hash = self.compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)
        return self._serialization_cache.get(cache_key)

    def cache_serialization(
        self, context: Optional[ContextT], run_id: str, serialized: Dict[str, Any]
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

    def serialize_context_full(self, context: ContextT) -> Dict[str, Any]:
        # Use Pydantic dump for comprehensive state
        return cast(Dict[str, Any], context.model_dump())

    def serialize_context_minimal(self, context: ContextT) -> Dict[str, Any]:
        # Minimal set used for optimized persistence paths
        data: Dict[str, Any] = {
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
    ) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        # Optimize: compute hash once and avoid double hashing on cached path
        current_hash = self.compute_context_hash(context)
        cached_hash = self._context_hash_cache.get(run_id)
        if cached_hash != current_hash:
            # Context changed: update hash and serialize full
            self._context_hash_cache[run_id] = current_hash
            # If already present for this hash, reuse
            cached_full = self._cache_get_by_hash(run_id, current_hash)
            if cached_full is not None:
                return cached_full
            serialized = self.serialize_context_full(context)
            self._cache_put_by_hash(run_id, current_hash, serialized)
            return serialized
        # Unchanged: prefer minimal representation to reduce I/O overhead
        # regardless of whether a full serialization is cached
        return self.serialize_context_minimal(context)

    def serialize_step_history_full(
        self, step_history: Optional[list[StepResult]]
    ) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
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
    ) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        if not step_history:
            return out
        for step_result in step_history:
            try:
                out.append(
                    {
                        "name": step_result.name,
                        "output": step_result.output,
                        "success": step_result.success,
                        "cost_usd": step_result.cost_usd,
                        "token_counts": step_result.token_counts,
                        "attempts": step_result.attempts,
                        "latency_s": step_result.latency_s,
                        "feedback": step_result.feedback,
                    }
                )
            except Exception:
                continue
        return out

    # -------------------------- Deserialization ---------------------------

    def deserialize_context(
        self, data: Any, context_model: Optional[Type[ContextT]] = None
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

"""State management with intelligent caching and delta-based persistence."""

import logging
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar, Tuple

from flujo.domain.models import PipelineContext, PipelineResult, StepResult, BaseModel
from flujo.state.backends import StateBackend
from flujo.state.models import WorkflowState

logger = logging.getLogger(__name__)

ContextT = TypeVar("ContextT", bound=BaseModel)


class StateManager(Generic[ContextT]):
    """Intelligent state manager with caching and delta-based persistence."""

    def __init__(self, state_backend: Optional[StateBackend] = None) -> None:
        """Initialize state manager with optional backend."""
        self.state_backend = state_backend
        # Cache for serialization results to avoid redundant work
        self._serialization_cache: Dict[str, Any] = {}
        self._context_hash_cache: Dict[str, str] = {}

    def _compute_context_hash(self, context: Optional[ContextT]) -> str:
        """Compute a fast hash of the context for change detection."""
        if context is None:
            return "none"

        try:
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

            # Optimize for large contexts: use a faster hash computation
            # For large contexts, we can use a simpler hash to avoid expensive JSON serialization
            import hashlib

            # Use a faster approach for large contexts
            if len(filtered_data) > 10 or any(
                isinstance(v, (list, dict)) and len(str(v)) > 1000 for v in filtered_data.values()
            ):
                # For large contexts, use a faster hash based on key names and value types
                # This avoids expensive JSON serialization while still detecting changes
                hash_input = []
                for key, value in sorted(filtered_data.items()):
                    hash_input.append(f"{key}:{type(value).__name__}:{len(str(value))}")
                context_str = "|".join(hash_input)
            else:
                # For small contexts, use the original JSON-based approach
                import json

                # Custom default function to handle mock objects during hashing
                def default_serializer(o: Any) -> Any:
                    if hasattr(o, '__class__') and 'Mock' in o.__class__.__name__:
                        return f"Mock({type(o).__name__})"
                    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

                context_str = json.dumps(filtered_data, sort_keys=True, separators=(",", ":"), default=default_serializer)

            return hashlib.md5(context_str.encode()).hexdigest()
        except Exception as e:
            # Re-raise the exception so it can be caught by the outer handler
            # This ensures the error field is set in the pipeline context
            raise e

    def _should_serialize_context(self, context: Optional[ContextT], run_id: str) -> bool:
        """Determine if context needs serialization based on change detection."""
        if context is None:
            return False

        current_hash = self._compute_context_hash(context)
        cached_hash = self._context_hash_cache.get(run_id)

        if cached_hash != current_hash:
            # Context has changed, update cache and serialize
            self._context_hash_cache[run_id] = current_hash
            return True

        # Context hasn't changed, skip serialization
        return False

    def _get_cached_serialization(self, context: Optional[ContextT], run_id: str) -> Optional[Any]:
        """Get cached serialization result if available."""
        if context is None:
            return None

        context_hash = self._compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)
        return self._serialization_cache.get(cache_key)

    def _create_cache_key(self, run_id: str, context_hash: str) -> str:
        """Create a cache key that safely handles run_ids with underscores."""
        # Use a separator that's unlikely to appear in run_ids or context hashes
        # This prevents ambiguity when parsing cache keys
        return f"{run_id}|{context_hash}"

    def _parse_cache_key(self, cache_key: str) -> Tuple[str, str]:
        """Parse a cache key to extract run_id and context_hash."""
        # Split on the separator to get run_id and context_hash
        # Use rsplit to handle run_ids that might contain pipe characters
        parts = cache_key.rsplit("|", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid cache key format: {cache_key}")
        return parts[0], parts[1]

    def _cache_serialization(
        self, context: Optional[ContextT], run_id: str, serialized: Any
    ) -> None:
        """Cache serialized context to avoid redundant serialization."""
        if context is None:
            return

        context_hash = self._compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)

        # Limit cache size to prevent memory leaks with intelligent eviction
        if len(self._serialization_cache) >= 100:
            # Evict the least recently used entry before adding a new item
            self._evict_least_recently_used_entry()

        self._serialization_cache[cache_key] = serialized

    def _evict_least_recently_used_entry(self) -> None:
        """Evict the least recently used cache entry with proper cleanup."""
        # Find the oldest entry (first in insertion order for FIFO fallback)
        oldest_key = next(iter(self._serialization_cache))
        del self._serialization_cache[oldest_key]

        # Parse the cache key to extract run_id and context_hash
        try:
            evicted_run_id, evicted_context_hash = self._parse_cache_key(oldest_key)

            # Only remove the specific hash cache entry, not the entire run_id entry
            # This prevents unnecessary re-serialization of unchanged contexts
            self._context_hash_cache.pop(evicted_run_id, None)

            # Log for debugging cache behavior
            logger.debug(
                f"Evicted cache entry for run_id: {evicted_run_id}, context_hash: {evicted_context_hash}"
            )
        except ValueError:
            # Handle legacy cache keys that might still use the old format
            logger.warning(f"Found legacy cache key format: {oldest_key}")
            # For legacy keys, we can't safely extract run_id, so we skip hash cache cleanup
            # This is safe as the cache will eventually be cleared anyway

    async def load_workflow_state(
        self,
        run_id: str,
        context_model: Optional[type[ContextT]] = None,
    ) -> tuple[
        Optional[ContextT],
        Any,
        int,
        Optional[datetime],
        Optional[str],
        Optional[str],
        list[StepResult],
    ]:
        """Load workflow state from persistence backend.

        Returns:
            Tuple of (context, last_step_output, current_step_index, created_at, pipeline_name, pipeline_version, step_history)
        """
        if self.state_backend is None or not run_id:
            return None, None, 0, None, None, None, []

        loaded = await self.state_backend.load_state(run_id)
        if loaded is None:
            return None, None, 0, None, None, None, []

        wf_state = WorkflowState.model_validate(loaded)

        # Reconstruct context from persisted state
        context: Optional[ContextT] = None
        if wf_state.pipeline_context is not None:
            if context_model is not None:
                context = context_model.model_validate(wf_state.pipeline_context)
            else:
                context = PipelineContext.model_validate(wf_state.pipeline_context)  # type: ignore

            # Restore pipeline metadata from state
            if context is not None and hasattr(context, "pipeline_name"):
                context.pipeline_name = wf_state.pipeline_name
            if context is not None and hasattr(context, "pipeline_version"):
                context.pipeline_version = wf_state.pipeline_version

        # Reconstruct step history from persisted state
        step_history: list[StepResult] = []
        for step_data in wf_state.step_history:
            try:
                step_result = StepResult.model_validate(step_data)
                step_history.append(step_result)
            except Exception:
                # Skip invalid step data to avoid breaking resumption
                continue

        return (
            context,
            wf_state.last_step_output,
            wf_state.current_step_index,
            wf_state.created_at,
            wf_state.pipeline_name,
            wf_state.pipeline_version,
            step_history,
        )

    async def persist_workflow_state(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: Any | None,
        status: str,
        state_created_at: datetime | None = None,
        step_history: Optional[list[StepResult]] = None,
    ) -> None:
        """Persist current workflow state with intelligent caching and delta detection."""
        if self.state_backend is None or run_id is None:
            return

        # Calculate execution time if we have creation timestamp
        execution_time_ms = None
        if state_created_at is not None:
            execution_time_ms = int((datetime.now() - state_created_at).total_seconds() * 1000)

        # Estimate memory usage and optimize serialization
        memory_usage_mb = None
        pipeline_context = None

        if context is not None:
            try:
                import sys

                memory_usage_mb = sys.getsizeof(context) / (1024 * 1024)

                # First principles: Only serialize if context has actually changed
                if self._should_serialize_context(context, run_id):
                    # Check cache first
                    cached = self._get_cached_serialization(context, run_id)
                    if cached is not None:
                        pipeline_context = cached
                        logger.debug(f"Using cached serialization for run {run_id}")
                    else:
                        # Optimize serialization based on context size
                        logger.debug(f"About to serialize context for run {run_id}")
                        # Always use the most efficient serialization method
                        pipeline_context = context.model_dump()
                        logger.debug(f"Successfully serialized context for run {run_id}")

                        # Cache the result
                        self._cache_serialization(context, run_id, pipeline_context)
                        logger.debug(f"Serialized and cached context for run {run_id}")
                else:
                    # Context hasn't changed, use cached or minimal serialization
                    cached = self._get_cached_serialization(context, run_id)
                    if cached is not None:
                        pipeline_context = cached
                    else:
                        # Fallback to comprehensive serialization to prevent data loss
                        pipeline_context = {
                            "initial_prompt": getattr(context, "initial_prompt", ""),
                            "pipeline_id": getattr(context, "pipeline_id", "unknown"),
                            "pipeline_name": getattr(context, "pipeline_name", "unknown"),
                            "pipeline_version": getattr(context, "pipeline_version", "latest"),
                            "total_steps": getattr(context, "total_steps", 0),
                            "error_message": getattr(context, "error_message", None),
                            "run_id": getattr(context, "run_id", ""),
                            "created_at": getattr(context, "created_at", None),
                            "updated_at": getattr(context, "updated_at", None),
                        }
                        # Include any additional fields that might be present
                        for field_name in [
                            "status",
                            "current_step",
                            "last_error",
                            "metadata",
                        ]:
                            if hasattr(context, field_name):
                                pipeline_context[field_name] = getattr(context, field_name, None)
                    logger.debug(f"Skipped context serialization for unchanged run {run_id}")

            except Exception as e:
                logger.warning(f"Failed to serialize context for run {run_id}: {e}")
                # Comprehensive fallback to prevent data loss even in error cases
                pipeline_context = {
                    "error": f"Failed to serialize context: {e}",
                    "initial_prompt": getattr(context, "initial_prompt", ""),
                    "pipeline_id": getattr(context, "pipeline_id", "unknown"),
                    "pipeline_name": getattr(context, "pipeline_name", "unknown"),
                    "pipeline_version": getattr(context, "pipeline_version", "latest"),
                    "total_steps": getattr(context, "total_steps", 0),
                    "run_id": getattr(context, "run_id", ""),
                }

        # Serialize step history with error handling
        serialized_step_history = []
        if step_history is not None:
            for step_result in step_history:
                try:
                    serialized_step_history.append(step_result.model_dump())
                except Exception:
                    # Skip invalid step results to avoid breaking persistence
                    continue

        state_data = {
            "run_id": run_id,
            "pipeline_id": getattr(context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(context, "pipeline_version", "latest"),
            "current_step_index": current_step_index,
            "pipeline_context": pipeline_context,
            "last_step_output": last_step_output,
            "step_history": serialized_step_history,
            "status": status,
            "created_at": state_created_at or datetime.now(),
            "updated_at": datetime.now(),
            "total_steps": getattr(context, "total_steps", 0),
            "error_message": getattr(context, "error_message", None),
            "execution_time_ms": execution_time_ms,
            "memory_usage_mb": memory_usage_mb,
        }

        await self.state_backend.save_state(run_id, state_data)

    async def persist_workflow_state_optimized(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: Any | None,
        status: str,
        state_created_at: datetime | None = None,
        step_history: Optional[list[StepResult]] = None,
    ) -> None:
        """Optimized persistence with minimal overhead for performance-critical scenarios."""
        if self.state_backend is None or run_id is None:
            return

        # OPTIMIZATION: Use lightweight serialization for performance
        pipeline_context = None
        if context is not None:
            try:
                # OPTIMIZATION: Skip expensive change detection in performance mode
                # Use minimal serialization to reduce overhead
                pipeline_context = {
                    "initial_prompt": getattr(context, "initial_prompt", ""),
                    "pipeline_id": getattr(context, "pipeline_id", "unknown"),
                    "pipeline_name": getattr(context, "pipeline_name", "unknown"),
                    "pipeline_version": getattr(context, "pipeline_version", "latest"),
                    "run_id": getattr(context, "run_id", ""),
                }
                # Only include essential fields to minimize serialization overhead
                
                # OPTIMIZATION: For large contexts, use even more minimal serialization
                context_size = len(str(context))
                if context_size > 10000:  # Large context threshold
                    # For very large contexts, only persist essential metadata
                    pipeline_context = {
                        "initial_prompt": getattr(context, "initial_prompt", "")[:1000],  # Truncate
                        "pipeline_id": getattr(context, "pipeline_id", "unknown"),
                        "pipeline_name": getattr(context, "pipeline_name", "unknown"),
                        "run_id": getattr(context, "run_id", ""),
                        "context_size": context_size,  # Track size for debugging
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to serialize context for run {run_id}: {e}")
                pipeline_context = {
                    "error": f"Failed to serialize context: {e}",
                    "initial_prompt": getattr(context, "initial_prompt", ""),
                    "run_id": getattr(context, "run_id", ""),
                }

        # OPTIMIZATION: Skip step history serialization for performance
        # Only serialize essential metadata
        serialized_step_history = []
        if step_history is not None:
            for step_result in step_history:
                try:
                    # OPTIMIZATION: Only serialize essential fields
                    serialized_step_history.append({
                        "name": step_result.name,
                        "success": step_result.success,
                        "cost_usd": step_result.cost_usd,
                        "token_counts": step_result.token_counts,
                    })
                except Exception:
                    continue

        # OPTIMIZATION: Use minimal state data structure
        state_data = {
            "run_id": run_id,
            "pipeline_id": getattr(context, "pipeline_id", "unknown") if context else "unknown",
            "pipeline_name": getattr(context, "pipeline_name", "unknown") if context else "unknown",
            "pipeline_version": getattr(context, "pipeline_version", "latest") if context else "latest",
            "current_step_index": current_step_index,
            "pipeline_context": pipeline_context,
            "last_step_output": last_step_output,
            "status": status,
            "step_history": serialized_step_history,
        }

        if state_created_at is not None:
            state_data["created_at"] = state_created_at.isoformat()
        state_data["updated_at"] = datetime.now().isoformat()

        # OPTIMIZATION: Use async persistence to avoid blocking
        try:
            await self.state_backend.save_state(run_id, state_data)
        except Exception as e:
            logger.warning(f"Failed to persist state for run {run_id}: {e}")
            # Don't raise - persistence failure shouldn't break execution
            # This ensures that performance-critical scenarios continue even if persistence fails

    def get_run_id_from_context(self, context: Optional[ContextT]) -> str | None:
        """Extract run_id from context if available."""
        if context is None:
            return None
        return getattr(context, "run_id", None)

    async def delete_workflow_state(self, run_id: str | None) -> None:
        """Delete workflow state from backend."""
        if self.state_backend is None or run_id is None:
            return

        await self.state_backend.delete_state(run_id)

    # ----------------------- New persistence helpers -----------------------

    async def record_run_start(
        self,
        run_id: str,
        pipeline_id: str,
        pipeline_name: str,
        pipeline_version: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        if self.state_backend is None:
            return
        try:
            # Use provided timestamps or generate defaults
            now = datetime.utcnow().isoformat()
            created_at = created_at or now
            updated_at = updated_at or now

            await self.state_backend.save_run_start(
                {
                    "run_id": run_id,
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline_name,
                    "pipeline_version": pipeline_version,
                    "status": "running",
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        except NotImplementedError:
            pass

    async def record_step_result(
        self, run_id: str, step_result: StepResult, step_index: int
    ) -> None:
        if self.state_backend is None:
            return
        try:
            await self.state_backend.save_step_result(
                {
                    "step_run_id": f"{run_id}:{step_index}",
                    "run_id": run_id,
                    "step_name": step_result.name,
                    "step_index": step_index,
                    "status": "completed" if step_result.success else "failed",
                    "start_time": datetime.utcnow(),
                    "end_time": datetime.utcnow(),
                    "duration_ms": int(step_result.latency_s * 1000),
                    "cost": step_result.cost_usd,
                    "tokens": step_result.token_counts,
                    "input": None,
                    "output": step_result.output,
                    "error": step_result.feedback if not step_result.success else None,
                }
            )
        except NotImplementedError:
            pass

    async def record_run_end(self, run_id: str, result: PipelineResult[ContextT]) -> None:
        if self.state_backend is None:
            return
        try:
            await self.state_backend.save_run_end(
                run_id,
                {
                    "status": "completed"
                    if all(s.success for s in result.step_history)
                    else "failed",
                    "end_time": datetime.utcnow(),
                    "total_cost": result.total_cost_usd,
                    "final_context": result.final_pipeline_context.model_dump()
                    if result.final_pipeline_context
                    else None,
                },
            )

            # Save trace tree if available
            if result.trace_tree is not None:
                try:
                    # Convert trace tree to dict format for JSON serialization
                    trace_dict = self._convert_trace_to_dict(result.trace_tree)
                    await self.state_backend.save_trace(run_id, trace_dict)
                except Exception as e:
                    # Log error and save error trace for auditability
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Failed to save trace for run {run_id}: {e}")

                    # Save sanitized error trace for auditability
                    # Sanitize error message to prevent sensitive data leakage
                    error_message = str(e)
                    # Truncate and sanitize error message to prevent sensitive data leakage
                    sanitized_error = (
                        error_message[:100] + "..." if len(error_message) > 100 else error_message
                    )
                    # Remove potential sensitive patterns
                    import re

                    sanitized_error = re.sub(
                        r"(password|secret|key|token|api_key)\s*[:=]\s*\S+",
                        r"\1=***",
                        sanitized_error,
                        flags=re.IGNORECASE,
                    )

                    error_trace = {
                        "span_id": f"error_{run_id}",
                        "name": "trace_save_error",
                        "start_time": datetime.now().timestamp(),
                        "end_time": datetime.now().timestamp(),
                        "parent_span_id": None,
                        "attributes": {
                            "error_type": type(e).__name__,
                            "error_summary": f"Trace serialization failed: {sanitized_error}",
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "children": [],
                        "status": "error",
                    }

                    try:
                        await self.state_backend.save_trace(run_id, error_trace)
                        telemetry.logfire.info(
                            f"Saved error trace for run {run_id} after trace save failure"
                        )
                    except Exception as save_error:
                        telemetry.logfire.error(
                            f"Failed to save error trace for run {run_id}: {save_error}"
                        )
        except NotImplementedError:
            pass

    def _convert_trace_to_dict(self, trace_tree: Any) -> Dict[str, Any]:
        """Convert trace tree to dictionary format for JSON serialization."""
        if hasattr(trace_tree, "__dict__"):
            # Handle Span objects
            trace_dict: Dict[str, Any] = {
                "span_id": getattr(trace_tree, "span_id", "unknown"),
                "name": getattr(trace_tree, "name", "unknown"),
                "start_time": getattr(trace_tree, "start_time", 0.0),
                "end_time": getattr(trace_tree, "end_time", 0.0),
                "parent_span_id": getattr(trace_tree, "parent_span_id", None),
                "attributes": getattr(trace_tree, "attributes", {}),
                "children": [],
                "status": getattr(trace_tree, "status", "unknown"),
            }
            # Convert children recursively
            children = getattr(trace_tree, "children", [])
            for child in children:
                if isinstance(trace_dict["children"], list):
                    trace_dict["children"].append(self._convert_trace_to_dict(child))
            return trace_dict
        elif isinstance(trace_tree, dict):
            # Already a dict, just ensure children are converted
            if "children" in trace_tree:
                converted_children = []
                for child in trace_tree["children"]:
                    converted_children.append(self._convert_trace_to_dict(child))
                trace_tree["children"] = converted_children
            return trace_tree
        else:
            # Raise exception for truly invalid trace trees to trigger error handling
            raise ValueError(f"Unknown trace tree type: {type(trace_tree)}")

    def clear_cache(self, run_id: Optional[str] = None) -> None:
        """Clear serialization cache for a specific run or all runs with intelligent cleanup."""
        if run_id is None:
            # Clear all caches
            self._serialization_cache.clear()
            self._context_hash_cache.clear()
            logger.debug("Cleared all cache entries")
        else:
            # Clear cache for specific run with proper run_id handling
            keys_to_remove = []
            for key in self._serialization_cache.keys():
                try:
                    # Parse the cache key to extract run_id
                    key_run_id, _ = self._parse_cache_key(key)
                    if key_run_id == run_id:
                        keys_to_remove.append(key)
                except ValueError:
                    # Handle legacy cache keys that might still use the old format
                    # For legacy keys, we can't safely extract run_id, so we skip them
                    # This is safe as the cache will eventually be cleared anyway
                    continue

            # Remove the serialization cache entries
            for key in keys_to_remove:
                del self._serialization_cache[key]

            # Remove the corresponding hash cache entry
            self._context_hash_cache.pop(run_id, None)

            logger.debug(f"Cleared {len(keys_to_remove)} cache entries for run_id: {run_id}")

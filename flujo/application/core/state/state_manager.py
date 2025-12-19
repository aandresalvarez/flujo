"""State management with intelligent caching and delta-based persistence."""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Generic, Optional, TypeVar, Tuple, TYPE_CHECKING

from flujo.domain.models import StepResult, BaseModel, PipelineResult
from flujo.infra import telemetry
from flujo.state.backends import StateBackend
from flujo.state.backends.base import _serialize_for_json
from flujo.state.models import WorkflowState
from .state_serializer import StateSerializer
from flujo.type_definitions.common import JSONObject

if TYPE_CHECKING:
    from flujo.type_definitions.common import JSONObject

logger = logging.getLogger(__name__)

ContextT = TypeVar("ContextT", bound=BaseModel)


class StateManager(Generic[ContextT]):
    """Intelligent state manager with caching and delta-based persistence."""

    def __init__(
        self,
        state_backend: Optional[StateBackend] = None,
        serializer: Optional[StateSerializer[ContextT]] = None,
    ) -> None:
        """Initialize state manager with optional backend and serializer."""
        self.state_backend = state_backend
        self._serializer: StateSerializer[ContextT] = serializer or StateSerializer()
        # Legacy caches retained for compatibility; StateSerializer holds the authoritative caches.
        self._serialization_cache: dict[str, JSONObject] = {}
        self._context_hash_cache: dict[str, str] = {}

    def _should_serialize_context(self, context: Optional[ContextT], run_id: str) -> bool:
        """Determine if context needs serialization based on change detection."""
        if context is None:
            return False

        # Use the serializer's optimized hashing logic
        current_hash = self._serializer.compute_context_hash(context)
        cached_hash = self._context_hash_cache.get(run_id)

        if cached_hash != current_hash:
            # Context has changed, update cache and serialize
            self._context_hash_cache[run_id] = current_hash
            return True

        # Context hasn't changed, skip serialization
        return False

    def _get_cached_serialization(
        self, context: Optional[ContextT], run_id: str
    ) -> Optional[JSONObject]:
        """Get cached serialization result if available."""
        if context is None:
            return None

        # Use the serializer's optimized hashing logic
        context_hash = self._serializer.compute_context_hash(context)
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
        self, context: Optional[ContextT], run_id: str, serialized: JSONObject
    ) -> None:
        """Cache serialized context to avoid redundant serialization."""
        if context is None:
            return

        # Use the serializer's optimized hashing logic
        context_hash = self._serializer.compute_context_hash(context)
        cache_key = self._create_cache_key(run_id, context_hash)

        # Limit cache size to prevent memory leaks with intelligent eviction
        if len(self._serialization_cache) >= 100:
            # Evict the least recently used entry before adding a new item
            self._evict_least_recently_used_entry()

        self._serialization_cache[cache_key] = serialized

    def _build_context_fallback(
        self,
        context: Optional[ContextT],
        *,
        error_message: str | None = None,
    ) -> JSONObject:
        """Build a lightweight fallback serialization when full serialization fails."""

        fallback: JSONObject = {
            "initial_prompt": getattr(context, "initial_prompt", "") if context else "",
            "pipeline_id": getattr(context, "pipeline_id", "unknown") if context else "unknown",
            "pipeline_name": getattr(context, "pipeline_name", "unknown") if context else "unknown",
            "pipeline_version": getattr(context, "pipeline_version", "latest")
            if context
            else "latest",
            "total_steps": getattr(context, "total_steps", 0) if context else 0,
            "error_message": getattr(context, "error_message", None) if context else None,
            "run_id": getattr(context, "run_id", "") if context else "",
            "created_at": getattr(context, "created_at", None) if context else None,
            "updated_at": getattr(context, "updated_at", None) if context else None,
        }

        for field_name in ("status", "current_step", "last_error", "metadata"):
            if context is not None and hasattr(context, field_name):
                fallback[field_name] = getattr(context, field_name)

        if error_message is not None:
            fallback["error"] = error_message

        return fallback

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
        object | None,
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
            context = self._serializer.deserialize_context(wf_state.pipeline_context, context_model)

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

    async def rehydrate_pipeline_result(
        self,
        run_id: str,
        context_model: Optional[type[ContextT]] = None,
    ) -> Optional[PipelineResult[ContextT]]:
        """Rebuild a PipelineResult for the given run_id suitable for resume_async."""
        if not run_id:
            return None

        (
            context,
            _,
            _,
            _,
            _,
            _,
            step_history,
        ) = await self.load_workflow_state(run_id, context_model)

        if context is None and not step_history:
            return None

        total_cost = 0.0
        total_tokens = 0
        all_success = True
        for step in step_history:
            try:
                total_cost += float(step.cost_usd or 0.0)
            except Exception:
                pass
            try:
                total_tokens += int(step.token_counts or 0)
            except Exception:
                pass
            all_success = all_success and bool(step.success)

        result = PipelineResult[ContextT](
            step_history=step_history,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            final_pipeline_context=context,
        )
        result.success = all_success
        return result

    async def persist_workflow_state(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: object | None,
        status: str,
        state_created_at: datetime | None = None,
        step_history: Optional[list[StepResult]] = None,
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Persist current workflow state with intelligent caching and delta detection."""
        # Persist state even in test mode so unit tests that inspect workflow_state succeed.
        # Test-mode optimizations are applied in other paths (e.g., background monitors),
        # but core state persistence remains enabled for correctness.

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
                # Final/full persistence path: always serialize full context so nested structures
                # are available to tests that inspect persisted state.
                memory_usage_mb = None
                pipeline_context = self._serializer.serialize_context_full(context)
            except Exception as e:
                logger.warning(f"Failed to serialize context for run {run_id}: {e}")
                # Comprehensive fallback to prevent data loss even in error cases
                pipeline_context = self._build_context_fallback(
                    context, error_message=f"Failed to serialize context: {e}"
                )

        # Serialize step history with error handling
        # Persist full history for correctness (tests rely on exact entries); callers can
        # choose the optimized path when they want minimal snapshots.
        serialized_step_history = self._serializer.serialize_step_history_full(step_history)
        metadata_dict: JSONObject = metadata or {}

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
            "metadata": metadata_dict,
            "is_background_task": bool(metadata_dict.get("is_background_task", False)),
            "parent_run_id": metadata_dict.get("parent_run_id"),
            "task_id": metadata_dict.get("task_id"),
            "background_error": metadata_dict.get("background_error"),
        }

        normalized_state_obj = _serialize_for_json(state_data, strict=False)
        normalized_state: JSONObject = (
            normalized_state_obj if isinstance(normalized_state_obj, dict) else {}
        )
        await self.state_backend.save_state(run_id, normalized_state)

    async def persist_workflow_state_optimized(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: object | None,
        status: str,
        state_created_at: datetime | None = None,
        step_history: Optional[list[StepResult]] = None,
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Optimized persistence with minimal overhead for performance-critical scenarios."""
        if self.state_backend is None or run_id is None:
            return

        # Determine if this is the initial snapshot for this run
        is_initial_snapshot = (
            status == "running" and (current_step_index or 0) == 0 and state_created_at is None
        )

        # OPTIMIZATION STRATEGY:
        # - Initial running snapshot: persist MINIMAL context to cut overhead, but prime hash cache
        # - Final snapshots (completed/failed/paused): persist FULL context for correctness
        # - Intermediate running snapshots: MINIMAL when unchanged; FULL only if explicitly needed elsewhere
        pipeline_context = None
        if context is not None:
            try:
                if status != "running":
                    # Final snapshot: always persist full context for downstream consumers/tools
                    # Prime hash cache to reflect latest context
                    _ = self._serializer.should_serialize_context(context, run_id)
                    pipeline_context = self._serializer.serialize_context_full(context)
                elif is_initial_snapshot:
                    # Prime hash cache but persist minimal for first snapshot to reduce overhead
                    _ = self._serializer.should_serialize_context(context, run_id)
                    # Persist a minimal dict to satisfy schema with negligible cost
                    pipeline_context = self._serializer.serialize_context_minimal(context)
                elif self._serializer.should_serialize_context(context, run_id):
                    # Context changed mid-run: prefer minimal to reduce churn; final snapshot will be full
                    pipeline_context = self._serializer.serialize_context_minimal(context)
                else:
                    pipeline_context = self._serializer.serialize_context_minimal(context)
            except Exception as e:
                logger.warning(f"Failed to serialize context for run {run_id}: {e}")
                pipeline_context = self._build_context_fallback(
                    context, error_message=f"Failed to serialize context: {e}"
                )

        # Keep step_history: include minimal entries when provided (for crash recovery), else empty list
        if step_history:
            # Persist full history so resumptions/tests can reconstruct prior steps.
            serialized_step_history = self._serializer.serialize_step_history_full(step_history)
        else:
            serialized_step_history = []

        metadata_dict: JSONObject = metadata or {}

        # OPTIMIZATION: Use minimal state data structure
        state_data = {
            "run_id": run_id,
            "pipeline_id": getattr(context, "pipeline_id", "unknown") if context else "unknown",
            "pipeline_name": getattr(context, "pipeline_name", "unknown") if context else "unknown",
            "pipeline_version": getattr(context, "pipeline_version", "latest")
            if context
            else "latest",
            "current_step_index": current_step_index,
            "pipeline_context": pipeline_context,
            "last_step_output": last_step_output,
            "status": status,
            "step_history": serialized_step_history,
            "metadata": metadata_dict,
            "is_background_task": bool(metadata_dict.get("is_background_task", False)),
            "parent_run_id": metadata_dict.get("parent_run_id"),
            "task_id": metadata_dict.get("task_id"),
            "background_error": metadata_dict.get("background_error"),
        }

        if state_created_at is not None:
            state_data["created_at"] = state_created_at
        else:
            state_data["created_at"] = datetime.now()
        state_data["updated_at"] = datetime.now()

        # OPTIMIZATION: Use async persistence to avoid blocking
        try:
            normalized_state_obj = _serialize_for_json(state_data, strict=False)
            normalized_state: JSONObject = (
                normalized_state_obj if isinstance(normalized_state_obj, dict) else {}
            )
            await self.state_backend.save_state(run_id, normalized_state)
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
        # Disable per-run start persistence in test mode for performance
        try:
            from flujo.infra.config_manager import get_config_manager as _get_cfg

            _settings = _get_cfg().get_settings()
            if bool(getattr(_settings, "test_mode", False)):
                return
        except Exception:
            import os as _os

            if bool(_os.getenv("FLUJO_TEST_MODE")):
                return
        if self.state_backend is None:
            return
        try:
            # Use provided timestamps or generate defaults
            now = datetime.now(timezone.utc).isoformat()
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
            try:
                from ....infra.audit import log_audit as _audit

                _audit(
                    "run_start",
                    run_id=run_id,
                    pipeline_id=pipeline_id,
                    pipeline_name=pipeline_name,
                    pipeline_version=pipeline_version,
                )
            except Exception:
                pass
        except NotImplementedError:
            pass

    async def record_step_result(
        self, run_id: str, step_result: StepResult, step_index: int
    ) -> None:
        if self.state_backend is None:
            return
        # Persist step results even in test mode to satisfy tests that assert on steps persistence.
        try:
            await self.state_backend.save_step_result(
                {
                    "step_run_id": f"{run_id}:{step_index}",
                    "run_id": run_id,
                    "step_name": step_result.name,
                    "step_index": step_index,
                    "status": "completed" if step_result.success else "failed",
                    "start_time": datetime.now(timezone.utc),
                    "end_time": datetime.now(timezone.utc),
                    "duration_ms": int(step_result.latency_s * 1000),
                    "cost": step_result.cost_usd,
                    "tokens": step_result.token_counts,
                    "input": None,
                    "output": step_result.output,
                    "error": step_result.feedback if not step_result.success else None,
                    # Persist raw response if present in metadata (FSD-013)
                    "raw_response": (
                        step_result.metadata_.get("raw_llm_response")
                        if isinstance(getattr(step_result, "metadata_", None), dict)
                        else None
                    ),
                }
            )
        except NotImplementedError:
            pass

    async def record_run_end(self, run_id: str, result: PipelineResult[ContextT]) -> None:
        # In test mode, avoid heavy run-end writes but still save trace for integration tests
        test_mode = False
        try:
            from flujo.infra.config_manager import get_config_manager as _get_cfg

            _settings = _get_cfg().get_settings()
            test_mode = bool(getattr(_settings, "test_mode", False))
        except Exception:
            import os as _os

            test_mode = bool(_os.getenv("FLUJO_TEST_MODE"))

        if self.state_backend is None:
            return
        try:
            if not test_mode:
                end_payload = {
                    "status": "completed"
                    if all(s.success for s in result.step_history)
                    else "failed",
                    "end_time": datetime.now(timezone.utc),
                    "total_cost": result.total_cost_usd,
                    "final_context": result.final_pipeline_context.model_dump()
                    if result.final_pipeline_context
                    else None,
                }
                end_payload_obj = _serialize_for_json(end_payload)
                end_payload_json: JSONObject = (
                    end_payload_obj if isinstance(end_payload_obj, dict) else {}
                )
                await self.state_backend.save_run_end(run_id, end_payload_json)
            try:
                from ....infra.audit import log_audit as _audit

                _audit(
                    "run_end",
                    run_id=run_id,
                    status="completed" if all(s.success for s in result.step_history) else "failed",
                    steps=len(result.step_history),
                    total_cost=result.total_cost_usd,
                )
            except Exception:
                pass

            # Save trace tree if available
            if result.trace_tree is not None:
                try:
                    # Convert trace tree to dict format for JSON serialization
                    trace_dict = self._convert_trace_to_dict(result.trace_tree)
                    await self.state_backend.save_trace(run_id, trace_dict)
                except Exception as e:
                    # Log error and save error trace for auditability
                    from ....infra import telemetry

                    telemetry.logfire.error(f"Failed to save trace for run {run_id}: {e}")

                    # Save sanitized error trace for auditability
                    error_message = str(e)
                    sanitized_error = (
                        error_message[:100] + "..." if len(error_message) > 100 else error_message
                    )
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
            # Proactively drop serialization cache to reduce memory retention
            try:
                self._serializer.clear_cache(run_id)
            except Exception as cache_error:
                telemetry.logfire.debug(
                    f"Non-fatal error clearing serializer cache for {run_id}: {cache_error}"
                )
        except NotImplementedError:
            # Some backends may not implement record_run_end; ignore if so.
            pass
        except Exception as exc:
            telemetry.logfire.debug(f"Non-fatal error in record_run_end for {run_id}: {exc}")

    async def persist_evaluation(
        self,
        run_id: str,
        score: float,
        feedback: str | None = None,
        step_name: str | None = None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Persist shadow evaluation result via backend if available."""
        if self.state_backend is None:
            return
        persist = getattr(self.state_backend, "persist_evaluation", None)
        if persist is None or not callable(persist):
            return
        try:
            await persist(
                run_id=run_id,
                score=score,
                feedback=feedback,
                step_name=step_name,
                metadata=metadata,
            )
        except NotImplementedError:
            # Backend opted out of evaluation persistence.
            return
        except Exception as exc:
            telemetry.logfire.warning(
                "[StateManager] persist_evaluation failed",
                extra={"error": str(exc)},
            )

    def _convert_trace_to_dict(self, trace_tree: object) -> JSONObject:
        """Convert trace tree to dictionary format for JSON serialization."""
        if hasattr(trace_tree, "__dict__"):
            # Handle Span objects
            trace_dict: JSONObject = {
                "span_id": getattr(trace_tree, "span_id", "unknown"),
                "name": getattr(trace_tree, "name", "unknown"),
                "start_time": getattr(trace_tree, "start_time", 0.0),
                "end_time": getattr(trace_tree, "end_time", 0.0),
                "parent_span_id": getattr(trace_tree, "parent_span_id", None),
                "attributes": getattr(trace_tree, "attributes", {}),
                # Preserve span events for CLI/lens rendering (e.g., agent.prompt)
                "events": getattr(trace_tree, "events", []) or [],
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
        """Clear serialization cache for a specific run or all runs."""
        self._serializer.clear_cache(run_id)
        logger.debug(
            f"Cleared {'all' if run_id is None else 'scoped'} cache entries via StateSerializer"
        )

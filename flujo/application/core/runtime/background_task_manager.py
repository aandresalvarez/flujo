"""Background task lifecycle management."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Callable, Coroutine

from ....domain.models import BackgroundLaunched
from ....infra import telemetry
from ....infra.settings import get_settings
from ..context_manager import ContextManager
from ....exceptions import UsageLimitExceededError, PipelineAbortSignal, PausedException

if TYPE_CHECKING:  # pragma: no cover
    from ....domain.models import BaseModel

    from ..executor_core import ExecutorCore
    from ..types import ExecutionFrame
    from ....domain.models import StepOutcome, StepResult


class BackgroundTaskManager:
    """Manages the lifecycle of background tasks."""

    def __init__(self) -> None:
        self._background_tasks: set[asyncio.Task[object]] = set()

    def add_task(self, task: asyncio.Task[object]) -> None:
        """Add a background task to tracking."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def get_active_task_count(self) -> int:
        """Get the number of currently active background tasks."""
        return len(self._background_tasks)

    def has_active_tasks(self) -> bool:
        """Check if there are any active background tasks."""
        return bool(self._background_tasks)

    async def launch_background_task(
        self,
        *,
        step_name: str,
        run_coro: Callable[[], Coroutine[object, object, object]],
        task_id: str | None = None,
    ) -> "BackgroundLaunched[StepResult]":
        """Create, track, and return metadata for a background step execution."""
        task_id = task_id or f"bg_{uuid.uuid4().hex}"
        task = asyncio.create_task(run_coro(), name=f"flujo_bg_{step_name}_{task_id}")
        self.add_task(task)
        telemetry.logfire.info(f"Launched background step '{step_name}' (task_id={task_id})")
        return BackgroundLaunched(task_id=task_id, step_name=step_name)

    async def maybe_launch_background_step(
        self,
        *,
        core: "ExecutorCore[BaseModel]",
        frame: "ExecutionFrame[BaseModel]",
    ) -> "StepOutcome[StepResult] | None":
        """Launch the given frame in background mode if requested."""
        step = frame.step
        try:
            if not (
                hasattr(step, "config")
                and getattr(step.config, "execution_mode", "sync") == "background"
            ):
                return None
        except Exception:
            return None

        task_id = f"bg_{uuid.uuid4().hex}"

        async def _run_background() -> None:
            await self._execute_background_step(core=core, frame=frame, task_id=task_id)

        return await self.launch_background_task(
            step_name=core._safe_step_name(step), run_coro=_run_background, task_id=task_id
        )

    async def _execute_background_step(
        self,
        *,
        core: "ExecutorCore[BaseModel]",
        frame: "ExecutionFrame[BaseModel]",
        task_id: str,
    ) -> None:
        """Execute a background frame with isolated context and optional persistence."""
        try:
            from ..types import ExecutionFrame  # local import to avoid cycles

            bg_settings = getattr(get_settings(), "background_tasks", None)
            enable_state_tracking = bool(getattr(bg_settings, "enable_state_tracking", False))
            enable_resumability = bool(getattr(bg_settings, "enable_resumability", False))

            parent_context: BaseModel | None = getattr(frame, "context", None)
            parent_run_id: str | None = None
            if parent_context is not None:
                parent_run_id = getattr(parent_context, "run_id", None)
            bg_run_id = f"{parent_run_id}_bg_{task_id}" if parent_run_id else task_id

            try:
                isolated_context = ContextManager.isolate_strict(getattr(frame, "context", None))
            except Exception:
                try:
                    telemetry.logfire.warning(
                        "Background task using lenient context isolation; potential shared-state risk"
                    )
                except Exception:
                    pass
                isolated_context = ContextManager.isolate(getattr(frame, "context", None))
            if isolated_context is not None:
                try:
                    isolated_context.run_id = bg_run_id  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    if hasattr(isolated_context, "parent_run_id"):
                        isolated_context.parent_run_id = parent_run_id
                    else:
                        setattr(isolated_context, "parent_run_id", parent_run_id)
                except Exception:
                    pass
                try:
                    if hasattr(isolated_context, "is_background_task"):
                        isolated_context.is_background_task = True
                    if hasattr(isolated_context, "task_id"):
                        isolated_context.task_id = task_id
                except Exception:
                    pass

            step_name = core._safe_step_name(getattr(frame, "step", None))
            metadata = {
                "is_background_task": True,
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "step_name": step_name,
                "input_data": getattr(frame, "data", None),
            }

            state_manager = getattr(core, "state_manager", None)
            if enable_state_tracking and state_manager is not None:
                await core._register_background_task(
                    task_id=task_id,
                    bg_run_id=bg_run_id,
                    parent_run_id=parent_run_id,
                    step_name=step_name,
                    data=getattr(frame, "data", None),
                    context=isolated_context,
                    metadata=metadata,
                )

            try:
                step_copy = getattr(frame, "step")
                copy_error: Exception | None = None
                try:
                    step_copy = step_copy.model_copy(deep=True)
                except Exception as exc:  # best-effort defensive copy
                    copy_error = exc
                if hasattr(step_copy, "config"):
                    step_copy.config.execution_mode = "sync"
                elif hasattr(getattr(frame, "step", None), "config"):
                    frame.step.config.execution_mode = "sync"
                if copy_error is not None:
                    telemetry.logfire.debug(
                        "Background task step copy failed; using original step instance",
                        extra={"error": str(copy_error)},
                    )

                parent_quota = None
                try:
                    parent_quota = core._quota_manager.get_current_quota()
                except Exception:
                    parent_quota = None
                try:
                    quota = core._get_background_quota(parent_quota=parent_quota)
                except Exception:
                    quota = parent_quota

                # Best-effort preflight check: avoid launching when quota is already exhausted.
                # Quota enforcement itself happens inside the normal execution flow via
                # Reserve -> Execute -> Reconcile on the frame quota (do not double-count here).
                if quota is not None and bool(getattr(bg_settings, "enable_quota", False)):
                    try:
                        remaining_cost, remaining_tokens = quota.get_remaining()
                    except Exception:
                        remaining_cost, remaining_tokens = (0.0, 0)
                    # Only gate on availability; do not deplete quota for background bookkeeping.
                    if remaining_cost <= 0 and remaining_tokens <= 0:
                        metadata["background_error"] = "Background quota exhausted"
                        if (
                            enable_state_tracking
                            and state_manager is not None
                            and enable_resumability
                        ):
                            await core._mark_background_task_failed(
                                task_id=task_id,
                                context=isolated_context,
                                error=UsageLimitExceededError("Background quota exhausted"),
                                metadata=metadata,
                            )
                        return

                final_context: BaseModel | None = isolated_context

                def _context_setter(_res: object, updated_ctx: BaseModel | None) -> None:
                    nonlocal final_context
                    if updated_ctx is not None:
                        final_context = updated_ctx

                bg_frame: ExecutionFrame[BaseModel] = ExecutionFrame(
                    step=step_copy,
                    data=getattr(frame, "data", None),
                    context=isolated_context,
                    resources=getattr(frame, "resources", None),
                    limits=getattr(frame, "limits", None),
                    quota=quota,
                    stream=False,
                    on_chunk=None,
                    context_setter=_context_setter,
                    result=None,
                    _fallback_depth=0,
                )
                outcome = await core.execute(bg_frame)
                step_result = core._unwrap_outcome_to_step_result(
                    outcome, core._safe_step_name(step_copy)
                )
                if not getattr(step_result, "success", False):
                    raise RuntimeError(
                        getattr(step_result, "feedback", None) or "Background task failed"
                    )

                if enable_state_tracking and state_manager is not None and enable_resumability:
                    await core._mark_background_task_completed(
                        task_id=task_id,
                        context=final_context,
                        metadata=metadata,
                    )
            except PausedException as control_flow_err:
                if enable_state_tracking and state_manager is not None and enable_resumability:
                    metadata["background_error"] = str(control_flow_err)
                    metadata["error_category"] = "control_flow"
                    await core._mark_background_task_paused(
                        task_id=task_id,
                        context=final_context,
                        error=control_flow_err,
                        metadata=metadata,
                    )
                telemetry.logfire.warning(
                    f"Background task '{getattr(frame.step, 'name', 'unknown')}' paused: {control_flow_err}"
                )
            except PipelineAbortSignal as control_flow_err:
                if enable_state_tracking and state_manager is not None and enable_resumability:
                    metadata["background_error"] = str(control_flow_err)
                    metadata["error_category"] = "control_flow"
                    await core._mark_background_task_failed(
                        task_id=task_id,
                        context=final_context,
                        error=control_flow_err,
                        metadata=metadata,
                    )
                telemetry.logfire.warning(
                    f"Background task '{getattr(frame.step, 'name', 'unknown')}' aborted: {control_flow_err}"
                )
            except Exception as e:
                if enable_state_tracking and state_manager is not None and enable_resumability:
                    metadata["background_error"] = str(e)

                    def _classify_error(exc: Exception) -> str:
                        """Deterministically classify background errors without raising."""
                        try:
                            if isinstance(exc, (PausedException, PipelineAbortSignal)):
                                return "control_flow"
                            if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
                                return "network"
                            if isinstance(exc, (ValueError, TypeError, AttributeError)):
                                return "validation"
                            if isinstance(exc, (PermissionError, OSError)):
                                return "system"

                            err_str = str(exc).lower()
                            err_name = type(exc).__name__.lower()

                            # String-based fallbacks for wrapped errors (e.g., RuntimeError wrapping ValueError)
                            if "valueerror" in err_str or "value_error" in err_str:
                                return "validation"
                            if "typeerror" in err_str or "type_error" in err_str:
                                return "validation"
                            if "attributeerror" in err_str or "attribute_error" in err_str:
                                return "validation"

                            if "auth" in err_str or "auth" in err_name:
                                return "authentication"
                            if "quota" in err_str or "limit" in err_str or "exhaust" in err_str:
                                return "resource_exhaustion"
                            if "config" in err_str or "setting" in err_str:
                                return "configuration"

                            # Fallback to a known category to avoid persisting "unknown"
                            return "system"
                        except Exception:
                            return "system"

                    metadata["error_category"] = _classify_error(e)
                    if final_context is not None:
                        try:
                            if hasattr(final_context, "background_error_category"):
                                final_context.background_error_category = str(
                                    metadata.get("error_category", "system")
                                )
                        except Exception:
                            pass

                    await core._mark_background_task_failed(
                        task_id=task_id,
                        context=final_context,
                        error=e,
                        metadata=metadata,
                    )
                telemetry.logfire.error(
                    f"Background task failed for step '{getattr(frame.step, 'name', 'unknown')}': {e}"
                )
        except Exception as e:
            telemetry.logfire.error(f"Error during background task cleanup: {e}")
            # Only cancel tasks that are already completed with exceptions to avoid
            # impacting unrelated background tasks.
            for task in list(self._background_tasks):
                try:
                    if task.done() and not task.cancelled():
                        task_exc = task.exception()
                        if task_exc is not None:
                            telemetry.logfire.error(
                                "Cancelling background task after cleanup error",
                                extra={
                                    "task": repr(task),
                                    "task_name": task.get_name(),
                                    "error": str(task_exc),
                                },
                            )
                            task.cancel()
                            self._background_tasks.discard(task)
                except Exception:
                    pass

    async def wait_for_completion(self, timeout: float = 5.0) -> None:
        """Wait for all background tasks to complete with a timeout."""
        if not self._background_tasks:
            return

        pending = list(self._background_tasks)
        if not pending:
            return

        telemetry.logfire.info(
            f"Waiting for {len(pending)} background tasks to complete (timeout={timeout}s)..."
        )
        try:
            done, pending_set = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )

            for task in done:
                self._background_tasks.discard(task)

            if pending_set:
                telemetry.logfire.warning(
                    f"{len(pending_set)} background tasks timed out during shutdown. Cancelling..."
                )
                for task in pending_set:
                    task.cancel()
                    self._background_tasks.discard(task)

                if pending_set:
                    try:
                        await asyncio.wait(pending_set, timeout=0.5)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
        except Exception as e:
            telemetry.logfire.error(f"Error during background task cleanup: {e}")
            for task in list(self._background_tasks):
                try:
                    task.cancel()
                    self._background_tasks.discard(task)
                except Exception:
                    pass

    def cancel_all_tasks(self) -> None:
        """Cancel all active background tasks."""
        for task in list(self._background_tasks):
            try:
                task.cancel()
                self._background_tasks.discard(task)
            except Exception:
                pass

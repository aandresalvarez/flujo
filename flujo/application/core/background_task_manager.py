"""Background task lifecycle management."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Coroutine, Set

from ...domain.models import BackgroundLaunched
from ...infra import telemetry


class BackgroundTaskManager:
    """Manages the lifecycle of background tasks."""

    def __init__(self) -> None:
        self._background_tasks: Set[asyncio.Task[Any]] = set()

    def add_task(self, task: asyncio.Task[Any]) -> None:
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
        run_coro: Callable[[], Coroutine[Any, Any, Any]],
    ) -> BackgroundLaunched[Any]:
        """Create, track, and return metadata for a background step execution."""
        task_id = f"bg_{uuid.uuid4().hex}"
        task = asyncio.create_task(run_coro(), name=f"flujo_bg_{step_name}_{task_id}")
        self.add_task(task)
        telemetry.logfire.info(f"Launched background step '{step_name}' (task_id={task_id})")
        return BackgroundLaunched(task_id=task_id, step_name=step_name)

    async def wait_for_completion(self, timeout: float = 5.0) -> None:
        """Wait for all background tasks to complete with a timeout.

        Cancels any tasks that don't complete within the timeout period.
        """
        if not self._background_tasks:
            return

        # Create a copy to avoid modification during iteration if tasks complete
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

            # Remove completed tasks from tracking
            for task in done:
                self._background_tasks.discard(task)

            # Cancel any tasks that didn't complete within the timeout
            if pending_set:
                telemetry.logfire.warning(
                    f"{len(pending_set)} background tasks timed out during shutdown. Cancelling..."
                )
                for task in pending_set:
                    task.cancel()
                    self._background_tasks.discard(task)

                # Wait briefly for cancellations to complete
                if pending_set:
                    try:
                        await asyncio.wait(pending_set, timeout=0.5)
                    except asyncio.CancelledError:
                        raise  # Re-raise cancellation to propagate
                    except Exception:
                        pass  # Ignore other exceptions during cleanup
        except Exception as e:
            telemetry.logfire.error(f"Error during background task cleanup: {e}")
            # Continue with best-effort cleanup
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

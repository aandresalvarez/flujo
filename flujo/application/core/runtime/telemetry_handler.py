from __future__ import annotations

from typing import TYPE_CHECKING

from ....infra import telemetry

if TYPE_CHECKING:
    from ..executor_core import ExecutorCore
    from ..types import TContext


class TelemetryHandler:
    """Handles lightweight telemetry/logging concerns for ExecutorCore."""

    def __init__(self, core: "ExecutorCore[TContext]") -> None:
        self._core: "ExecutorCore[TContext]" = core

    def log_step_start(self, step: object, *, stream: bool, fallback_depth: int) -> None:
        try:
            telemetry.logfire.debug(
                f"Executing step: {self._core._safe_step_name(step)} "
                f"type={type(step).__name__} stream={stream} depth={fallback_depth}"
            )
        except Exception:
            pass

    def log_execution_error(self, step_name: str, exc: Exception) -> None:
        try:
            telemetry.logfire.error(
                f"[DEBUG] ExecutorCore caught unexpected exception at step '{step_name}': {type(exc).__name__}: {exc}"
            )
        except Exception:
            pass

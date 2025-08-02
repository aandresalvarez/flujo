from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING


from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import StepResult

if TYPE_CHECKING:
    from ..application.core.ultra_executor import ExecutorCore

    pass


class LocalBackend(ExecutionBackend):
    """Backend that executes steps in the current process."""

    def __init__(
        self,
        executor: "ExecutorCore[Any]",
        agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None,
    ) -> None:
        self.agent_registry = agent_registry or {}
        # ✅ INJECT the executor dependency instead of hard-coding it
        self._executor = executor

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        step = request.step

        # ✅ DELEGATE to the injected executor
        import flujo.infra.telemetry as telemetry

        telemetry.logfire.debug("=== LOCAL BACKEND EXECUTE STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step is ParallelStep: {hasattr(step, 'branches')}")

        return await self._executor.execute(
            step=step,
            data=request.input_data,
            context=request.context,
            resources=request.resources,
            limits=request.usage_limits,
            stream=request.stream,
            on_chunk=request.on_chunk,
            breach_event=request.breach_event,
        )

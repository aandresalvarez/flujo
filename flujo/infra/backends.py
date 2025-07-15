from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, cast


from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import StepResult
from ..application.core.ultra_executor import UltraStepExecutor

if TYPE_CHECKING:
    pass


class LocalBackend(ExecutionBackend):
    """Backend that executes steps in the current process."""

    def __init__(
        self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None
    ) -> None:
        self.agent_registry = agent_registry or {}
        # Create ultra executor for optimal performance
        self._ultra_executor: UltraStepExecutor[Any] = UltraStepExecutor()

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        step = request.step

        # Use ultra executor for all steps (replaces iterative executor)
        return cast(
            StepResult,
            await self._ultra_executor.execute_step(
                step=step,
                data=request.input_data,
                context=request.context,
                resources=request.resources,
                usage_limits=request.usage_limits,
                stream=request.stream,
                on_chunk=request.on_chunk,
            ),
        )

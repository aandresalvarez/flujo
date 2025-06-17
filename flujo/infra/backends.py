from __future__ import annotations

from typing import Dict

from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import StepResult
from ..application.flujo_engine import _run_step_logic


class LocalBackend(ExecutionBackend):
    """Backend that executes steps in the current process."""

    def __init__(self, agent_registry: Dict[str, AsyncAgentProtocol] | None = None) -> None:
        self.agent_registry = agent_registry or {}

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        return await _run_step_logic(
            request.step,
            request.input_data,
            request.pipeline_context,
            request.resources,
        )

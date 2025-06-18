from __future__ import annotations

from typing import Any, Dict, Optional

from ..domain.pipeline_dsl import Step
from ..domain.resources import AppResources

from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import StepResult, BaseModel
from ..application.flujo_engine import _run_step_logic


class LocalBackend(ExecutionBackend):
    """Backend that executes steps in the current process."""

    def __init__(
        self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None
    ) -> None:
        self.agent_registry = agent_registry or {}

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        async def executor(
            step: Step[Any, Any],
            data: Any,
            pipeline_context: Optional[BaseModel],
            resources: Optional[AppResources],
        ) -> StepResult:
            nested_request = StepExecutionRequest(
                step=step,
                input_data=data,
                pipeline_context=pipeline_context,
                resources=resources,
                context_model_defined=request.context_model_defined,
                usage_limits=request.usage_limits,
            )
            return await self.execute_step(nested_request)

        return await _run_step_logic(
            request.step,
            request.input_data,
            request.pipeline_context,
            request.resources,
            step_executor=executor,
            context_model_defined=request.context_model_defined,
            usage_limits=request.usage_limits,
        )

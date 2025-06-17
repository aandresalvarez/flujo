from __future__ import annotations

from typing import Protocol, Any, Dict, Optional
from pydantic import BaseModel

from .pipeline_dsl import Step
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..application.flujo_engine import Flujo
from .models import StepResult
from .resources import AppResources
from .agent_protocol import AsyncAgentProtocol


class StepExecutionRequest(BaseModel):
    """Serializable request for executing a single step."""

    step: Step
    input_data: Any
    pipeline_context: Optional[BaseModel] | None = None
    resources: Optional[AppResources] = None
    engine: Optional["Flujo"] = None

    model_config = {"arbitrary_types_allowed": True}


class ExecutionBackend(Protocol):
    """Protocol for executing pipeline steps."""

    agent_registry: Dict[str, AsyncAgentProtocol]

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        """Execute a single step and return the result."""
        ...

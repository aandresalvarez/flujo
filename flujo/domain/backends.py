from __future__ import annotations

from typing import Protocol, Any, Dict, Optional
from pydantic import BaseModel

from .pipeline_dsl import Step
from .models import StepResult, PipelineContext
from .resources import AppResources
from .agent_protocol import AsyncAgentProtocol


class StepExecutionRequest(BaseModel):
    """Serializable request for executing a single step."""

    step: Step[Any, Any]
    input_data: Any
    pipeline_context: Optional[BaseModel] | None = None
    resources: Optional[AppResources] = None
    # Whether the runner was created with a context model. Needed for
    # proper context passing semantics.
    context_model_defined: bool = False

    model_config = {"arbitrary_types_allowed": True}


class ExecutionBackend(Protocol):
    """Protocol for executing pipeline steps."""

    agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]]

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        """Execute a single step and return the result."""
        ...

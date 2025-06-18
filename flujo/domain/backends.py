from __future__ import annotations

from typing import Protocol, Any, Dict, Optional
from dataclasses import dataclass
from flujo.domain.models import BaseModel

from .pipeline_dsl import Step
from .models import StepResult, UsageLimits
from .resources import AppResources
from .agent_protocol import AsyncAgentProtocol


@dataclass
class StepExecutionRequest:
    """Serializable request for executing a single step."""

    # Use unparameterized ``Step`` type so Pydantic will not recreate the object
    # and accidentally reset attributes like ``max_retries``.
    step: Step[Any, Any]
    input_data: Any
    pipeline_context: Optional[BaseModel] | None = None
    resources: Optional[AppResources] = None
    # Whether the runner was created with a context model. Needed for
    # proper context passing semantics.
    context_model_defined: bool = False
    # Usage limits, propagated so nested executions (e.g., LoopStep) can enforce
    # governor checks mid-execution.
    usage_limits: Optional["UsageLimits"] = None


class ExecutionBackend(Protocol):
    """Protocol for executing pipeline steps."""

    agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]]

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        """Execute a single step and return the result."""
        ...

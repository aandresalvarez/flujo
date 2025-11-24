from __future__ import annotations

from typing import Protocol, Any, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass
from flujo.domain.models import BaseModel

# Local import to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dsl import Step

from .models import StepResult, UsageLimits, StepOutcome, Quota
from .resources import AppResources
from .agent_protocol import AsyncAgentProtocol


@dataclass
class StepExecutionRequest:
    """Serializable request for executing a single step.

    Contains the step to execute, input data, context object, resources,
    and execution configuration.
    """

    # Use unparameterized ``Step`` type so Pydantic will not recreate the object
    # and accidentally reset attributes like ``max_retries``.
    step: "Step[Any, Any]"
    input_data: Any
    context: Optional[BaseModel] | None = None
    resources: Optional[AppResources] = None
    # Whether the runner was created with a context model. Needed for
    # proper context passing semantics.
    context_model_defined: bool = False
    # Usage limits, propagated so nested executions (e.g., LoopStep) can enforce
    # governor checks mid-execution.
    usage_limits: Optional["UsageLimits"] = None
    # Quota passed proactively for pre-execution reservations
    quota: Optional[Quota] = None
    # Streaming support
    stream: bool = False
    on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None


class ExecutionBackend(Protocol):
    """Protocol for executing pipeline steps."""

    agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]]

    async def execute_step(self, request: StepExecutionRequest) -> StepOutcome[StepResult]:
        """Execute a single step and return a typed outcome."""
        ...

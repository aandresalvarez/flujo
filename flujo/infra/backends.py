from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Generic, TypeVar


from ..domain.backends import StepExecutionRequest
from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import StepResult, StepOutcome
from ..application.core.types import ExecutionFrame

if TYPE_CHECKING:
    from ..application.core.executor_core import ExecutorCore

    pass


TContext = TypeVar("TContext")


class LocalBackend(Generic[TContext]):
    """Backend that executes steps in the current process."""

    def __init__(
        self,
        executor: "ExecutorCore[Any]",
        agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]] | None = None,
    ) -> None:
        self.agent_registry = agent_registry or {}
        # ✅ INJECT the executor dependency instead of hard-coding it
        self._executor = executor

    async def execute_step(self, request: StepExecutionRequest) -> StepOutcome[StepResult]:
        step = request.step

        # ✅ DELEGATE to the injected executor
        import flujo.infra.telemetry as telemetry

        telemetry.logfire.debug("=== LOCAL BACKEND EXECUTE STEP ===")
        telemetry.logfire.debug(f"Step type: {type(step)}")
        telemetry.logfire.debug(f"Step name: {step.name}")
        telemetry.logfire.debug(f"Step is ParallelStep: {hasattr(step, 'branches')}")

        # Create ExecutionFrame for the new ExecutorCore.execute signature
        frame: ExecutionFrame[Any] = ExecutionFrame(
            step=step,
            data=request.input_data,
            context=request.context,
            resources=request.resources,
            limits=request.usage_limits,
            quota=request.quota,
            stream=request.stream,
            on_chunk=request.on_chunk,
            context_setter=lambda result, ctx: None,  # Default context setter for backend calls
        )

        outcome = await self._executor.execute(frame)
        try:
            import flujo.infra.telemetry as telemetry

            telemetry.logfire.debug(
                f"[LocalBackend] outcome for step '{getattr(step, 'name', '<unnamed>')}' -> {type(outcome).__name__}"
            )
        except Exception:
            pass
        # Always return a typed outcome
        assert not isinstance(outcome, StepResult)
        return outcome

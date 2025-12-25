"""Step type routing and dispatch logic."""

from __future__ import annotations

import inspect
from typing import TypeAlias

from ....domain.models import Failure, StepOutcome, StepResult
from ..types import ExecutionFrame, TContext
from ..policy_registry import PolicyCallable, PolicyRegistry, StepPolicy, StepType


RegisteredPolicy: TypeAlias = PolicyCallable | StepPolicy[StepType]


class ExecutionDispatcher:
    """Routes step execution to the appropriate policy handler."""

    def __init__(
        self, registry: PolicyRegistry | None = None, *, core: object | None = None
    ) -> None:
        self._registry: PolicyRegistry = registry or PolicyRegistry()
        self._core = core

    def register(self, step_type: type[StepType], policy: PolicyCallable) -> None:
        """Register a policy; thin wrapper around PolicyRegistry."""
        self._registry.register(step_type, policy)

    def get_policy(self, step: StepType) -> RegisteredPolicy | None:
        """Return the policy callable for a step instance, if any."""
        policy = self._registry.get(type(step))
        return policy

    async def dispatch(self, frame: ExecutionFrame[TContext]) -> StepOutcome[StepResult]:
        """Dispatch execution to the appropriate policy or return a Failure."""
        step: StepType = frame.step
        policy = self.get_policy(step)
        if policy is None:
            return Failure(
                error=TypeError(f"No policy registered for step type: {type(step).__name__}"),
                feedback=f"Unhandled step type: {type(step).__name__}",
                step_result=StepResult(name=str(getattr(step, "name", "<unknown>")), success=False),
            )
        if isinstance(policy, StepPolicy):
            core_obj: object | None = (
                self._core if self._core is not None else getattr(frame, "core", None)
            )
            if core_obj is None:
                raise TypeError("Executor core is required for policy execution")
            self._validate_frame_signature(policy)
            return await policy.execute(core_obj, frame)  # type: ignore[arg-type]
        return await policy(frame)  # type: ignore[arg-type]

    def _validate_frame_signature(self, policy: StepPolicy[StepType]) -> None:
        """Ensure the policy execute method accepts a `frame` parameter."""
        try:
            sig = inspect.signature(policy.execute)
            params = list(sig.parameters.values())
            if len(params) < 2:
                raise TypeError(
                    f"{type(policy).__name__}.execute must accept (core, frame); got {sig}"
                )
            frame_param = params[1]
            if frame_param.name != "frame":
                raise TypeError(
                    f"{type(policy).__name__}.execute must accept `frame` as second parameter; got {sig}"
                )
        except TypeError:
            raise
        except Exception as exc:
            raise TypeError(
                f"Failed to validate execute signature for {type(policy).__name__}: {exc}"
            ) from exc

    @property
    def registry(self) -> PolicyRegistry:
        """Expose underlying registry (compatibility)."""
        return self._registry

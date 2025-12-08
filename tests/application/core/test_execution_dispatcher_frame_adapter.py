from typing import Any

import pytest

from flujo.application.core.execution_dispatcher import ExecutionDispatcher
from flujo.application.core.policy_registry import PolicyRegistry, StepPolicy
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepOutcome, StepResult, Success


class _DummyCore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._enable_cache = False

    def _cache_key(self, frame: ExecutionFrame[Any]) -> str:
        return f"cache::{getattr(frame.step, 'name', 'unknown')}"


class _NewStylePolicy(StepPolicy[Step[Any, Any]]):
    @property
    def handles_type(self) -> type[Step[Any, Any]]:
        return Step

    async def execute(self, core: _DummyCore, frame: ExecutionFrame[Any]) -> StepOutcome[Any]:
        core.calls.append(("frame", frame.step.name))
        return Success(step_result=StepResult(name=frame.step.name, success=True))


def _make_frame(name: str, *, fallback_depth: int = 0) -> ExecutionFrame[Any]:
    step = Step(name=name, agent=None)
    frame = ExecutionFrame(
        step=step,
        data=f"data:{name}",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda *_args: None,
    )
    frame._fallback_depth = fallback_depth
    return frame


@pytest.mark.asyncio
async def test_dispatcher_prefers_frame_signature_when_available() -> None:
    registry = PolicyRegistry()
    policy = _NewStylePolicy()
    registry.register(Step, policy)
    core = _DummyCore()
    dispatcher = ExecutionDispatcher(registry, core=core)

    frame = _make_frame("frame-policy")
    outcome = await dispatcher.dispatch(frame)

    assert isinstance(outcome, Success)
    assert core.calls == [("frame", "frame-policy")]


@pytest.mark.asyncio
async def test_dispatcher_rejects_legacy_signature() -> None:
    class _LegacyPolicy(StepPolicy[Step[Any, Any]]):
        @property
        def handles_type(self) -> type[Step[Any, Any]]:
            return Step

        async def execute(self, core: _DummyCore, *args: Any, **kwargs: Any) -> StepOutcome[Any]:
            return Success(step_result=StepResult(name="legacy", success=True))

    registry = PolicyRegistry()
    registry.register(Step, _LegacyPolicy())
    core = _DummyCore()
    dispatcher = ExecutionDispatcher(registry, core=core)

    frame = _make_frame("legacy-policy", fallback_depth=2)
    with pytest.raises(TypeError):
        await dispatcher.dispatch(frame)

from typing import Any

import pytest

from flujo.application.core.step_policies import DefaultHitlStepExecutor, PolicyRegistry
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.models import Paused
from flujo.exceptions import ConfigurationError
from flujo.domain.dsl.step import HumanInTheLoopStep, Step


class _DummyCore:
    def _update_context_state(self, *_args, **_kwargs):
        pass

    def __init__(self) -> None:
        class _QM:
            def get_current_quota(self) -> Any:
                return None

        self._quota_manager = _QM()


class _CoreWithStack(_DummyCore):
    def __init__(self, frames: list[Any]) -> None:
        super().__init__()
        self._execution_stack = frames


class _Frame:
    def __init__(self, kind: str, name: str) -> None:
        self.step_kind = kind
        self.name = name


async def test_hitl_executor_returns_paused_outcome():
    core = _DummyCore()
    step = HumanInTheLoopStep(name="hitl", message_for_user="Please confirm")
    frame = make_execution_frame(
        core,
        step,
        "input",
        None,
        None,
        None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultHitlStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome, Paused)
    assert "Please confirm" in outcome.message


async def test_hitl_executor_allows_loop_without_conditional() -> None:
    core = _CoreWithStack([_Frame("loop", "outer_loop")])
    step = HumanInTheLoopStep(name="hitl", message_for_user="Loop ok")
    frame = make_execution_frame(
        core,
        step,
        "input",
        None,
        None,
        None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultHitlStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome, Paused)


async def test_hitl_executor_raises_on_loop_conditional_nesting() -> None:
    frames = [_Frame("loop", "outer_loop"), _Frame("conditional", "router")]
    core = _CoreWithStack(frames)
    step = HumanInTheLoopStep(name="hitl", message_for_user="Nested error")

    with pytest.raises(ConfigurationError) as exc:
        frame = make_execution_frame(
            core,
            step,
            "input",
            None,
            None,
            None,
            context_setter=None,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            result=None,
            quota=None,
        )
        await DefaultHitlStepExecutor().execute(core=core, frame=frame)

    assert "HITL" in str(exc.value)
    assert "HITL-NESTED-001" in str(exc.value)


class DummyStep(Step[Any, Any]):
    pass


def test_policy_registry_register_and_get():
    registry = PolicyRegistry()

    class DummyPolicy:
        pass

    policy = DummyPolicy()
    registry.register(DummyStep, policy)
    assert registry.get(DummyStep) is policy


def test_policy_registry_get_unregistered_returns_none():
    registry = PolicyRegistry()

    class AnotherStep(DummyStep):
        pass

    assert registry.get(AnotherStep) is None


def test_policy_registry_rejects_non_step_types():
    registry = PolicyRegistry()

    class NotAStep:
        pass

    try:
        registry.register(NotAStep, object())  # type: ignore[arg-type]
        assert False, "Expected TypeError for non-Step registration"
    except TypeError:
        pass

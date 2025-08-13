from typing import Any

from flujo.application.core.step_policies import DefaultHitlStepExecutor, PolicyRegistry
from flujo.domain.models import Paused
from flujo.domain.dsl.step import HumanInTheLoopStep, Step


class _DummyCore:
    def _update_context_state(self, *_args, **_kwargs):
        pass


async def test_hitl_executor_returns_paused_outcome():
    core = _DummyCore()
    step = HumanInTheLoopStep(name="hitl", message_for_user="Please confirm")
    outcome = await DefaultHitlStepExecutor().execute(
        core=core,
        step=step,
        data="input",
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
    )
    assert isinstance(outcome, Paused)
    assert "Please confirm" in outcome.message


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

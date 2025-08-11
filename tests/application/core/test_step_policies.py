import pytest

from flujo.application.core.step_policies import DefaultHitlStepExecutor
from flujo.domain.models import Paused
from flujo.domain.dsl.step import HumanInTheLoopStep


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


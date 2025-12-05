from __future__ import annotations

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultConditionalStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import Success


class _ConstAgentTrue:
    async def run(self, *_args, **_kwargs):
        return "It was true"


class _ConstAgentFalse:
    async def run(self, *_args, **_kwargs):
        return "It was false"


@pytest.mark.asyncio
async def test_conditional_policy_coerces_boolean_branch_key_true() -> None:
    """Policy converts boolean condition True -> 'true' branch key (FSD-026)."""
    core = ExecutorCore()

    def cond_bool(data, _ctx):
        return bool(data.get("flag"))

    cond = ConditionalStep(
        name="cond_bool",
        condition_callable=cond_bool,
        branches={
            "true": Pipeline.from_step(Step(name="T", agent=_ConstAgentTrue())),
            "false": Pipeline.from_step(Step(name="F", agent=_ConstAgentFalse())),
        },
    )

    frame = make_execution_frame(
        core,
        cond,
        {"flag": True},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    out = await DefaultConditionalStepExecutor().execute(core=core, frame=frame)
    assert isinstance(out, Success)
    assert out.step_result.success is True
    assert out.step_result.output == "It was true"


@pytest.mark.asyncio
async def test_conditional_policy_coerces_boolean_branch_key_false() -> None:
    core = ExecutorCore()

    def cond_bool(data, _ctx):
        return bool(data.get("flag"))

    cond = ConditionalStep(
        name="cond_bool",
        condition_callable=cond_bool,
        branches={
            "true": Pipeline.from_step(Step(name="T", agent=_ConstAgentTrue())),
            "false": Pipeline.from_step(Step(name="F", agent=_ConstAgentFalse())),
        },
    )

    frame = make_execution_frame(
        core,
        cond,
        {"flag": False},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    out = await DefaultConditionalStepExecutor().execute(core=core, frame=frame)
    assert isinstance(out, Success)
    assert out.step_result.success is True
    assert out.step_result.output == "It was false"

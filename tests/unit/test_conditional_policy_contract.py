import pytest
from unittest.mock import MagicMock

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultConditionalStepExecutor
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.conditional import ConditionalStep


class _EchoAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        return payload


@pytest.mark.asyncio
async def test_conditional_policy_calls_condition_with_original_data_and_context() -> None:
    core = ExecutorCore()

    # condition mock returns a branch key without inspecting the payload
    cond_fn = MagicMock(return_value="ok")

    cond = ConditionalStep(
        name="contract",
        condition_callable=cond_fn,
        branches={
            "ok": Pipeline.from_step(Step(name="OK", agent=_EchoAgent())),
        },
    )

    data = "test_data"

    # Execute policy
    await DefaultConditionalStepExecutor().execute(
        core,
        cond,
        data=data,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        _fallback_depth=0,
    )

    # Verify the condition received the original data and context
    cond_fn.assert_called_once_with(data, None)

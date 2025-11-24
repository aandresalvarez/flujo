import pytest
from typing import Any, Dict
from pydantic import BaseModel

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl import Step, Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import MergeStrategy, BranchFailureStrategy
from flujo.domain.models import StepResult


class Ctx(BaseModel):
    value: str = "base"
    scratchpad: Dict[str, Any] = {}


@pytest.mark.asyncio
async def test_parallel_default_context_update_conflict_fails():
    async def branch_a(_: Any) -> Any:
        return _

    async def branch_b(_: Any) -> Any:
        return _

    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step.from_callable(branch_a, name="a")),
            "b": Pipeline.from_step(Step.from_callable(branch_b, name="b")),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )

    base_ctx = Ctx(value="X")

    # Fake executor to inject branch contexts with conflicting values
    async def fake_step_executor(step, input_data, context, resources):
        nm = getattr(step.steps[0], "name", "")
        if nm == "a":
            return StepResult(
                name="a", output=input_data, success=True, branch_context=Ctx(value="A")
            )
        if nm == "b":
            return StepResult(
                name="b", output=input_data, success=True, branch_context=Ctx(value="B")
            )
        return StepResult(name="?", output=input_data, success=True)

    core = ExecutorCore()
    from flujo.application.core.step_policies import DefaultParallelStepExecutor

    execu = DefaultParallelStepExecutor()
    outcome = await execu.execute(
        core,
        p,
        data={"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
        step_executor=fake_step_executor,
    )

    sr = outcome.step_result if hasattr(outcome, "step_result") else outcome
    assert not sr.success
    assert "Merge conflict for key 'value'" in (sr.feedback or "")


@pytest.mark.asyncio
async def test_parallel_overwrite_allows_conflict():
    async def branch_a(_: Any) -> Any:
        return _

    async def branch_b(_: Any) -> Any:
        return _

    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step.from_callable(branch_a, name="a")),
            "b": Pipeline.from_step(Step.from_callable(branch_b, name="b")),
        },
        merge_strategy=MergeStrategy.OVERWRITE,
    )

    base_ctx = Ctx(value="X")

    async def fake_step_executor(step, input_data, context, resources):
        nm = getattr(step.steps[0], "name", "")
        if nm == "a":
            return StepResult(
                name="a", output=input_data, success=True, branch_context=Ctx(value="A")
            )
        if nm == "b":
            return StepResult(
                name="b", output=input_data, success=True, branch_context=Ctx(value="B")
            )
        return StepResult(name="?", output=input_data, success=True)

    from flujo.application.core.step_policies import DefaultParallelStepExecutor

    execu = DefaultParallelStepExecutor()
    outcome = await execu.execute(
        ExecutorCore(),
        p,
        data={"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
        step_executor=fake_step_executor,
    )

    sr = outcome.step_result if hasattr(outcome, "step_result") else outcome
    assert sr.success

from typing import Any

import pytest
from pydantic import BaseModel, Field

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.domain.dsl import Step, Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import BranchFailureStrategy, MergeStrategy
from flujo.domain.models import StepResult
from flujo.type_definitions.common import JSONObject
from tests.test_types.fixtures import create_test_step_result


class Ctx(BaseModel):
    value: str = "base"
    scratchpad: JSONObject = Field(default_factory=dict)


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
    async def fake_step_executor(
        step: Any, input_data: Any, context: Any, resources: Any
    ) -> StepResult:
        nm = getattr(step.steps[0], "name", "")
        if nm == "a":
            return create_test_step_result(
                name="a", output=input_data, success=True, branch_context=Ctx(value="A")
            )
        if nm == "b":
            return create_test_step_result(
                name="b", output=input_data, success=True, branch_context=Ctx(value="B")
            )
        return create_test_step_result(name="?", output=input_data, success=True)

    core = ExecutorCore()

    execu = DefaultParallelStepExecutor()
    frame = make_execution_frame(
        core,
        p,
        {"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    setattr(frame, "step_executor", fake_step_executor)
    outcome = await execu.execute(core=core, frame=frame)

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

    async def fake_step_executor(
        step: Any, input_data: Any, context: Any, resources: Any
    ) -> StepResult:
        nm = getattr(step.steps[0], "name", "")
        if nm == "a":
            return create_test_step_result(
                name="a", output=input_data, success=True, branch_context=Ctx(value="A")
            )
        if nm == "b":
            return create_test_step_result(
                name="b", output=input_data, success=True, branch_context=Ctx(value="B")
            )
        return StepResult(name="?", output=input_data, success=True)

    from flujo.application.core.step_policies import DefaultParallelStepExecutor

    execu = DefaultParallelStepExecutor()
    core = ExecutorCore()
    frame = make_execution_frame(
        core,
        p,
        {"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    setattr(frame, "step_executor", fake_step_executor)
    outcome = await execu.execute(core=core, frame=frame)

    sr = outcome.step_result if hasattr(outcome, "step_result") else outcome
    assert sr.success

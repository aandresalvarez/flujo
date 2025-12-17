import pytest
from flujo.domain.models import BaseModel

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl import Step, Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.step import BranchFailureStrategy, MergeStrategy
from flujo.domain.models import StepResult


class Ctx(BaseModel):
    value: str = "base"


@pytest.mark.asyncio
async def test_parallel_default_context_update_conflict_fails():
    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step(name="a", agent=None)),
            "b": Pipeline.from_step(Step(name="b", agent=None)),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )

    base_ctx = Ctx(value="X")

    core = ExecutorCore()

    class _MutatingExecutor:
        async def execute(
            self,
            _core: object,
            step: object,
            data: object,
            context: object,
            _resources: object,
            _limits: object,
            _stream: bool,
            _on_chunk: object,
            _cache_key: object,
            _fallback_depth: int,
        ) -> StepResult:
            nm = getattr(step, "name", "")
            if hasattr(context, "value"):
                if nm == "a":
                    setattr(context, "value", "A")
                elif nm == "b":
                    setattr(context, "value", "B")
            return StepResult(name=nm, output=data, success=True, attempts=1)

    core.agent_step_executor = _MutatingExecutor()
    sr = await core._handle_parallel_step(
        parallel_step=p,
        data={"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
    )
    assert not sr.success
    assert "Merge conflict for key 'value'" in (sr.feedback or "")


@pytest.mark.asyncio
async def test_parallel_overwrite_allows_conflict():
    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step(name="a", agent=None)),
            "b": Pipeline.from_step(Step(name="b", agent=None)),
        },
        merge_strategy=MergeStrategy.OVERWRITE,
    )

    base_ctx = Ctx(value="X")

    core = ExecutorCore()

    class _MutatingExecutor:
        async def execute(
            self,
            _core: object,
            step: object,
            data: object,
            context: object,
            _resources: object,
            _limits: object,
            _stream: bool,
            _on_chunk: object,
            _cache_key: object,
            _fallback_depth: int,
        ) -> StepResult:
            nm = getattr(step, "name", "")
            if hasattr(context, "value"):
                if nm == "a":
                    setattr(context, "value", "A")
                elif nm == "b":
                    setattr(context, "value", "B")
            return StepResult(name=nm, output=data, success=True, attempts=1)

    core.agent_step_executor = _MutatingExecutor()
    sr = await core._handle_parallel_step(
        parallel_step=p,
        data={"x": 1},
        context=base_ctx,
        resources=None,
        limits=None,
        context_setter=None,
    )
    assert sr.success

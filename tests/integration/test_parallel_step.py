import os
import asyncio
from typing import Any
import pytest
from flujo.domain.models import BaseModel, PipelineContext, StepResult
from flujo.domain import Step, MergeStrategy, BranchFailureStrategy, UsageLimits
from flujo.exceptions import UsageLimitExceededError
from flujo.testing.utils import gather_result
from flujo.application.runner import Flujo

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class Ctx(BaseModel):
    val: int = 0


class AddAgent:
    def __init__(self, inc: int) -> None:
        self.inc = inc

    async def run(self, data: int, *, context: Ctx | None = None) -> int:
        if context is not None:
            context.val += self.inc
        await asyncio.sleep(0)
        return data + self.inc


@pytest.mark.asyncio
async def test_parallel_step_context_isolation() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": AddAgent(1)}),
        "b": Step.model_validate({"name": "b", "agent": AddAgent(2)}),
    }
    parallel = Step.parallel("par", branches)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.output == {"a": 1, "b": 2}
    assert result.final_pipeline_context.val == 0


@pytest.mark.asyncio
async def test_parallel_step_result_structure() -> None:
    branches = {
        "x": Step.model_validate({"name": "x", "agent": AddAgent(3)}),
        "y": Step.model_validate({"name": "y", "agent": AddAgent(4)}),
    }
    parallel = Step.parallel("par_out", branches)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, 1)
    step_result = result.step_history[-1]
    assert isinstance(step_result.output, dict)
    assert set(step_result.output.keys()) == {"x", "y"}
    assert step_result.success is True


class ScratchCtx(PipelineContext):
    val: int = 0


class ScratchAgent:
    def __init__(self, key: str, val: int, fail: bool = False, delay: float = 0.0) -> None:
        self.key = key
        self.val = val
        self.fail = fail
        self.delay = delay

    async def run(self, data: int, *, context: ScratchCtx | None = None) -> int:
        if self.fail:
            raise RuntimeError("boom")
        await asyncio.sleep(self.delay)
        if context is not None:
            context.scratchpad[self.key] = self.val
        return data + self.val


class CostlyAgent:
    def __init__(self, cost: float = 0.1, delay: float = 0.0) -> None:
        self.cost = cost
        self.delay = delay

    async def run(self, data: int) -> Any:
        await asyncio.sleep(self.delay)

        class Output(BaseModel):
            value: int
            cost_usd: float = self.cost
            token_counts: int = 0

        return Output(value=data)


@pytest.mark.asyncio
async def test_parallel_merge_scratchpad() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("a", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("b", 2)}),
    }
    parallel = Step.parallel(
        "merge_sp",
        branches,
        merge_strategy=MergeStrategy.MERGE_SCRATCHPAD,
    )
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    assert result.final_pipeline_context.scratchpad["a"] == 1
    assert result.final_pipeline_context.scratchpad["b"] == 2


@pytest.mark.asyncio
async def test_parallel_overwrite_conflict() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("v", 1, delay=0.1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("v", 2, delay=0.2)}),
    }
    parallel = Step.parallel("overwrite", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    assert result.final_pipeline_context.scratchpad["v"] == 2


@pytest.mark.asyncio
async def test_parallel_overwrite_preserves_context() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2)}),
    }
    parallel = Step.parallel(
        "overwrite_keep",
        branches,
        context_include_keys=["scratchpad", "initial_prompt"],
        merge_strategy=MergeStrategy.OVERWRITE,
    )
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(
        runner,
        0,
        initial_context_data={"initial_prompt": "x", "val": 5},
    )
    assert result.final_pipeline_context.val == 5
    assert result.final_pipeline_context.scratchpad["y"] == 2


@pytest.mark.asyncio
async def test_parallel_overwrite_multi_branch_order() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("v", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("v", 2)}),
        "c": Step.model_validate({"name": "c", "agent": ScratchAgent("w", 3)}),
    }
    parallel = Step.parallel("overwrite_multi", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    assert result.final_pipeline_context.scratchpad["v"] == 2
    assert result.final_pipeline_context.scratchpad["w"] == 3


@pytest.mark.asyncio
async def test_parallel_propagate_failure() -> None:
    branches = {
        "good": Step.model_validate({"name": "good", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel(
        "fail_prop", branches, on_branch_failure=BranchFailureStrategy.PROPAGATE
    )
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert not step_result.success
    assert isinstance(step_result.output["bad"], StepResult)


@pytest.mark.asyncio
async def test_parallel_ignore_failure() -> None:
    branches = {
        "good": Step.model_validate({"name": "good", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel(
        "fail_ignore", branches, on_branch_failure=BranchFailureStrategy.IGNORE
    )
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert step_result.success
    assert isinstance(step_result.output["bad"], StepResult)


@pytest.mark.asyncio
async def test_parallel_ignore_failure_all_fail() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("a", 1, fail=True)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel(
        "all_fail_ignore",
        branches,
        on_branch_failure=BranchFailureStrategy.IGNORE,
    )
    runner = Flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert not step_result.success
    assert all(isinstance(step_result.output[name], StepResult) for name in branches)


@pytest.mark.asyncio
async def test_governor_precedence_over_failure_strategy() -> None:
    branches = {
        "costly": Step.model_validate(
            {"name": "costly", "agent": CostlyAgent(cost=0.2, delay=0.0)}
        ),
        "slow": Step.model_validate({"name": "slow", "agent": CostlyAgent(cost=0.0, delay=0.5)}),
    }
    parallel = Step.parallel(
        "gov_precedence",
        branches,
        on_branch_failure=BranchFailureStrategy.IGNORE,
    )
    limits = UsageLimits(total_cost_usd_limit=0.1)
    runner = Flujo(parallel, usage_limits=limits, context_model=ScratchCtx)
    with pytest.raises(UsageLimitExceededError):
        await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})

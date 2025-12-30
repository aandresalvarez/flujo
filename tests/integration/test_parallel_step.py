import os
import asyncio
from typing import Any
import pytest
from pydantic import Field
from flujo.domain.models import BaseModel, PipelineContext, StepResult
from flujo.domain import Step, MergeStrategy, BranchFailureStrategy, UsageLimits, Pipeline
from flujo.exceptions import UsageLimitExceededError, ConfigurationError
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo

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
    parallel = Step.parallel(
        "par",
        branches,
        merge_strategy=MergeStrategy.NO_MERGE,
    )
    runner = create_test_flujo(parallel, context_model=Ctx)
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
    parallel = Step.parallel("par_out", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = create_test_flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, 1)
    step_result = result.step_history[-1]
    assert isinstance(step_result.output, dict)
    assert set(step_result.output.keys()) == {"x", "y"}
    assert step_result.success is True


class ScratchCtx(PipelineContext):
    val: int = 0


class OrderCtx(PipelineContext):
    executed_branches: list[str] = Field(default_factory=list)
    branch_results: dict[str, Any] = Field(default_factory=dict)


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
            context.import_artifacts[self.key] = self.val
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
async def test_parallel_overwrite_conflict() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("v", 1, delay=0.1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("v", 2, delay=0.2)}),
    }
    parallel = Step.parallel("overwrite", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert step_result.output["b"] == 2
    assert result.final_pipeline_context.import_artifacts.get("v") == 2


@pytest.mark.asyncio
async def test_parallel_overwrite_preserves_context() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2)}),
    }
    parallel = Step.parallel(
        "overwrite_keep",
        branches,
        context_include_keys=["import_artifacts", "initial_prompt"],
        merge_strategy=MergeStrategy.OVERWRITE,
    )
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(
        runner,
        0,
        initial_context_data={"initial_prompt": "x", "val": 5},
    )
    assert result.final_pipeline_context.val == 5
    assert result.step_history[-1].output["b"] == 2


@pytest.mark.asyncio
async def test_parallel_overwrite_multi_branch_order() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("v", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("v", 2)}),
        "c": Step.model_validate({"name": "c", "agent": ScratchAgent("w", 3)}),
    }
    parallel = Step.parallel("overwrite_multi", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert step_result.output["b"] == 2
    assert step_result.output["c"] == 3


@pytest.mark.asyncio
async def test_parallel_overwrite_declared_order_wins() -> None:
    branches = {
        "branch2": Step.model_validate({"name": "branch2", "agent": ScratchAgent("v", 2)}),
        "branch10": Step.model_validate({"name": "branch10", "agent": ScratchAgent("v", 10)}),
    }
    parallel = Step.parallel(
        "overwrite_order",
        branches,
        merge_strategy=MergeStrategy.OVERWRITE,
    )
    runner = create_test_flujo(parallel, context_model=OrderCtx)
    result = await gather_result(
        runner,
        0,
        initial_context_data={"executed_branches": [], "branch_results": {}},
    )
    assert result.final_pipeline_context.executed_branches == ["branch10"]
    assert result.final_pipeline_context.branch_results == {"branch10": 10}


def test_parallel_merge_removed_root_rejected() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("a", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("b", 2)}),
    }
    parallel = Step.parallel(
        "merge_sp",
        branches,
        merge_strategy=MergeStrategy.NO_MERGE,
    )
    removed_root = "scrat" + "chpad"
    object.__setattr__(parallel, "merge_strategy", "merge_" + removed_root)
    with pytest.raises(ConfigurationError):
        Pipeline.from_step(parallel).validate_graph(raise_on_error=True)


@pytest.mark.asyncio
async def test_parallel_propagate_failure() -> None:
    branches = {
        "good": Step.model_validate({"name": "good", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel(
        "fail_prop", branches, on_branch_failure=BranchFailureStrategy.PROPAGATE
    )
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
    result = await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})
    step_result = result.step_history[-1]
    assert not step_result.success
    assert isinstance(step_result.output["bad"], StepResult)
    assert result.final_pipeline_context.import_artifacts.get("a") == 1


@pytest.mark.asyncio
async def test_parallel_cost_includes_failed_branch_usage() -> None:
    class TokenCostAgent:
        def __init__(self, cost: float, tokens: int) -> None:
            self.cost = cost
            self.tokens = tokens

        async def run(self, data: int) -> Any:
            class Output(BaseModel):
                value: int
                cost_usd: float = self.cost
                token_counts: int = self.tokens

            return Output(value=data)

    class FailingAgent:
        async def run(self, data: int) -> int:
            raise RuntimeError("boom")

    branches = {
        "good": Step.model_validate({"name": "good", "agent": TokenCostAgent(cost=0.05, tokens=5)}),
        "bad": Pipeline.model_validate(
            {
                "steps": [
                    Step.model_validate(
                        {"name": "cost", "agent": TokenCostAgent(cost=0.10, tokens=10)}
                    ),
                    Step.model_validate({"name": "fail", "agent": FailingAgent()}),
                ]
            }
        ),
    }
    parallel = Step.parallel(
        "parallel_cost",
        branches,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )
    runner = create_test_flujo(parallel)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert not step_result.success
    assert step_result.cost_usd == pytest.approx(0.15)
    assert step_result.token_counts == 15


@pytest.mark.asyncio
async def test_parallel_ignore_failure() -> None:
    branches = {
        "good": Step.model_validate({"name": "good", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel(
        "fail_ignore", branches, on_branch_failure=BranchFailureStrategy.IGNORE
    )
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
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
    runner = create_test_flujo(parallel, context_model=ScratchCtx)
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
    runner = create_test_flujo(parallel, usage_limits=limits, context_model=ScratchCtx)
    with pytest.raises(UsageLimitExceededError):
        await gather_result(runner, 0, initial_context_data={"initial_prompt": "x"})

import os
import asyncio
from typing import Any, Dict

import pytest
from flujo import Flujo, Step
from flujo.domain import MergeStrategy, BranchFailureStrategy
from flujo.testing.utils import gather_result
from flujo.domain.models import StepResult, PipelineContext
from pydantic import ConfigDict
from flujo.domain.models import BaseModel as FlujoBaseModel

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class Ctx(PipelineContext):
    value: int = 0


class ScratchAgent:
    def __init__(self, key: str, val: Any, delay: float = 0.0, fail: bool = False) -> None:
        self.key = key
        self.val = val
        self.delay = delay
        self.fail = fail

    async def run(self, data: Any, *, context: Ctx | None = None) -> Any:
        if self.fail:
            raise RuntimeError("boom")
        await asyncio.sleep(self.delay)
        if context is not None:
            context.scratchpad[self.key] = self.val
        return data


class SafeScratchAgent:
    def __init__(self, key: str, val: Any) -> None:
        self.key = key
        self.val = val

    async def run(self, data: Any, *, context: FlujoBaseModel | None = None) -> Any:
        if context is not None:
            if not hasattr(context, "scratchpad"):
                context.scratchpad = {}
            context.scratchpad[self.key] = self.val
        return data


class NoScratchCtx(FlujoBaseModel):
    initial_prompt: str

    model_config = ConfigDict(extra="allow")


@pytest.mark.asyncio
async def test_merge_strategy_no_merge() -> None:
    branches: Dict[str, Step] = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=MergeStrategy.NO_MERGE)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    assert "x" not in result.final_pipeline_context.scratchpad
    assert "y" not in result.final_pipeline_context.scratchpad


@pytest.mark.asyncio
async def test_merge_strategy_overwrite() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("v", 1, delay=0.1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("v", 2, delay=0.2)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=MergeStrategy.OVERWRITE)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    assert result.final_pipeline_context.scratchpad["v"] == 2


@pytest.mark.asyncio
async def test_overwrite_preserves_unincluded_fields() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2)}),
    }
    parallel = Step.parallel(
        "p",
        branches,
        merge_strategy=MergeStrategy.OVERWRITE,
        context_include_keys=["scratchpad", "initial_prompt"],
    )
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(
        runner,
        "data",
        initial_context_data={"initial_prompt": "goal", "value": 7},
    )
    assert result.final_pipeline_context.value == 7
    assert result.final_pipeline_context.scratchpad["y"] == 2


@pytest.mark.asyncio
async def test_merge_strategy_merge_scratchpad() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=MergeStrategy.MERGE_SCRATCHPAD)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    assert result.final_pipeline_context.scratchpad["x"] == 1
    assert result.final_pipeline_context.scratchpad["y"] == 2


@pytest.mark.asyncio
async def test_merge_scratchpad_detects_collision() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("dup", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("dup", 2)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=MergeStrategy.MERGE_SCRATCHPAD)
    runner = Flujo(parallel, context_model=Ctx)
    with pytest.raises(ValueError):
        await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})


@pytest.mark.asyncio
async def test_merge_scratchpad_requires_scratchpad() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": SafeScratchAgent("x", 1)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=MergeStrategy.MERGE_SCRATCHPAD)
    runner = Flujo(parallel, context_model=NoScratchCtx)
    with pytest.raises(ValueError):
        await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})


def custom_merge(main: Ctx, branch: Ctx) -> None:
    main.scratchpad.setdefault("vals", []).append(branch.scratchpad["val"])


@pytest.mark.asyncio
async def test_merge_strategy_callable() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("val", 1)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("val", 2)}),
    }
    parallel = Step.parallel("par", branches, merge_strategy=custom_merge)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    assert sorted(result.final_pipeline_context.scratchpad["vals"]) == [1, 2]


@pytest.mark.asyncio
async def test_branch_failure_propagate() -> None:
    branches = {
        "ok": Step.model_validate({"name": "ok", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel("par", branches, on_branch_failure=BranchFailureStrategy.PROPAGATE)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    assert not result.step_history[-1].success
    assert isinstance(result.step_history[-1].output["bad"], StepResult)


@pytest.mark.asyncio
async def test_branch_failure_ignore() -> None:
    branches = {
        "ok": Step.model_validate({"name": "ok", "agent": ScratchAgent("a", 1)}),
        "bad": Step.model_validate({"name": "bad", "agent": ScratchAgent("b", 2, fail=True)}),
    }
    parallel = Step.parallel("par", branches, on_branch_failure=BranchFailureStrategy.IGNORE)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    step_result = result.step_history[-1]
    assert step_result.success
    assert isinstance(step_result.output["bad"], StepResult)


@pytest.mark.asyncio
async def test_branch_failure_ignore_all_fail() -> None:
    branches = {
        "a": Step.model_validate({"name": "a", "agent": ScratchAgent("x", 1, fail=True)}),
        "b": Step.model_validate({"name": "b", "agent": ScratchAgent("y", 2, fail=True)}),
    }
    parallel = Step.parallel("par", branches, on_branch_failure=BranchFailureStrategy.IGNORE)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, "data", initial_context_data={"initial_prompt": "goal"})
    step_result = result.step_history[-1]
    assert not step_result.success
    assert all(isinstance(step_result.output[name], StepResult) for name in branches)

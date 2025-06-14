import pytest
from pydantic import BaseModel

from pydantic_ai_orchestrator.domain import Step, Pipeline, ConditionalStep
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.testing.utils import StubAgent, DummyPlugin
from pydantic_ai_orchestrator.domain.plugins import PluginOutcome


class EchoAgent:
    async def run(self, data, **kwargs):
        return data


@pytest.mark.asyncio
async def test_branch_a_executes() -> None:
    classify = Step("classify", StubAgent(["a"]))
    branches = {
        "a": Pipeline.from_step(Step("a", StubAgent(["A"]))),
        "b": Pipeline.from_step(Step("b", StubAgent(["B"]))),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = PipelineRunner(classify >> branch_step)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.output == "A"
    assert step_result.metadata_["executed_branch_key"] == "a"


@pytest.mark.asyncio
async def test_branch_b_executes() -> None:
    classify = Step("classify", StubAgent(["b"]))
    branches = {
        "a": Pipeline.from_step(Step("a", StubAgent(["A"]))),
        "b": Pipeline.from_step(Step("b", StubAgent(["B"]))),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = PipelineRunner(classify >> branch_step)
    result = await runner.run_async("in")
    assert result.step_history[-1].output == "B"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "b"


@pytest.mark.asyncio
async def test_default_branch_used() -> None:
    classify = Step("classify", StubAgent(["x"]))
    branches = {
        "a": Pipeline.from_step(Step("a", StubAgent(["A"]))),
    }
    default = Pipeline.from_step(Step("def", StubAgent(["DEF"])))
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
        default_branch_pipeline=default,
    )
    runner = PipelineRunner(classify >> branch_step)
    result = await runner.run_async("in")
    assert result.step_history[-1].output == "DEF"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "x"


@pytest.mark.asyncio
async def test_no_match_no_default_fails() -> None:
    classify = Step("classify", StubAgent(["x"]))
    branches = {"a": Pipeline.from_step(Step("a", StubAgent(["A"])))}
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = PipelineRunner(classify >> branch_step)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "no default" in step_result.feedback.lower()


class FlagCtx(BaseModel):
    flag: str = "a"


@pytest.mark.asyncio
async def test_condition_uses_context() -> None:
    classify = Step("classify", StubAgent(["ignored"]))
    branches = {
        "a": Pipeline.from_step(Step("a", StubAgent(["A"]))),
        "b": Pipeline.from_step(Step("b", StubAgent(["B"]))),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: ctx.flag if ctx else "a",
        branches=branches,
    )
    runner = PipelineRunner(
        classify >> branch_step, context_model=FlagCtx, initial_context_data={"flag": "b"}
    )
    result = await runner.run_async("in")
    assert result.step_history[-1].output == "B"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "b"


@pytest.mark.asyncio
async def test_mappers_applied() -> None:
    branches = {
        "x": Pipeline.from_step(Step("inc", EchoAgent())),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "x",
        branches=branches,
        branch_input_mapper=lambda inp, ctx: inp + 1,
        branch_output_mapper=lambda out, key, ctx: out * 10,
    )
    runner = PipelineRunner(branch_step)
    result = await runner.run_async(1)
    assert result.step_history[-1].output == 20


@pytest.mark.asyncio
async def test_failure_in_branch_propagates() -> None:
    fail_plugin = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    bad_step = Step("bad", StubAgent(["oops"]), plugins=[fail_plugin])
    branches = {"a": Pipeline.from_step(bad_step)}
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
    )
    runner = PipelineRunner(branch_step)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "bad" in step_result.feedback


@pytest.mark.asyncio
async def test_condition_exception_fails_step() -> None:
    branches = {"a": Pipeline.from_step(Step("a", StubAgent(["A"])))}

    def condition(_: str, __: FlagCtx | None) -> str:
        raise RuntimeError("boom")

    branch_step = Step.branch_on(
        name="branch",
        condition_callable=condition,
        branches=branches,
    )
    runner = PipelineRunner(branch_step)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "boom" in step_result.feedback

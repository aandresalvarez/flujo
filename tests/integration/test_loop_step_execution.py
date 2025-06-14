import pytest
from pydantic import BaseModel

from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.domain import Step, Pipeline, LoopStep
from pydantic_ai_orchestrator.testing.utils import StubAgent, DummyPlugin
from pydantic_ai_orchestrator.domain.plugins import PluginOutcome


class IncrementAgent:
    async def run(self, data: int, **kwargs) -> int:
        return data + 1


@pytest.mark.asyncio
async def test_basic_loop_until_condition_met() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 3,
        max_loops=5,
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 3
    assert step_result.output == 3


@pytest.mark.asyncio
async def test_loop_max_loops_reached() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 10,
        max_loops=2,
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert step_result.attempts == 2


class Ctx(BaseModel):
    counter: int = 0


@pytest.mark.asyncio
async def test_loop_with_context_modification() -> None:
    class IncRecordAgent:
        async def run(self, x: int, *, pipeline_context: Ctx | None = None) -> int:
            if pipeline_context:
                pipeline_context.counter += 1
            return x + 1

    body = Pipeline.from_step(Step("inc", IncRecordAgent()))
    loop = Step.loop_until(
        name="loop_ctx",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: ctx and ctx.counter >= 2,
        max_loops=5,
    )
    runner = PipelineRunner(loop, context_model=Ctx)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert result.final_pipeline_context.counter == 2


@pytest.mark.asyncio
async def test_loop_step_error_in_exit_condition_callable() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))

    def bad_exit(_: int, __: Ctx | None) -> bool:
        raise RuntimeError("boom")

    loop = Step.loop_until(
        name="loop_error_exit",
        loop_body_pipeline=body,
        exit_condition_callable=bad_exit,
        max_loops=3,
    )
    after = Step("after", IncrementAgent())
    runner = PipelineRunner(loop >> after, context_model=Ctx)
    result = await runner.run_async(0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert "boom" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_initial_input_mapper() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))

    def bad_initial_mapper(_: int, __: Ctx | None) -> int:
        raise RuntimeError("init map err")

    loop = Step.loop_until(
        name="loop_bad_init",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=2,
        initial_input_to_loop_body_mapper=bad_initial_mapper,
    )
    after = Step("after", IncrementAgent())
    runner = PipelineRunner(loop >> after, context_model=Ctx)
    result = await runner.run_async(0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert step_result.attempts == 0
    assert "init map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_iteration_input_mapper() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))

    def iteration_mapper(_: int, __: Ctx | None, ___: int) -> int:
        raise RuntimeError("iter map err")

    loop = Step.loop_until(
        name="loop_iter_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 5,
        max_loops=3,
        iteration_input_mapper=iteration_mapper,
    )
    runner = PipelineRunner(loop, context_model=Ctx)
    result = await runner.run_async(0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert step_result.attempts == 1
    assert "iter map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_loop_output_mapper() -> None:
    body = Pipeline.from_step(Step("inc", IncrementAgent()))

    def bad_output_mapper(_: int, __: Ctx | None) -> int:
        raise RuntimeError("output map err")

    loop = Step.loop_until(
        name="loop_out_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 1,
        max_loops=2,
        loop_output_mapper=bad_output_mapper,
    )
    after = Step("after", IncrementAgent())
    runner = PipelineRunner(loop >> after, context_model=Ctx)
    result = await runner.run_async(0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert "output map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_body_failure_with_robust_exit_condition() -> None:
    fail_plugin = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    bad_step = Step("bad", StubAgent(["oops"]), plugins=[fail_plugin])
    body = Pipeline.from_step(bad_step)

    loop = Step.loop_until(
        name="loop_body_fail", loop_body_pipeline=body, exit_condition_callable=lambda out, ctx: True
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "last iteration body failed" in (step_result.feedback or "")


@pytest.mark.asyncio
async def test_loop_step_body_failure_causing_exit_condition_error() -> None:
    fail_plugin = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    bad_step = Step("bad", StubAgent([{}]), plugins=[fail_plugin])
    body = Pipeline.from_step(bad_step)

    def exit_condition(out: dict, _: Ctx | None) -> bool:
        return out["missing"]  # will raise KeyError

    loop = Step.loop_until(
        name="loop_exit_err",
        loop_body_pipeline=body,
        exit_condition_callable=exit_condition,
    )
    runner = PipelineRunner(loop)
    result = await runner.run_async("in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "exception" in (step_result.feedback or "").lower()

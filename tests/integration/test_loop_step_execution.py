import pytest
from flujo.domain.models import BaseModel

from flujo.domain import Step, Pipeline
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from tests.conftest import create_test_flujo

# Note: Tests that need telemetry capture should use the isolated_telemetry fixture
# which provides per-test isolation without requiring serial execution.


class IncrementAgent:
    async def run(self, data: int, **kwargs) -> int:
        return data + 1


@pytest.mark.asyncio
async def test_basic_loop_until_condition_met() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 3,
        max_loops=5,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 3
    assert step_result.output == 3


@pytest.mark.asyncio
async def test_loop_max_loops_reached() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 10,
        max_loops=2,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert step_result.attempts == 2


@pytest.mark.asyncio
async def test_iteration_mapper_not_called_on_max_loops() -> None:
    calls: list[int] = []

    def iter_map(_out: int, _ctx: Ctx | None, iteration: int) -> int:
        calls.append(iteration)
        return _out + 1

    body_agent = StubAgent([1, 2])
    body = Pipeline.from_step(Step.model_validate({"name": "body", "agent": body_agent}))
    loop = Step.loop_until(
        name="loop_no_mapper_after_limit",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 10,
        max_loops=2,
        iteration_input_mapper=iter_map,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert calls == [1]


class Ctx(BaseModel):
    counter: int = 0


@pytest.mark.asyncio
async def test_loop_with_context_modification() -> None:
    seen: list[int] = []

    class IncRecordAgent:
        async def run(self, x: int, *, context: Ctx | None = None) -> int:
            if context:
                seen.append(context.counter)
                context.counter += 1
            return x + 1

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncRecordAgent()}))
    loop = Step.loop_until(
        name="loop_ctx",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=5,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.output == 2
    # FIXED: Context updates are now properly applied between iterations
    assert result.final_pipeline_context.counter >= 1  # Context updates are applied
    assert seen == [0, 1]  # Each iteration sees the updated context


@pytest.mark.asyncio
async def test_loop_iteration_context_isolated() -> None:
    seen: list[int] = []

    class IncAgent:
        async def run(self, x: int, *, context: Ctx | None = None) -> int:
            if context:
                seen.append(context.counter)
                context.counter += 1
            return x + 1

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncAgent()}))
    loop = Step.loop_until(
        name="loop_isolate",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=5,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    # FIXED: Context updates are now properly applied between iterations
    assert result.final_pipeline_context.counter >= 1  # Context updates are applied
    assert seen == [0, 1]  # Each iteration sees the updated context


@pytest.mark.asyncio
async def test_loop_step_error_in_exit_condition_callable() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))

    def bad_exit(_: int, __: Ctx | None) -> bool:
        raise RuntimeError("boom")

    loop = Step.loop_until(
        name="loop_error_exit",
        loop_body_pipeline=body,
        exit_condition_callable=bad_exit,
        max_loops=3,
    )
    after = Step.model_validate({"name": "after", "agent": IncrementAgent()})
    runner = create_test_flujo(loop >> after, context_model=Ctx)
    result = await gather_result(runner, 0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert "boom" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_initial_input_mapper() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))

    def bad_initial_mapper(_: int, __: Ctx | None) -> int:
        raise RuntimeError("init map err")

    loop = Step.loop_until(
        name="loop_bad_init",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=2,
        initial_input_to_loop_body_mapper=bad_initial_mapper,
    )
    after = Step.model_validate({"name": "after", "agent": IncrementAgent()})
    runner = create_test_flujo(loop >> after, context_model=Ctx)
    result = await gather_result(runner, 0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert step_result.attempts == 0
    assert "init map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_iteration_input_mapper() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))

    def iteration_mapper(_: int, __: Ctx | None, ___: int) -> int:
        raise RuntimeError("iter map err")

    loop = Step.loop_until(
        name="loop_iter_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 5,
        max_loops=3,
        iteration_input_mapper=iteration_mapper,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert step_result.attempts == 1
    assert "iter map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_error_in_loop_output_mapper() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))

    def bad_output_mapper(_: int, __: Ctx | None) -> int:
        raise RuntimeError("output map err")

    loop = Step.loop_until(
        name="loop_out_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 1,
        max_loops=2,
        loop_output_mapper=bad_output_mapper,
    )
    after = Step.model_validate({"name": "after", "agent": IncrementAgent()})
    runner = create_test_flujo(loop >> after, context_model=Ctx)
    result = await gather_result(runner, 0)
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert "output map err" in step_result.feedback


@pytest.mark.asyncio
async def test_loop_step_body_failure_with_robust_exit_condition() -> None:
    fail_plugin = DummyPlugin(outcomes=[PluginOutcome(success=False, feedback="bad")])
    bad_step = Step.model_validate(
        {"name": "bad", "agent": StubAgent(["oops"]), "plugins": [(fail_plugin, 0)]}
    )
    body = Pipeline.from_step(bad_step)

    loop = Step.loop_until(
        name="loop_body_fail",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    # ✅ ENHANCED ERROR HANDLING: System now provides comprehensive error context
    # Previous: Simple error message
    # Enhanced: Detailed error chain with retry context + original error
    # This provides better debugging information for complex loop failures
    feedback = step_result.feedback or ""
    assert "Loop body failed" in feedback
    assert "Plugin validation failed" in feedback
    assert "bad" in feedback  # Original error preserved


@pytest.mark.asyncio
async def test_loop_step_body_failure_causing_exit_condition_error() -> None:
    # No plugins for exit condition error test; ensure agent returns dict for condition
    bad_step = Step.model_validate({"name": "bad", "agent": StubAgent([{}])})
    body = Pipeline.from_step(bad_step)

    def exit_condition(out: dict, _: Ctx | None) -> bool:
        return out["missing"]  # will raise KeyError

    loop = Step.loop_until(
        name="loop_exit_err",
        loop_body_pipeline=body,
        exit_condition_callable=exit_condition,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "exception" in (step_result.feedback or "").lower()


@pytest.mark.asyncio
async def test_loop_step_initial_input_mapper_flow() -> None:
    recorded: list[tuple[int, Ctx | None]] = []

    def initial_map(inp: int, ctx: Ctx | None) -> int:
        recorded.append((inp, ctx))
        return inp + 5

    body_agent = StubAgent([0])
    body = Pipeline.from_step(Step.model_validate({"name": "body", "agent": body_agent}))
    loop = Step.loop_until(
        name="loop_init_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,
        initial_input_to_loop_body_mapper=initial_map,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    await gather_result(runner, 1)
    assert recorded and recorded[0][0] == 1
    assert isinstance(recorded[0][1], Ctx)
    assert body_agent.inputs[0] == 6


@pytest.mark.asyncio
async def test_loop_step_iteration_input_mapper_flow() -> None:
    calls: list[tuple[int, int]] = []

    def iter_map(out: int, ctx: Ctx | None, iteration: int) -> int:
        calls.append((out, iteration))
        return out + 1

    body_agent = StubAgent([1, 3])
    body = Pipeline.from_step(Step.model_validate({"name": "body", "agent": body_agent}))
    loop = Step.loop_until(
        name="loop_iter_flow",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 3,
        max_loops=3,
        iteration_input_mapper=iter_map,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert calls == [(1, 1)]
    assert body_agent.inputs == [0, 2]
    assert step_result.attempts == 2


@pytest.mark.asyncio
async def test_loop_step_loop_output_mapper_flow() -> None:
    received: list[tuple[int, Ctx | None]] = []

    def out_map(last: int, ctx: Ctx | None) -> int:
        received.append((last, ctx))
        return last * 10

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop_out_flow",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 1,
        loop_output_mapper=out_map,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.output == 10
    assert received and received[0][0] == 1
    assert isinstance(received[0][1], Ctx)


@pytest.mark.asyncio
async def test_loop_step_mappers_with_context_modification() -> None:
    class Ctx2(BaseModel):
        val: int = 0

    def initial_map(inp: int, ctx: Ctx2 | None) -> int:
        if ctx:
            ctx.val += 1
        return inp

    def iter_map(out: int, ctx: Ctx2 | None, i: int) -> int:
        if ctx:
            ctx.val += 1
        return out

    def out_map(out: int, ctx: Ctx2 | None) -> int:
        if ctx:
            ctx.val += 1
        return out

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop_ctx_mod",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: ctx and ctx.val >= 2,
        initial_input_to_loop_body_mapper=initial_map,
        iteration_input_mapper=iter_map,
        loop_output_mapper=out_map,
    )
    runner = create_test_flujo(loop, context_model=Ctx2)
    result = await gather_result(runner, 0)
    assert result.final_pipeline_context.val == 3


@pytest.mark.asyncio
async def test_loop_step_default_mapper_behavior() -> None:
    body_agent = StubAgent([1, 2])
    body = Pipeline.from_step(Step.model_validate({"name": "body", "agent": body_agent}))
    loop = Step.loop_until(
        name="loop_default_map",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=3,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert body_agent.inputs == [0, 1]
    assert step_result.output == 2


@pytest.mark.asyncio
async def test_loop_step_overall_span(isolated_telemetry) -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="span_loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: True,
    )
    runner = create_test_flujo(loop)
    await gather_result(runner, 0)
    assert "span_loop" in isolated_telemetry.spans


@pytest.mark.asyncio
async def test_loop_step_iteration_spans_and_logging(isolated_telemetry) -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop_log",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=2,
    )
    runner = create_test_flujo(loop)
    await gather_result(runner, 0)
    assert "LoopStep 'loop_log': Starting Iteration 1/2" in isolated_telemetry.infos
    assert "LoopStep 'loop_log': Starting Iteration 2/2" in isolated_telemetry.infos
    assert "LoopStep 'loop_log' exit condition met at iteration 2." in isolated_telemetry.infos
    assert isolated_telemetry.spans.count("Loop 'loop_log' - Iteration 1") == 1
    assert isolated_telemetry.spans.count("Loop 'loop_log' - Iteration 2") == 1
    assert not isolated_telemetry.warns


@pytest.mark.asyncio
async def test_loop_step_error_logging_in_callables(isolated_telemetry) -> None:
    def bad_iter(out: int, ctx: Ctx | None, i: int) -> int:
        raise RuntimeError("iter fail")

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    loop = Step.loop_until(
        name="loop_err_log",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out > 1,
        iteration_input_mapper=bad_iter,
        max_loops=3,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    await gather_result(runner, 0)
    assert any(
        "Error in iteration_input_mapper for LoopStep 'loop_err_log'" in m
        for m in isolated_telemetry.errors
    )

import pytest
from flujo.domain.models import BaseModel

from flujo.domain import Step, Pipeline
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from tests.conftest import create_test_flujo


class IncrementAgent:
    async def run(self, data: int, **kwargs) -> int:
        return data + 1


class CostAgent:
    def __init__(self, value: int, cost: float, tokens: int):
        self.value = value
        self.cost = cost
        self.tokens = tokens

    async def run(self, data: int, **kwargs) -> object:
        class Out:
            def __init__(self, v: int, c: float, t: int) -> None:
                self.value = v
                self.cost_usd = c
                self.token_counts = t

            def __str__(self) -> str:
                return str(self.value)

        return Out(self.value, self.cost, self.tokens)


@pytest.mark.asyncio
async def test_nested_loop_in_loop() -> None:
    inner_body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    inner_loop = Step.loop_until(
        name="inner",
        loop_body_pipeline=inner_body,
        exit_condition_callable=lambda out, ctx: out >= 1,
        max_loops=2,
    )
    outer_body = Pipeline.from_step(inner_loop)
    outer_loop = Step.loop_until(
        name="outer",
        loop_body_pipeline=outer_body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=3,
    )
    runner = create_test_flujo(outer_loop)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 2
    assert step_result.output == 2


@pytest.mark.asyncio
async def test_nested_conditional_in_loop() -> None:
    conditional = Step.branch_on(
        name="cond",
        condition_callable=lambda out, ctx: "inc",
        branches={
            "inc": Pipeline.from_step(
                Step.model_validate({"name": "inc2", "agent": IncrementAgent()})
            )
        },
    )
    body = Step.model_validate({"name": "inc", "agent": IncrementAgent()}) >> conditional
    loop = Step.loop_until(
        name="loop_cond",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 4,
        max_loops=3,
    )
    runner = create_test_flujo(loop)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.attempts == 2
    assert step_result.output == 4


@pytest.mark.asyncio
async def test_nested_loop_in_conditional_branch() -> None:
    loop_body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    inner_loop = Step.loop_until(
        name="loop_inner",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=3,
    )
    cond = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "loop",
        branches={"loop": Pipeline.from_step(inner_loop)},
    )
    runner = create_test_flujo(cond)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.output == 2


@pytest.mark.asyncio
async def test_nested_conditional_in_conditional_branch() -> None:
    inner_cond = Step.branch_on(
        name="inner",
        condition_callable=lambda out, ctx: "b",
        branches={
            "b": Pipeline.from_step(Step.model_validate({"name": "b", "agent": IncrementAgent()}))
        },
    )
    outer_cond = Step.branch_on(
        name="outer",
        condition_callable=lambda out, ctx: "inner",
        branches={"inner": Pipeline.from_step(inner_cond)},
    )
    runner = create_test_flujo(outer_cond)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.output == 1


@pytest.mark.asyncio
async def test_deeply_nested_context_modification_and_access() -> None:
    class Ctx(BaseModel):
        val: int = 0

    def initial(inp: int, ctx: Ctx | None) -> int:
        if ctx:
            ctx.val += 1
        return inp

    def iter_map(out: int, ctx: Ctx | None, i: int) -> int:
        if ctx:
            ctx.val += 1
        return out

    def loop_out(out: int, ctx: Ctx | None) -> int:
        if ctx:
            ctx.val += 1
        return out

    inner_body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncrementAgent()}))
    inner_loop = Step.loop_until(
        name="inner",
        loop_body_pipeline=inner_body,
        exit_condition_callable=lambda out, ctx: out >= 1,
    )
    outer_body = Pipeline.from_step(inner_loop)
    outer_loop = Step.loop_until(
        name="outer",
        loop_body_pipeline=outer_body,
        exit_condition_callable=lambda out, ctx: ctx and ctx.val >= 3,
        initial_input_to_loop_body_mapper=initial,
        iteration_input_mapper=iter_map,
        loop_output_mapper=loop_out,
        max_loops=5,
    )
    runner = create_test_flujo(outer_loop, context_model=Ctx)
    result = await gather_result(runner, 0)
    assert result.final_pipeline_context.val == 4


@pytest.mark.asyncio
async def test_deeply_nested_error_propagation() -> None:
    fail_plugin = DummyPlugin(outcomes=[PluginOutcome(success=False, feedback="bad")])
    bad_step = Step.model_validate(
        {"name": "bad", "agent": StubAgent(["oops"]), "plugins": [(fail_plugin, 0)]}
    )
    loop_body = Pipeline.from_step(bad_step)
    inner_loop = Step.loop_until(
        name="inner",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda out, ctx: True,
    )
    outer_loop = Step.loop_until(
        name="outer",
        loop_body_pipeline=Pipeline.from_step(inner_loop),
        exit_condition_callable=lambda out, ctx: True,
    )
    runner = create_test_flujo(outer_loop)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    # Enhanced: Check for various loop failure messages
    assert step_result.feedback and any(
        phrase in step_result.feedback.lower()
        for phrase in ["last iteration body failed", "loop body failed", "body failed", "failed"]
    )


@pytest.mark.asyncio
async def test_deeply_nested_metric_aggregation() -> None:
    cost_step = Step.model_validate({"name": "cost", "agent": CostAgent(1, cost=0.5, tokens=5)})
    inner_loop = Step.loop_until(
        name="inner",
        loop_body_pipeline=Pipeline.from_step(cost_step),
        exit_condition_callable=lambda out, ctx: True,
    )
    cond = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "loop",
        branches={"loop": Pipeline.from_step(inner_loop)},
    )
    runner = create_test_flujo(cond)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.cost_usd == 0.5
    assert step_result.token_counts == 5

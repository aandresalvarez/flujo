import pytest
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import step, StandardStep, HumanInTheLoopStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.steps.cache_step import CacheStep
from flujo.registry import CallableRegistry
from flujo.application.runner import Flujo


@pytest.mark.asyncio
async def test_ir_roundtrip_callable_step():
    # Define a simple callable step
    @step
    async def echo_step(data):
        return f"Echo: {data}"

    pipeline = Pipeline.from_step(echo_step)
    callable_registry = CallableRegistry()

    # DSL -> IR
    ir = pipeline.to_model(callable_registry)
    assert ir.steps, "IR should have steps"
    assert hasattr(ir.steps[0], "agent"), "IR step should have agent"
    assert ir.steps[0].agent.agent_type == "callable", "Agent type should be callable"

    # IR -> DSL
    rehydrated_pipeline = Pipeline.from_model(ir, callable_registry=callable_registry)
    assert rehydrated_pipeline.steps, "Rehydrated pipeline should have steps"
    rehydrated_step = rehydrated_pipeline.steps[0]
    assert callable(getattr(rehydrated_step.agent, "run", None)), (
        "Rehydrated agent should have a run method"
    )

    # Execution
    result = await rehydrated_step.arun("test")
    assert result == "Echo: test", f"Expected 'Echo: test', got {result}"


@pytest.mark.asyncio
async def test_ir_roundtrip_standard_agent_step():
    class DummyAgent:
        async def run(self, data, **kwargs):
            return f"Agent: {data}"

    step_obj = StandardStep(name="agent_step", agent=DummyAgent())
    pipeline = Pipeline.from_step(step_obj)
    ir = pipeline.to_model()
    # Provide agent registry for rehydration
    agent_registry = {"DummyAgent": DummyAgent}
    rehydrated_pipeline = Pipeline.from_model(ir, agent_registry=agent_registry)
    runner = Flujo(rehydrated_pipeline, agent_registry=agent_registry)
    result = await runner.run_async("foo").__anext__()
    assert result.step_history[-1].output == "Agent: foo"


@pytest.mark.asyncio
async def test_ir_roundtrip_loop_step():
    @step
    async def inc_step(data):
        return data + 1

    def exit_cond(out, ctx):
        return out >= 3

    loop = LoopStep(
        name="loop",
        loop_body_pipeline=Pipeline.from_step(inc_step),
        exit_condition_callable=exit_cond,
        max_loops=5,
    )
    pipeline = Pipeline.from_step(loop)
    callable_registry = CallableRegistry()
    ir = pipeline.to_model(callable_registry)
    rehydrated_pipeline = Pipeline.from_model(ir, callable_registry=callable_registry)
    # Use Flujo runner to execute the pipeline
    runner = Flujo(rehydrated_pipeline)
    result = await runner.run_async(0).__anext__()
    assert result.step_history[-1].output >= 3


@pytest.mark.asyncio
async def test_ir_roundtrip_conditional_step():
    @step
    async def a_step(data):
        return f"A: {data}"

    @step
    async def b_step(data):
        return f"B: {data}"

    def cond(data, ctx):
        return "a" if data < 0 else "b"

    cond_step = ConditionalStep(
        name="cond",
        condition_callable=cond,
        branches={"a": Pipeline.from_step(a_step), "b": Pipeline.from_step(b_step)},
    )
    pipeline = Pipeline.from_step(cond_step)
    callable_registry = CallableRegistry()
    ir = pipeline.to_model(callable_registry)
    rehydrated_pipeline = Pipeline.from_model(ir, callable_registry=callable_registry)
    runner = Flujo(rehydrated_pipeline)
    result_a = await runner.run_async(-1).__anext__()
    result_b = await runner.run_async(1).__anext__()
    assert result_a.step_history[-1].output == "A: -1"
    assert result_b.step_history[-1].output == "B: 1"


@pytest.mark.asyncio
async def test_ir_roundtrip_parallel_step():
    @step
    async def s1(data):
        return data + 1

    @step
    async def s2(data):
        return data * 2

    par = ParallelStep(
        name="par",
        branches={"one": Pipeline.from_step(s1), "two": Pipeline.from_step(s2)},
    )
    pipeline = Pipeline.from_step(par)
    callable_registry = CallableRegistry()
    ir = pipeline.to_model(callable_registry)
    rehydrated_pipeline = Pipeline.from_model(ir, callable_registry=callable_registry)
    runner = Flujo(rehydrated_pipeline)
    result = await runner.run_async(3).__anext__()
    # ParallelStep returns dict of outputs in step_history[-1].output
    out = result.step_history[-1].output
    assert out["one"] == 4
    assert out["two"] == 6


@pytest.mark.asyncio
async def test_ir_roundtrip_human_in_the_loop():
    hitl = HumanInTheLoopStep(
        name="hitl",
        message_for_user="Please provide input",
        input_schema=None,
    )
    pipeline = Pipeline.from_step(hitl)
    ir = pipeline.to_model()
    rehydrated_pipeline = Pipeline.from_model(ir)
    # Just check structure, do not call arun (no agent)
    assert isinstance(rehydrated_pipeline.steps[0], HumanInTheLoopStep)
    assert rehydrated_pipeline.steps[0].message_for_user == "Please provide input"


@pytest.mark.asyncio
async def test_ir_roundtrip_cache_step():
    @step
    async def echo(data):
        return f"CACHED: {data}"

    cache_step = CacheStep(
        name="cache",
        wrapped_step=echo,  # echo is now a StandardStep instance from the @step decorator
    )
    pipeline = Pipeline.from_step(cache_step)
    callable_registry = CallableRegistry()
    ir = pipeline.to_model(callable_registry)
    rehydrated_pipeline = Pipeline.from_model(ir, callable_registry=callable_registry)
    result = await rehydrated_pipeline.steps[0].arun("foo")
    assert result == "CACHED: foo"

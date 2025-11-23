import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import HumanInTheLoopStep, Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import Success, Chunk, Paused


class _EchoAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        return f"ok:{payload}"


@pytest.mark.asyncio
async def test_runner_run_outcomes_non_streaming_yields_success():
    step = Step(name="echo", agent=_EchoAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)
    outcomes = []
    async for item in f.run_outcomes_async("hi"):
        outcomes.append(item)
    assert isinstance(outcomes[-1], Success)
    assert outcomes[-1].step_result.success is True
    assert outcomes[-1].step_result.name == "echo"


class _StreamAgent:
    async def stream(self, payload, context=None, resources=None, **kwargs):
        yield "a"
        yield "b"
        return


@pytest.mark.asyncio
async def test_runner_run_outcomes_streaming_yields_chunks_then_success():
    step = Step(name="stream", agent=_StreamAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)
    chunks = []
    final = None
    async for item in f.run_outcomes_async("hi"):
        if isinstance(item, Chunk):
            chunks.append(item.data)
        if isinstance(item, Success):
            final = item
    assert chunks == ["a", "b"]
    assert isinstance(final, Success)
    assert final.step_result.success is True


class _HitlAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        from flujo.exceptions import PausedException

        raise PausedException("wait")


@pytest.mark.asyncio
async def test_runner_run_outcomes_paused_yields_paused():
    step = Step(name="hitl", agent=_HitlAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)
    outs = []
    async for item in f.run_outcomes_async("hi"):
        outs.append(item)
        break
    assert outs and isinstance(outs[0], Paused)


@pytest.mark.asyncio
async def test_runner_run_outcomes_nested_pause_bubbles():
    """Nested control-flow should still surface Paused outcomes."""
    hitl_step = HumanInTheLoopStep(name="hitl_nested", message_for_user="pause here")
    conditional = ConditionalStep(
        name="route_to_hitl",
        condition_callable=lambda _out, _ctx: "hitl",
        branches={"hitl": Pipeline.from_step(hitl_step)},
    )
    loop = LoopStep(
        name="outer_loop",
        loop_body_pipeline=Pipeline.from_step(conditional),
        exit_condition_callable=lambda _out, _ctx: True,
        max_loops=1,
    )
    f = Flujo(Pipeline.from_step(loop))

    outcomes = []
    async for item in f.run_outcomes_async({"payload": "start"}):
        outcomes.append(item)
        break

    assert outcomes and isinstance(outcomes[0], Paused)

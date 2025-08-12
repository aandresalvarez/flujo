import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
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

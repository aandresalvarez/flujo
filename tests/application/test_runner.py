import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import HumanInTheLoopStep, Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import Success, Chunk, Paused, PipelineResult, BackgroundLaunched


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


@pytest.mark.asyncio
async def test_runner_run_async_return_result():
    step = Step(name="echo", agent=_EchoAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)

    result = await f.run_result_async("hi")

    assert isinstance(result, PipelineResult)
    assert result.step_history[-1].success is True
    assert result.step_history[-1].name == "echo"


@pytest.mark.asyncio
async def test_runner_run_stream_and_run_outcomes_alias():
    step = Step(name="stream", agent=_StreamAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)

    stream_items = []
    async for outcome in f.run_stream("hi"):
        stream_items.append(outcome)

    alias_items = []
    async for outcome in f.run_outcomes("hi"):
        alias_items.append(outcome)

    assert any(isinstance(item, Success) for item in stream_items)
    assert any(isinstance(item, Success) for item in alias_items)
    # run_stream should surface streaming chunks; alias may be used for readability
    assert any(isinstance(item, Chunk) for item in stream_items)


@pytest.mark.asyncio
async def test_runner_run_with_events_background_emits_launch():
    async def bg_task(data: str) -> str:
        return f"bg:{data}"

    async def fg_task(data: str) -> str:
        return f"fg:{data}"

    bg_step = Step.from_callable(bg_task, name="bg", execution_mode="background")
    fg_step = Step.from_callable(fg_task, name="fg")
    pipe = bg_step >> fg_step
    f = Flujo(pipe)

    events = []
    async for ev in f.run_with_events("hi"):
        events.append(ev)

    assert any(isinstance(ev, BackgroundLaunched) for ev in events)
    final = next(ev for ev in events if isinstance(ev, PipelineResult))
    assert final.success
    assert final.step_history[0].name == "bg"
    assert "background" in (final.step_history[0].feedback or "").lower()
    assert final.step_history[-1].output == "fg:hi"


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


def test_runner_can_disable_persistence_with_flag():
    step = Step(name="echo", agent=_EchoAgent())
    pipe = Pipeline.from_step(step)
    from tests.conftest import NoOpStateBackend

    with pytest.warns(UserWarning):
        f = Flujo(pipe, state_backend=NoOpStateBackend(), persist_state=False)

    assert f.persist_state is False
    assert f.state_backend is None

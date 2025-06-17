import asyncio
from typing import AsyncIterator, Any

import pytest

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineResult
from flujo.domain.resources import AppResources
from flujo.testing.utils import StubAgent, FailingStreamAgent
from pydantic import BaseModel


class MockStreamingAgent:
    async def stream(self, data: str, **kwargs: Any) -> AsyncIterator[str]:
        for ch in data:
            await asyncio.sleep(0)  # mimic async work
            yield ch


@pytest.mark.asyncio
async def test_basic_streaming() -> None:
    pipeline = Step.solution(MockStreamingAgent())
    runner = Flujo(pipeline)

    chunks = []
    final: PipelineResult | None = None
    async for item in runner.stream_async("hi"):
        if isinstance(item, PipelineResult):
            final = item
        else:
            chunks.append(item)
    assert final is not None
    assert "".join(chunks) == "hi"
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_non_streaming_pipeline() -> None:
    pipeline = Step.solution(StubAgent(["ok"]))
    runner = Flujo(pipeline)

    items = [c async for c in runner.stream_async("x")]
    assert len(items) == 1
    assert isinstance(items[0], PipelineResult)


class Ctx(BaseModel):
    count: int = 0


class MyResources(AppResources):
    increment: int = 1


class CtxStreamAgent:
    async def stream(
        self, data: int, *, pipeline_context: Ctx, resources: MyResources
    ) -> AsyncIterator[str]:
        pipeline_context.count += 1
        yield str(data + resources.increment)


@pytest.mark.asyncio
async def test_context_and_resources_in_stream() -> None:
    pipeline = Step("s", CtxStreamAgent())
    resources = MyResources(increment=2)
    runner = Flujo(
        pipeline,
        context_model=Ctx,
        initial_context_data={"count": 0},
        resources=resources,
    )

    async for item in runner.stream_async(3):
        final = item  # only final result is yielded
    assert isinstance(final, PipelineResult)
    assert final.final_pipeline_context.count == 1
    assert final.step_history[-1].output == "5"


@pytest.mark.asyncio
async def test_pipeline_handles_streaming_agent_failure_gracefully() -> None:
    agent = FailingStreamAgent(["H", "e", "l"], RuntimeError("Stream connection lost"))
    pipeline = Step.solution(agent)
    runner = Flujo(pipeline)

    collected: list[str] = []
    final: PipelineResult | None = None
    async for item in runner.stream_async("Hello"):
        if isinstance(item, PipelineResult):
            final = item
        else:
            collected.append(item)

    assert collected == ["H", "e", "l"]
    assert final is not None
    step_result = final.step_history[-1]
    assert not step_result.success
    assert "Stream connection lost" in (step_result.feedback or "")

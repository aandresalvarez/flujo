import pytest

from pydantic_ai_orchestrator.pipeline import (
    PipelineRunner,
    Step,
    StepConfig,
    PipelineResult,
)


@pytest.mark.asyncio
async def test_pipeline_runner_returns_result():
    async def step1(x: int) -> int:
        return x + 1

    async def step2(x: int) -> int:
        return x * 2

    runner = PipelineRunner(
        [Step(name="one", func=step1), Step(name="two", func=step2)]
    )

    result = await runner.run(1)
    assert isinstance(result, PipelineResult)
    assert result.output == 4
    assert len(result.steps) == 2


@pytest.mark.asyncio
async def test_pipeline_runner_retries():
    calls = 0

    async def flaky(x: int) -> int:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ValueError("fail")
        return x + 1

    runner = PipelineRunner([
        Step(name="flaky", func=flaky, config=StepConfig(max_retries=1))
    ])

    result = await runner.run(1)
    assert result.output == 2
    assert result.steps[0].attempts == 2

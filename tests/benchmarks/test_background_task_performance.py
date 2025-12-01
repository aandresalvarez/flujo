import asyncio
import time

import pytest

from flujo import Flujo, Pipeline, Step
from flujo.domain.dsl.step import StepConfig
from flujo.domain.models import PipelineContext

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


class PerfContext(PipelineContext):
    value: int = 0


async def short_bg(data: str, context: PerfContext | None = None) -> str:
    await asyncio.sleep(0.05)
    return f"bg_{data}"


@pytest.mark.asyncio
async def test_background_task_overhead_below_half_second() -> None:
    """Ensure background execution/persistence overhead stays low."""
    bg_step = Step.from_callable(
        short_bg,
        name="bg_perf",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(bg_step)

    start = time.perf_counter()
    async with Flujo(pipeline, context_model=PerfContext) as runner:
        await runner.run_async("payload")
        await asyncio.sleep(0.1)
    duration = time.perf_counter() - start
    # Generous bound to avoid flakiness while still catching regressions.
    assert duration < 0.5

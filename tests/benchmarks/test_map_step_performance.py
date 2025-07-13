import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.testing.utils import gather_result


class SleepAgent:
    async def run(self, data: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.01)
        return data


class Ctx(BaseModel):
    data_list: list[int]


items_to_process = list(range(100))


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="MapStep currently executes sequentially, so this benchmark is expected to fail until parallel mapping is implemented"
)
async def test_map_step_performance_vs_asyncio_gather() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "sleep", "agent": SleepAgent()}))
    map_step = Step.map_over("mapper", body, iterable_input="data_list")
    runner = Flujo(map_step, context_model=Ctx)

    start = time.monotonic()
    result = await gather_result(
        runner,
        None,
        initial_context_data={"data_list": items_to_process},
    )
    map_step_duration = time.monotonic() - start

    assert result.step_history[-1].output == items_to_process
    assert map_step_duration < 0.5

    agent = SleepAgent()
    tasks = [agent.run(i) for i in items_to_process]
    start = time.monotonic()
    await asyncio.gather(*tasks)
    baseline_duration = time.monotonic() - start

    assert baseline_duration < 0.5

    overhead_ms = (map_step_duration - baseline_duration) * 1000
    print("\nMapStep duration:", map_step_duration)
    print("Baseline duration:", baseline_duration)
    print("Overhead (ms):", overhead_ms)

    assert map_step_duration <= baseline_duration * 3.0

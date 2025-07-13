import os
import gc
import psutil
import pytest
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.testing.utils import gather_result


class LargeModel(BaseModel):
    idx: int
    payload: str


class LargeModelAgent:
    def __init__(self, size: int = 200_000) -> None:
        self.size = size

    async def run(self, idx: int) -> int:
        _ = LargeModel(idx=idx, payload="x" * self.size)
        return idx + 1


@pytest.mark.asyncio
async def test_loop_step_memory_stability() -> None:
    """Ensure LoopStep does not leak memory across many iterations."""

    iterations = 1000
    body_step = Step.model_validate({"name": "make_large", "agent": LargeModelAgent()})
    body_pipeline = Pipeline.from_step(body_step)
    loop = Step.loop_until(
        name="loop_mem_test",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda *_: False,
        max_loops=iterations,
    )
    runner = Flujo(loop)

    process = psutil.Process(os.getpid())
    gc.collect()
    initial_memory = process.memory_info().rss

    result = await gather_result(runner, 0)

    gc.collect()
    final_memory = process.memory_info().rss
    delta = final_memory - initial_memory

    print(f"\nInitial memory: {initial_memory / 1024**2:.2f} MB")
    print(f"Final memory: {final_memory / 1024**2:.2f} MB")
    print(f"Delta memory: {delta / 1024**2:.2f} MB")

    assert result.step_history[-1].attempts == iterations
    assert delta < 50 * 1024 * 1024

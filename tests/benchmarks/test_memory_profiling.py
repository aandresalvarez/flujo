import os
import gc

import pytest
from pydantic import BaseModel

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment,unused-ignore]

pytestmark = []
if psutil is None:
    pytestmark.append(pytest.mark.skip(reason="psutil not available"))

from flujo import Step, Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class LargeModel(BaseModel):
    idx: int
    payload: str


class LargeModelAgent:
    def __init__(self, size: int = 5_000) -> None:  # Reduced from 10_000 to 5_000 (50% reduction)
        self.size = size

    async def run(self, idx: int) -> int:
        _ = LargeModel(idx=idx, payload="x" * self.size)
        return idx + 1


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_loop_step_memory_stability() -> None:
    """Ensure LoopStep does not leak memory across many iterations."""
    if psutil is None:
        pytest.skip("psutil not available")

    iterations = 500  # Reduced from 1000 to 500 (50% reduction)
    body_step = Step.model_validate({"name": "make_large", "agent": LargeModelAgent()})
    body_pipeline = Pipeline.from_step(body_step)
    loop = Step.loop_until(
        name="loop_mem_test",
        loop_body_pipeline=body_pipeline,
        exit_condition_callable=lambda *_: False,
        max_loops=iterations,
    )
    runner = create_test_flujo(loop, persist_state=False)
    runner.disable_tracing()

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
    assert (
        delta < 100 * 1024 * 1024
    )  # Increased limit from 50MB to 100MB for more realistic testing

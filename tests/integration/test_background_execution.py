import asyncio
import time
import pytest
from typing import Any, Optional

from flujo import Step, Pipeline, Flujo
from flujo.domain.models import PipelineContext

# Shared state for verification
BACKGROUND_OP_COMPLETED = False
BACKGROUND_OP_START_TIME = 0.0
BACKGROUND_SLEEP_SECONDS = 1.5


class TestContext(PipelineContext):
    value: int = 0
    __test__ = False  # prevent pytest from collecting as a test class


async def slow_background_task(data: str, context: Optional[TestContext] = None) -> str:
    global BACKGROUND_OP_COMPLETED, BACKGROUND_OP_START_TIME
    BACKGROUND_OP_START_TIME = time.time()
    await asyncio.sleep(BACKGROUND_SLEEP_SECONDS)  # Sleep longer than pipeline execution
    BACKGROUND_OP_COMPLETED = True
    if context:
        # Modify context to test isolation - shouldn't affect main pipeline if isolated
        context.value = 999
    return f"processed_{data}"


async def fast_foreground_task(data: Optional[str]) -> str:
    return f"fast_{data}"


@pytest.mark.asyncio
async def test_background_execution_fire_and_forget():
    """Verify that background steps do not block pipeline execution."""
    global BACKGROUND_OP_COMPLETED
    BACKGROUND_OP_COMPLETED = False

    # Create steps
    bg_step = Step.from_callable(slow_background_task, name="bg_step", execution_mode="background")
    fg_step = Step.from_callable(fast_foreground_task, name="fg_step")

    pipeline = bg_step >> fg_step

    start_time = time.time()

    async with Flujo(pipeline, context_model=TestContext) as runner:
        result = await runner.run_async("input")

        end_time = time.time()
        duration = end_time - start_time

        # 1. Verify pipeline finished well before the background sleep completes.
        # It should never wait as long as the background task itself.
        assert duration < BACKGROUND_SLEEP_SECONDS, f"Pipeline took too long: {duration}s"

        # 2. Verify background task hasn't finished yet (at the moment of return)
        # Note: In a very slow environment this might be flaky, but 0.5s vs <0.2s margin is large
        assert not BACKGROUND_OP_COMPLETED, "Background task finished too early"

        # 3. Verify pipeline result
        # The background step returns a BackgroundLaunched outcome which unwrap to success=True
        # The foreground step should have run immediately
        assert result.success
        assert len(result.step_history) == 2
        assert result.step_history[0].name == "bg_step"
        assert result.step_history[0].success is True
        assert "Launched in background" in (result.step_history[0].feedback or "")

        # Verify the foreground step received the input data (passed through from bg step)
        assert result.step_history[1].name == "fg_step"
        assert result.step_history[1].output == "fast_input"  # fg_step ran with "input" data

        # 4. Wait for background tasks to complete (aclose does this automatically)
        # The 'async with' block exit triggers aclose()

    # After aclose(), background task should be done
    assert BACKGROUND_OP_COMPLETED, "Background task did not complete after runner close"


@pytest.mark.asyncio
async def test_background_execution_context_isolation():
    """Verify that background steps have isolated context."""
    global BACKGROUND_OP_COMPLETED
    BACKGROUND_OP_COMPLETED = False

    bg_step = Step.from_callable(slow_background_task, name="bg_step", execution_mode="background")

    pipeline = Pipeline.from_step(bg_step)

    async with Flujo(pipeline, context_model=TestContext) as runner:
        # Run with initial context value 10
        result = await runner.run_async("input", initial_context_data={"value": 10})

        # Verify immediate result context is unchanged
        assert result.final_pipeline_context.value == 10

    # After completion, verify background task ran
    assert BACKGROUND_OP_COMPLETED
    # We can't easily check the isolated context instance since it's gone,
    # but we verified it didn't crash and main context wasn't mutated during run.


async def failing_background_task(_data: Any) -> None:
    await asyncio.sleep(0.1)
    raise ValueError("Boom!")


@pytest.mark.asyncio
async def test_background_execution_failure_logging():
    """Verify that background failures don't crash the pipeline."""
    bg_step = Step.from_callable(
        failing_background_task, name="fail_step", execution_mode="background"
    )

    async with Flujo(bg_step) as runner:
        result = await runner.run_async("input")
        assert result.success
        assert "Launched in background" in (result.step_history[0].feedback or "")

        # Wait for failure to happen (logs should appear, but no crash)
        await asyncio.sleep(0.2)

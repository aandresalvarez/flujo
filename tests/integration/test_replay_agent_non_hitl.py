import os
import tempfile
import pytest
import asyncio
import logging
import signal
from contextlib import asynccontextmanager

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.state.backends.sqlite import SQLiteBackend

# Uses state backend and trace replay; mark as slow for fast subset exclusion
pytestmark = [pytest.mark.slow]

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout error for better error messages."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


async def _upper(x: object) -> str:
    return str(x).upper()


async def _suffix(x: object) -> str:
    return f"{x}!"


@asynccontextmanager
async def create_test_runner():
    """Create a test runner with proper cleanup to prevent hanging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "state.db")
        logger.info(f"Creating SQLite backend at {db_path}")
        backend = SQLiteBackend(db_path)
        try:
            s1 = Step.from_callable(_upper, name="Upper")
            s2 = Step.from_callable(_suffix, name="Suffix")
            p = Pipeline(steps=[s1, s2])
            r = Flujo(pipeline=p, state_backend=backend)
            logger.info("Test runner created successfully")
            yield r, backend
        finally:
            # Ensure backend is properly closed
            try:
                logger.info("Closing SQLite backend")
                await backend.close()
            except Exception as e:
                logger.warning(f"Error closing backend: {e}")


async def execute_pipeline_with_timeout(runner, input_data, timeout_seconds: float):
    """Execute the pipeline to completion with timeout protection."""

    async def _run():
        final = None
        async for item in runner.run_async(input_data):
            final = item  # take last yielded item as final
            logger.info(f"Pipeline step completed: {item}")
        return final

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Pipeline execution timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Pipeline execution timed out after {timeout_seconds} seconds")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 60 second timeout for entire test
async def test_replay_agent_replays_non_hitl_pipeline_from_state():
    """Test that replay agent can replay a non-HITL pipeline from state."""
    logger.info("Starting replay agent test")

    # Set up signal-based timeout as additional protection
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(55)  # 55 second alarm (5 seconds before pytest timeout)

    try:
        async with create_test_runner() as (runner, backend):
            logger.info("Running original pipeline execution")
            # Run original to persist step outputs/spans
            try:
                # Use timeout protection for pipeline execution
                final = await execute_pipeline_with_timeout(runner, "go", 20.0)
            except TimeoutError:
                pytest.fail("Pipeline execution timed out - possible hanging issue")
            except Exception as e:
                pytest.fail(f"Original pipeline execution failed: {e}")

            assert final is not None
            run_id = getattr(final.final_pipeline_context, "run_id", None)
            assert run_id
            logger.info(f"Pipeline completed with run_id: {run_id}")

            # Replay deterministically from stored records with timeout protection
            logger.info("Starting replay operation")
            try:
                # Add timeout to prevent hanging
                replayed = await asyncio.wait_for(
                    runner.replay_from_trace(run_id),
                    30.0,  # 30 second timeout for replay operation
                )
                assert replayed is not None
                # The step history should match the original shape and final output
                assert replayed.step_history[-1].output == "GO!"
                logger.info("Replay operation completed successfully")
            except asyncio.TimeoutError:
                logger.error("Replay operation timed out after 30 seconds")
                pytest.fail("Replay operation timed out after 30 seconds - possible hanging issue")
            except Exception as e:
                logger.error(f"Replay operation failed with error: {e}")
                pytest.fail(f"Replay operation failed with error: {e}")

        logger.info("Test completed successfully")

    finally:
        # Restore original signal handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

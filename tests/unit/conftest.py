import logging
import asyncio
import pytest
from io import StringIO
from contextlib import contextmanager


@contextmanager
def capture_logs(logger_name: str = "flujo", level: int = logging.DEBUG):
    """Context manager to capture log output for testing."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(level)

    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)  # Restore the original logger level


@pytest.fixture(autouse=True)
def cleanup_threads():
    """Cleanup fixture to handle hanging threads."""
    yield
    # Force cleanup of any remaining threads
    import gc

    gc.collect()

    # Cancel any remaining asyncio tasks and close event loops
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, we can't cancel tasks from here
        pass
    except RuntimeError:
        # No running loop, try to get the current loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    if not task.done():
                        task.cancel()
                # Run the loop to process cancellations
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                # Close the loop
                if not loop.is_closed():
                    loop.close()
        except RuntimeError:
            # No event loop
            pass

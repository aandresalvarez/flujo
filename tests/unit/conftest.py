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
    import threading
    import time

    gc.collect()

    # More aggressive asyncio cleanup
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop, can't close it from here
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
    except Exception:
        # Ignore any errors in asyncio cleanup
        pass

    # Force cleanup of any non-daemon threads that might be hanging
    # This is a last resort - only for threads that are clearly hanging
    active_threads = [t for t in threading.enumerate() if t != threading.current_thread()]
    hanging_threads = [t for t in active_threads if not t.daemon and t.is_alive()]

    if hanging_threads:
        # Give threads a moment to finish naturally
        time.sleep(0.1)
        # Force terminate any still hanging
        for thread in hanging_threads:
            if thread.is_alive():
                # This is a bit aggressive but should help with hanging tests
                try:
                    thread._stop()
                except Exception:
                    pass

    # Additional cleanup: try to close any remaining event loops
    try:
        # Force close any remaining event loops
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
    except Exception:
        pass

    # Clean up Prometheus server resources
    try:
        from flujo.telemetry.prometheus import cleanup_prometheus_server

        cleanup_prometheus_server()
    except Exception:
        pass

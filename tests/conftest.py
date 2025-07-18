"""Global test configuration and cleanup."""

import pytest
import asyncio
import threading
import gc
import signal
import sys


def pytest_configure(config):
    """Configure pytest with aggressive cleanup."""

    # Set up signal handlers for cleanup
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, cleaning up...")
        cleanup_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def cleanup_all():
    """Aggressive cleanup of all resources."""
    print("Performing aggressive cleanup...")

    # Force garbage collection
    gc.collect()

    # Cancel all asyncio tasks
    try:
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
    except Exception:
        pass

    # Force cleanup of any non-daemon threads
    active_threads = [t for t in threading.enumerate() if t != threading.current_thread()]
    hanging_threads = [t for t in active_threads if not t.daemon and t.is_alive()]

    if hanging_threads:
        print(f"Found {len(hanging_threads)} hanging threads, attempting cleanup...")
        for thread in hanging_threads:
            if thread.is_alive():
                try:
                    thread._stop()
                except Exception:
                    pass

    # Additional cleanup: close any remaining event loops
    try:
        # Try to close any remaining event loops
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

        # Force close any remaining event loops
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except RuntimeError:
            pass
    except Exception:
        pass

    # Clean up Prometheus server resources
    try:
        from flujo.telemetry.prometheus import cleanup_prometheus_server

        cleanup_prometheus_server()
    except Exception:
        pass

    # Force final garbage collection
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Session-level cleanup."""
    yield
    cleanup_all()


@pytest.fixture(autouse=True)
def test_cleanup():
    """Test-level cleanup."""
    yield
    cleanup_all()

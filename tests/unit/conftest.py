import logging
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


class MockStatelessAgent:
    def __init__(self):
        # Track calls without using AsyncMock to avoid unraisable exceptions
        self.call_count = 0
        self._call_args = None
        self._call_kwargs = None
        # Create a mock-like object that provides the expected interface
        self.run_mock = self._create_mock_interface()

    def _create_mock_interface(self):
        """Create a mock-like interface that doesn't use AsyncMock."""

        class MockInterface:
            def __init__(self, parent):
                self.parent = parent

            def assert_called_once(self):
                """Assert that the run method was called exactly once."""
                assert self.parent.call_count == 1, f"Expected 1 call, got {self.parent.call_count}"

            @property
            def call_args(self):
                """Return the arguments of the last call in the expected format (args, kwargs)."""
                if self.parent.call_count == 0:
                    raise AssertionError("call_args accessed but no calls have been made yet.")
                return (self.parent._call_args, self.parent._call_kwargs)

        return MockInterface(self)

    async def run(self, data: str) -> str:
        """Run method that does NOT accept context parameter - simulates stateless agent"""
        self.call_count += 1
        self._call_args = (data,)
        self._call_kwargs = {}
        return f"Mock response to: {data}"

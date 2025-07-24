import logging
from io import StringIO
from contextlib import contextmanager
from unittest.mock import AsyncMock


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
        self.run_mock = AsyncMock(return_value="Hello! I'm here to help.")

    async def run(self, data: str) -> str:
        """Run method that does NOT accept context parameter - simulates stateless agent"""
        await self.run_mock(data)
        return f"Mock response to: {data}"

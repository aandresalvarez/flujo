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

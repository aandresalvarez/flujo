import logging
from typing import Any

from flujo.infra.telemetry import _safe_log


def test_safe_log_handles_none_logger_gracefully() -> None:
    # Should not raise when logger is None
    _safe_log(None, logging.INFO, "message")  # type: ignore[arg-type]


class Dummy:
    pass


def test_safe_log_handles_object_without_log_method() -> None:
    dummy: Any = Dummy()
    # Should simply return without raising
    _safe_log(dummy, logging.INFO, "message")  # type: ignore[arg-type]

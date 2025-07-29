import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, List
from typing import Any as _TypeAny  # local alias to avoid name clash

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor
    from .settings import Settings as TelemetrySettings

_initialized = False

_fallback_logger = logging.getLogger("flujo")
_fallback_logger.setLevel(logging.INFO)

if not _fallback_logger.handlers:
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno <= logging.WARNING)
    info_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    info_handler.setFormatter(info_formatter)
    _fallback_logger.addHandler(info_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    error_handler.setFormatter(error_formatter)
    _fallback_logger.addHandler(error_handler)


def _is_cleanup_io_error(error: Exception) -> bool:
    """Check if an error is related to cleanup I/O operations."""
    error_str = str(error).lower()
    cleanup_phrases = ["i/o operation on closed file", "closed", "bad file descriptor"]
    return any(phrase in error_str for phrase in cleanup_phrases)


def _safe_log(logger: logging.Logger, level: int, message: str, *args: Any, **kwargs: Any) -> None:
    """Safely log a message, handling I/O errors gracefully during cleanup.

    This function ensures that:
    1. All valid handlers receive the log message (no premature exit)
    2. Binary streams are handled properly
    3. No side effects from stream testing
    4. Graceful degradation when some handlers fail
    """
    try:
        # Check if logger is still valid before attempting to log
        if logger is None or not hasattr(logger, "log"):
            return

        # Attempt to log the message - let the actual logging attempt happen
        # and catch any I/O errors that result. The logging system will handle
        # individual handler failures internally.
        logger.log(level, message, *args, **kwargs)

    except (ValueError, OSError, RuntimeError) as e:
        if _is_cleanup_io_error(e):
            # During cleanup, some handlers may be closed
            # This is expected behavior, so we silently ignore
            pass
        else:
            # Re-raise unexpected errors
            raise


class _SafeLogfireWrapper:
    """Wrapper for real logfire library that handles I/O errors gracefully."""

    def __init__(self, real_logfire: Any):
        self._real_logfire = real_logfire

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.info(message, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise

    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.warn(message, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.warning(message, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if any(
                phrase in str(e).lower()
                for phrase in [
                    "i/o operation on closed file",
                    "closed",
                    "bad file descriptor",
                ]
            ):
                pass  # Silently ignore during cleanup
            else:
                raise

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.error(message, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.debug(message, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise

    def configure(self, *args: Any, **kwargs: Any) -> None:
        try:
            self._real_logfire.configure(*args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise

    def instrument(self, name: str, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        try:
            result = self._real_logfire.instrument(name, *args, **kwargs)
            # Ensure the result is a callable
            if callable(result):
                return result  # type: ignore
            else:
                # Fallback to no-op decorator if result is not callable
                def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
                    return func

                return decorator
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                # Return a no-op decorator during cleanup
                def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
                    return func

                return decorator
            else:
                raise

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._real_logfire.span(name, *args, **kwargs)
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                # Return a mock span during cleanup
                return _MockLogfireSpan()
            else:
                raise

    def enable_stdout_viewer(self) -> None:
        try:
            self._real_logfire.enable_stdout_viewer()
        except (ValueError, OSError, RuntimeError) as e:
            if _is_cleanup_io_error(e):
                pass  # Silently ignore during cleanup
            else:
                raise


class _MockLogfireSpan:
    def __enter__(self) -> "_MockLogfireSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class _MockLogfire:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger, logging.INFO, message, *args, **kwargs)

    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger, logging.WARNING, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger, logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger, logging.ERROR, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        _safe_log(self._logger, logging.DEBUG, message, *args, **kwargs)

    def configure(self, *args: Any, **kwargs: Any) -> None:
        _safe_log(
            self._logger,
            logging.INFO,
            "Logfire.configure called, but Logfire is mocked. Using standard Python logging.",
        )

    def instrument(self, name: str, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            return func

        return decorator

    def span(self, name: str, *args: Any, **kwargs: Any) -> _MockLogfireSpan:
        return _MockLogfireSpan()

    def enable_stdout_viewer(self) -> None:
        _safe_log(
            self._logger,
            logging.INFO,
            "Logfire.enable_stdout_viewer called, but Logfire is mocked.",
        )


# We initially set `logfire` to a mocked implementation. Once
# `init_telemetry()` runs, we may replace it with the real `logfire` module.
# Annotate as `_TypeAny` so that MyPy accepts this reassignment.
logfire: _TypeAny = _MockLogfire(_fallback_logger)


def init_telemetry(settings_obj: Optional["TelemetrySettings"] = None) -> None:
    """Configure global logging and tracing for the process.

    Call once at application startup. If ``settings_obj`` is not provided the
    default :class:`~flujo.infra.settings.Settings` object is used. When telemetry
    is enabled the real ``logfire`` library is configured, otherwise a fallback
    logger that proxies to ``logging`` is provided.
    """

    global _initialized, logfire
    if _initialized:
        return

    from .settings import settings as default_settings_obj

    settings_to_use = settings_obj if settings_obj is not None else default_settings_obj

    if settings_to_use.telemetry_export_enabled:
        try:
            import logfire as _actual_logfire

            # Wrap the real logfire with our safe wrapper
            logfire = _SafeLogfireWrapper(_actual_logfire)

            additional_processors: List["SpanProcessor"] = []
            if settings_to_use.otlp_export_enabled:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter_args: Dict[str, Any] = {}
                if settings_to_use.otlp_endpoint:
                    exporter_args["endpoint"] = settings_to_use.otlp_endpoint

                exporter = OTLPSpanExporter(**exporter_args)
                additional_processors.append(BatchSpanProcessor(exporter))

            logfire.configure(
                service_name="flujo",
                send_to_logfire=True,
                additional_span_processors=additional_processors,
                console=False,
                api_key=(
                    settings_to_use.logfire_api_key.get_secret_value()
                    if settings_to_use.logfire_api_key
                    else None
                ),
            )
            _safe_log(
                _fallback_logger,
                logging.INFO,
                "Logfire initialized successfully (actual Logfire).",
            )
            _initialized = True
            return
        except ImportError:
            _safe_log(
                _fallback_logger,
                logging.WARNING,
                "Logfire library not installed. Falling back to standard Python logging.",
            )
        except Exception as e:
            _safe_log(
                _fallback_logger,
                logging.ERROR,
                f"Failed to configure Logfire: {e}. Falling back to standard Python logging.",
            )

    _safe_log(
        _fallback_logger,
        logging.INFO,
        "Logfire telemetry is disabled or failed to initialize. Using standard Python logging.",
    )
    logfire = _MockLogfire(_fallback_logger)
    _initialized = True


# Auto-initialize telemetry with default settings when module is imported
# This ensures telemetry is always available, even in tests
# Only auto-initialize if not already initialized and if we're not in a test environment
if not _initialized:
    try:
        # Check if we're in a test environment by looking for pytest markers
        import sys

        if "pytest" not in sys.modules and "test" not in sys.modules:
            init_telemetry()
    except Exception:
        # If auto-initialization fails, we still have the mock logfire available
        pass

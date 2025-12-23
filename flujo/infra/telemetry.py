import logging
import sys

import atexit
from typing import TYPE_CHECKING, Any, Callable, Optional, List
from typing import Any as _TypeAny  # local alias to avoid name clash

set_default_telemetry_sink_fn: Optional[Callable[[Any], None]]
try:  # pragma: no cover - import guard
    from flujo.domain.interfaces import set_default_telemetry_sink as _set_default_telemetry_sink_fn

    set_default_telemetry_sink_fn = _set_default_telemetry_sink_fn
except Exception:  # pragma: no cover - defensive fallback
    set_default_telemetry_sink_fn = None

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor
    from .settings import Settings as TelemetrySettings

_initialized = False

_fallback_logger = logging.getLogger("flujo")
# Default to INFO; init_telemetry() may tighten this for CI/test-mode.
_fallback_logger.setLevel(logging.INFO)
# Always propagate so external handlers (caplog, app) can capture logs
_fallback_logger.propagate = True

# Do not attach handlers here; let messages propagate to root handlers.
# This ensures test frameworks (caplog) and app-level configuration capture logs consistently.


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
        if sys.is_finalizing():
            return

        def _handler_closed(handler: logging.Handler) -> bool:
            stream = getattr(handler, "stream", None)
            if stream is None:
                return False
            try:
                return bool(getattr(stream, "closed", False))
            except Exception:
                return False

        def _prune_closed_handlers(target: logging.Logger) -> None:
            try:
                handlers = list(getattr(target, "handlers", []))
                if not handlers:
                    return
                keep: list[logging.Handler] = []
                for handler in handlers:
                    if _handler_closed(handler):
                        try:
                            handler.close()
                        except Exception:
                            pass
                        continue
                    keep.append(handler)
                if len(keep) != len(handlers):
                    target.handlers = keep
            except Exception:
                pass

        _prune_closed_handlers(logger)
        if getattr(logger, "propagate", False):
            _prune_closed_handlers(logging.getLogger())

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
            decorator: Callable[[Any], Any] = self._real_logfire.instrument(name, *args, **kwargs)
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

    def instrument(self, name: str, *args: Any, **kwargs: Any) -> Any:
        def decorator(func: Callable[[Any], Any]) -> Any:
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


class _LogfireTelemetrySink:
    """Adapter exposing module-level logfire through the domain TelemetrySink protocol."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            logfire.info(message, *args, **kwargs)
        except Exception:
            pass

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            logfire.warning(message, *args, **kwargs)
        except Exception:
            pass

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            logfire.error(message, *args, **kwargs)
        except Exception:
            pass

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            logfire.debug(message, *args, **kwargs)
        except Exception:
            pass

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return logfire.span(name, *args, **kwargs)
        except Exception:
            return _MockLogfireSpan()


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
    # Reduce log verbosity in CI/tests to improve performance determinism.
    # This must happen after settings are available; avoid reading env vars at module import time.
    try:
        import os

        in_ci = os.getenv("CI") == "true"
    except Exception:
        in_ci = False
    _fallback_logger.setLevel(
        logging.WARNING
        if (in_ci or bool(getattr(settings_to_use, "test_mode", False)))
        else logging.INFO
    )

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

                exporter_args: dict[str, Any] = {}
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
            # Ensure telemetry flushes before interpreter shutdown to avoid lost tail logs.
            try:
                flush_fn = None
                if hasattr(_actual_logfire, "force_flush"):
                    flush_fn = _actual_logfire.force_flush
                elif hasattr(_actual_logfire, "flush"):
                    flush_fn = _actual_logfire.flush
                if flush_fn:
                    atexit.register(flush_fn)
            except Exception:
                pass
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

    # Reduce noise in CLI: emit this at DEBUG level instead of INFO
    _safe_log(
        _fallback_logger,
        logging.DEBUG,
        "Using standard Python logging.",
    )
    logfire = _MockLogfire(_fallback_logger)
    _initialized = True


# Auto-initialize telemetry with default settings when module is imported
# This ensures telemetry is always available.
# We no longer check for test environments implicitly; tests must explicitly mock or configure telemetry if needed.
if not _initialized:
    try:
        init_telemetry()
    except Exception:
        # If auto-initialization fails, we still have the mock logfire available
        pass

# Expose telemetry sink to domain consumers (no-op if domain interfaces unavailable)
if set_default_telemetry_sink_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_telemetry_sink_fn(_LogfireTelemetrySink())
    except Exception:
        pass

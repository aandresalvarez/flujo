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
        self._logger.info(message, *args, **kwargs)

    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message, *args, **kwargs)

    def configure(self, *args: Any, **kwargs: Any) -> None:
        self._logger.info(
            "Logfire.configure called, but Logfire is mocked. Using standard Python logging."
        )

    def instrument(self, name: str, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            return func

        return decorator

    def span(self, name: str, *args: Any, **kwargs: Any) -> _MockLogfireSpan:
        return _MockLogfireSpan()

    def enable_stdout_viewer(self) -> None:
        self._logger.info("Logfire.enable_stdout_viewer called, but Logfire is mocked.")


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

            logfire = _actual_logfire

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
            _fallback_logger.info("Logfire initialized successfully (actual Logfire).")
            _initialized = True
            return
        except ImportError:
            _fallback_logger.warning(
                "Logfire library not installed. Falling back to standard Python logging."
            )
        except Exception as e:
            _fallback_logger.error(
                f"Failed to configure Logfire: {e}. Falling back to standard Python logging."
            )

    _fallback_logger.info(
        "Logfire telemetry is disabled or failed to initialize. Using standard Python logging."
    )
    logfire = _MockLogfire(_fallback_logger)
    _initialized = True

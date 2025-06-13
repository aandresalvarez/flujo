"""Telemetry and logging for pydantic-ai-orchestrator.""" 

import logfire
from .settings import settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor

_initialized = False

def init_telemetry() -> None:
    """
    Initialize Logfire telemetry for the orchestrator.
    This function is idempotent and safe to call multiple times.
    It configures logging based on application settings.
    """
    global _initialized
    if _initialized:
        return

    additional_processors: list['SpanProcessor'] = []
    if settings.otlp_export_enabled:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter_args: dict[str, object] = {}
        if settings.otlp_endpoint:
            exporter_args["endpoint"] = settings.otlp_endpoint
        
        exporter = OTLPSpanExporter(**exporter_args)  # type: ignore[arg-type]
        additional_processors.append(BatchSpanProcessor(exporter))

    logfire.configure(
        service_name="pydantic_ai_orchestrator",
        send_to_logfire=settings.telemetry_export_enabled,
        additional_span_processors=additional_processors,
    )
    _initialized = True 

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Literal, Optional, cast

from ..domain.events import (
    HookPayload,
    OnStepFailurePayload,
    PostRunPayload,
    PostStepPayload,
    PreRunPayload,
    PreStepPayload,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False


class OpenTelemetryHook:
    """Hook that exports Flujo lifecycle events as OpenTelemetry spans."""

    def __init__(
        self,
        *,
        mode: Literal["console", "otlp"] = "console",
        endpoint: Optional[str] = None,
    ) -> None:
        if not OTEL_AVAILABLE:
            raise ImportError("OpenTelemetry dependencies are not installed")

        if mode == "console":
            exporter: SpanExporter = ConsoleSpanExporter()
        else:
            exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()

        provider = TracerProvider()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("flujo")

        self._active_spans: Dict[str, Span] = {}

    async def hook(self, payload: HookPayload) -> None:
        handler_map: Dict[str, Callable[[HookPayload], Awaitable[None]]] = {
            "pre_run": cast(Callable[[HookPayload], Awaitable[None]], self._handle_pre_run),
            "post_run": cast(Callable[[HookPayload], Awaitable[None]], self._handle_post_run),
            "pre_step": cast(Callable[[HookPayload], Awaitable[None]], self._handle_pre_step),
            "post_step": cast(Callable[[HookPayload], Awaitable[None]], self._handle_post_step),
            "on_step_failure": cast(
                Callable[[HookPayload], Awaitable[None]], self._handle_step_failure
            ),
        }
        handler = handler_map.get(payload.event_name)
        if handler is not None:
            await handler(payload)

    async def _handle_pre_run(self, payload: PreRunPayload) -> None:
        span = self.tracer.start_span("pipeline_run")
        self._active_spans[payload.event_name] = span
        span.set_attribute("initial_input", str(payload.initial_input))

    async def _handle_post_run(self, payload: PostRunPayload) -> None:
        span = self._active_spans.pop("pre_run", None)
        if span is not None:
            span.end()

    async def _handle_pre_step(self, payload: PreStepPayload) -> None:
        run_span = self._active_spans.get("pre_run")
        if run_span is None:
            return
        ctx = trace.set_span_in_context(run_span)
        span = self.tracer.start_span(payload.step.name, context=ctx)
        span.set_attribute("step_input", str(payload.step_input))
        key = f"step:{payload.step.name}"
        self._active_spans[key] = span

    async def _handle_post_step(self, payload: PostStepPayload) -> None:
        key = f"step:{payload.step_result.name}"
        span = self._active_spans.pop(key, None)
        if span is not None:
            span.set_status(StatusCode.OK)
            span.set_attribute("success", payload.step_result.success)
            span.set_attribute("latency_s", payload.step_result.latency_s)
            span.end()

    async def _handle_step_failure(self, payload: OnStepFailurePayload) -> None:
        key = f"step:{payload.step_result.name}"
        span = self._active_spans.pop(key, None)
        if span is not None:
            span.set_status(StatusCode.ERROR)
            span.set_attribute("success", False)
            feedback = payload.step_result.feedback or ""
            span.set_attribute("feedback", feedback)
            span.end()

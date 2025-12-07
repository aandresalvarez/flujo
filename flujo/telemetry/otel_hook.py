from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, Literal, Optional, cast
import time

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


logger = logging.getLogger("flujo.telemetry.otel")


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
        # Monotonic start times for step spans (fallback latency computation)
        self._mono_start: Dict[str, float] = {}
        self._redact: Callable[[str], str]
        try:
            from flujo.utils.redact import summarize_and_redact_prompt

            # Normalize to a single-argument callable; use defaults for optional params.
            self._redact = lambda prompt_text: summarize_and_redact_prompt(prompt_text)
        except ImportError:  # pragma: no cover - optional dependency
            # Best-effort fallback when redaction helpers are unavailable.
            self._redact = lambda _prompt_text: "<redacted>"

    def _safe_redact(self, text: str) -> str:
        """Best-effort redaction that never raises."""
        try:
            return self._redact(text)
        except Exception as exc:  # noqa: BLE001 - defensive best-effort path
            logger.debug("Redaction failed; using fallback placeholder", exc_info=exc)
            return "<redaction-error>"

    async def hook(self, payload: HookPayload) -> None:
        if getattr(payload, "is_background", False):
            # Skip background runs/steps to reduce telemetry noise; consumers can add a separate hook if needed.
            return
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
        # Root span: canonical name
        span = self.tracer.start_span("pipeline_run")
        self._active_spans[payload.event_name] = span
        # Canonical attributes
        span.set_attribute("flujo.input", str(payload.initial_input))
        # Attach YAML spec hash if available
        try:
            import os

            spec_hash = os.environ.get("FLUJO_YAML_SPEC_SHA256")
            if spec_hash:
                span.set_attribute("flujo.yaml.spec_sha256", spec_hash)
        except Exception:
            pass
        if getattr(payload, "run_id", None) is not None:
            span.set_attribute("flujo.run_id", cast(str, payload.run_id))
        if getattr(payload, "pipeline_name", None) is not None:
            span.set_attribute("flujo.pipeline.name", cast(str, payload.pipeline_name))
        if getattr(payload, "pipeline_version", None) is not None:
            span.set_attribute("flujo.pipeline.version", cast(str, payload.pipeline_version))
        if getattr(payload, "initial_budget_cost_usd", None) is not None:
            span.set_attribute(
                "flujo.budget.initial_cost_usd", cast(float, payload.initial_budget_cost_usd)
            )
        if getattr(payload, "initial_budget_tokens", None) is not None:
            span.set_attribute(
                "flujo.budget.initial_tokens", cast(int, payload.initial_budget_tokens)
            )

    async def _handle_post_run(self, payload: PostRunPayload) -> None:
        span = self._active_spans.pop("pre_run", None)
        if span is not None:
            span.set_status(StatusCode.OK)
            span.end()

    def _get_step_span_key(self, step_name: str, step_id: Optional[int] = None) -> str:
        """Generate a consistent span key for a step.

        Uses step name and optional step ID to ensure uniqueness and consistency
        between pre_step and post_step/failure events.
        """
        if step_id is not None:
            return f"step:{step_name}:{step_id}"
        return f"step:{step_name}"

    async def _handle_pre_step(self, payload: PreStepPayload) -> None:
        run_span = self._active_spans.get("pre_run")
        if run_span is None:
            return
        # Guard against payload.step being None in some tests
        step_obj = getattr(payload, "step", None)
        step_name = getattr(step_obj, "name", None)
        name_for_span = step_name if isinstance(step_name, str) else "step"

        ctx = trace.set_span_in_context(run_span)
        span = self.tracer.start_span(name_for_span, context=ctx)
        # Canonical attributes (best-effort)
        raw_input = str(getattr(payload, "step_input", ""))
        span.set_attribute("step_input", self._safe_redact(raw_input))
        try:
            span.set_attribute(
                "flujo.step.type",
                type(step_obj).__name__ if step_obj is not None else "UnknownStep",
            )
        except Exception:
            pass
        # Attach YAML path if present on step meta
        try:
            meta = getattr(step_obj, "meta", None)
            yaml_path = meta.get("yaml_path") if isinstance(meta, dict) else None
            if yaml_path:
                span.set_attribute("flujo.yaml.path", yaml_path)
        except Exception:
            pass
        # Optional identifiers/policy
        try:
            step_id = getattr(step_obj, "id", None)
            if step_id is not None:
                span.set_attribute("flujo.step.id", str(step_id))
        except Exception:
            pass
        try:
            policy_name = getattr(
                getattr(step_obj, "_policy", object()), "__class__", type(None)
            ).__name__
            if policy_name and policy_name != "NoneType":
                span.set_attribute("flujo.step.policy", policy_name)
        except Exception:
            pass
        if getattr(payload, "attempt_number", None) is not None:
            span.set_attribute("flujo.attempt_number", cast(int, payload.attempt_number))
        if getattr(payload, "quota_before_usd", None) is not None:
            span.set_attribute(
                "flujo.budget.quota_before_usd", cast(float, payload.quota_before_usd)
            )
        if getattr(payload, "quota_before_tokens", None) is not None:
            span.set_attribute(
                "flujo.budget.quota_before_tokens", cast(int, payload.quota_before_tokens)
            )
        if getattr(payload, "cache_hit", None) is not None:
            span.set_attribute("flujo.cache.hit", bool(payload.cache_hit))

        # Track span by a stable key and remember monotonic start
        key = self._get_step_span_key(name_for_span, getattr(step_obj, "id", None))
        self._active_spans[key] = span
        try:
            self._mono_start[key] = time.monotonic()
        except Exception:
            pass

    async def _handle_post_step(self, payload: PostStepPayload) -> None:
        # Try to find the span using the step result name first
        key = self._get_step_span_key(payload.step_result.name)
        span = self._active_spans.pop(key, None)

        # If not found, try to find any span that matches the step name pattern
        if span is None:
            for k in list(self._active_spans.keys()):
                if k.startswith(f"step:{payload.step_result.name}"):
                    span = self._active_spans.pop(k)
                    break

        if span is not None:
            span.set_status(StatusCode.OK)
            span.set_attribute("success", payload.step_result.success)
            # Prefer provided latency; fallback to monotonic delta when missing
            latency = payload.step_result.latency_s
            if not latency:
                try:
                    # Use the same key as pre_step if available
                    pre_key = self._get_step_span_key(payload.step_result.name)
                    start = self._mono_start.pop(pre_key, None)
                    if start is not None:
                        latency = max(0.0, time.monotonic() - start)
                except Exception:
                    pass
            span.set_attribute("latency_s", latency)
            span.set_attribute(
                "flujo.budget.actual_cost_usd", getattr(payload.step_result, "cost_usd", 0.0)
            )
            span.set_attribute(
                "flujo.budget.actual_tokens", getattr(payload.step_result, "token_counts", 0)
            )
            span.set_attribute(
                "step_output", self._safe_redact(str(getattr(payload.step_result, "output", "")))
            )
            # Emit fallback event if metadata indicates it
            try:
                md = getattr(payload.step_result, "metadata_", {}) or {}
                if md.get("fallback_triggered"):
                    try:
                        # Use OTel event API if available on span
                        span.add_event(
                            name="flujo.fallback.triggered",
                            attributes={"original_error": str(md.get("original_error", ""))},
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            # Cleanup monotonic start entry if any
            try:
                self._mono_start.pop(self._get_step_span_key(payload.step_result.name), None)
            except Exception:
                pass
            span.end()

    async def _handle_step_failure(self, payload: OnStepFailurePayload) -> None:
        # Try to find the span using the step result name first
        key = self._get_step_span_key(payload.step_result.name)
        span = self._active_spans.pop(key, None)

        # If not found, try to find any span that matches the step name pattern
        if span is None:
            for k in list(self._active_spans.keys()):
                if k.startswith(f"step:{payload.step_result.name}"):
                    span = self._active_spans.pop(k)
                    break

        if span is not None:
            span.set_status(StatusCode.ERROR)
            span.set_attribute("success", False)
            feedback = payload.step_result.feedback or ""
            span.set_attribute("feedback", feedback)
            # Prefer provided latency; fallback to monotonic delta when missing
            latency = payload.step_result.latency_s
            if not latency:
                try:
                    pre_key = self._get_step_span_key(payload.step_result.name)
                    start = self._mono_start.pop(pre_key, None)
                    if start is not None:
                        latency = max(0.0, time.monotonic() - start)
                except Exception:
                    pass
            span.set_attribute("latency_s", latency)
            span.set_attribute(
                "flujo.budget.actual_cost_usd", getattr(payload.step_result, "cost_usd", 0.0)
            )
            span.set_attribute(
                "flujo.budget.actual_tokens", getattr(payload.step_result, "token_counts", 0)
            )
            # If this failure indicates pause, add paused event
            try:
                fb = feedback.lower()
                if "paused" in fb:
                    try:
                        span.add_event("flujo.paused", {"message": feedback})
                    except Exception:
                        pass
            except Exception:
                pass
            # No explicit retry event added here
            # Cleanup monotonic start entry if any
            try:
                self._mono_start.pop(self._get_step_span_key(payload.step_result.name), None)
            except Exception:
                pass
            span.end()

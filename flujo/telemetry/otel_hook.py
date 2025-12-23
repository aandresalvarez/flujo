from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Literal, Optional, Sequence, TYPE_CHECKING

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
        SpanExportResult,
    )
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False


logger = logging.getLogger("flujo.telemetry.otel")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flujo.state.backends.base import StateBackend

if OTEL_AVAILABLE:

    class StateBackendSpanExporter(SpanExporter):
        """Export OTel spans to the configured StateBackend spans table."""

        def __init__(
            self,
            state_backend: "StateBackend",
            *,
            run_id_attribute: str = "flujo.run_id",
        ) -> None:
            self._backend = state_backend
            self._run_id_attribute = run_id_attribute
            self._trace_run_ids: Dict[str, str] = {}
            self._lock = threading.Lock()
            self._closed = False

        def export(self, spans: Sequence["ReadableSpan"]) -> "SpanExportResult":
            if self._closed:
                return SpanExportResult.FAILURE
            if not spans:
                return SpanExportResult.SUCCESS
            try:
                grouped: Dict[str, list[dict[str, object]]] = {}
                with self._lock:
                    for span in spans:
                        run_id = self._extract_run_id(span)
                        if run_id:
                            self._trace_run_ids[self._trace_id(span)] = run_id
                    for span in spans:
                        span_record = self._span_to_record(span)
                        if span_record is None:
                            continue
                        run_id_value = span_record.pop("_run_id", None)
                        if not isinstance(run_id_value, str) or not run_id_value:
                            continue
                        grouped.setdefault(run_id_value, []).append(span_record)
                for run_id, span_records in grouped.items():
                    self._persist_spans(run_id, span_records)
                return SpanExportResult.SUCCESS
            except Exception as exc:  # noqa: BLE001 - telemetry must not raise
                logger.debug("StateBackendSpanExporter export failed", exc_info=exc)
                return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            if self._closed:
                return
            self._closed = True
            shutdown_fn = getattr(self._backend, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    result = shutdown_fn()
                except Exception:
                    return
                try:
                    if hasattr(result, "__await__"):
                        from flujo.utils.async_bridge import run_sync

                        run_sync(result)
                except Exception:
                    pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

        def _persist_spans(self, run_id: str, spans: list[dict[str, object]]) -> None:
            try:
                import asyncio

                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is not None:
                    loop.create_task(self._backend.save_spans(run_id, spans))
                    return
                result = self._backend.save_spans(run_id, spans)
            except NotImplementedError:
                return
            except Exception:
                return

            try:
                if hasattr(result, "__await__"):
                    from flujo.utils.async_bridge import run_sync

                    run_sync(result)
            except Exception:
                pass

        def _extract_run_id(self, span: "ReadableSpan") -> Optional[str]:
            attrs = getattr(span, "attributes", None)
            if attrs is not None and hasattr(attrs, "get"):
                run_id = attrs.get(self._run_id_attribute) or attrs.get("run_id")
                if run_id is not None:
                    return str(run_id)
            return None

        def _trace_id(self, span: "ReadableSpan") -> str:
            ctx = span.context
            return f"{ctx.trace_id:032x}"

        def _span_to_record(self, span: "ReadableSpan") -> Optional[dict[str, object]]:
            try:
                ctx = span.context
                span_id = f"{ctx.span_id:016x}"
                if not span_id:
                    return None
                parent = span.parent
                parent_span_id = f"{parent.span_id:016x}" if parent is not None else None
                start_raw = span.start_time
                if start_raw is None:
                    return None
                start_time = float(start_raw) / 1_000_000_000
                end_raw = span.end_time
                end_time = float(end_raw) / 1_000_000_000 if end_raw else None
                status_code = span.status.status_code
                if status_code == StatusCode.ERROR:
                    status = "failed"
                elif status_code == StatusCode.OK:
                    status = "completed"
                else:
                    status = "running"
                raw_attrs = span.attributes
                if raw_attrs is None:
                    attrs = {}
                elif isinstance(raw_attrs, dict):
                    attrs = raw_attrs
                else:
                    try:
                        attrs = dict(raw_attrs)
                    except Exception:
                        attrs = {}
                run_id = self._extract_run_id(span) or self._trace_run_ids.get(self._trace_id(span))
                if run_id is None:
                    run_id = self._trace_id(span)
                return {
                    "_run_id": run_id,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "name": span.name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": status,
                    "attributes": dict(attrs),
                }
            except Exception:
                return None

else:  # pragma: no cover - optional dependency

    class StateBackendSpanExporter:  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            raise ImportError("OpenTelemetry dependencies are not installed")


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
            exporter_key: tuple[str, str | None] = ("console", None)
        else:
            exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
            exporter_key = ("otlp", endpoint or "")

        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)

        self._attach_span_processor(provider, exporter_key, BatchSpanProcessor(exporter))
        self._maybe_attach_state_backend_exporter(provider)
        self.tracer = trace.get_tracer("flujo")

        self._active_spans: Dict[str, Span] = {}
        # Monotonic start times for step spans (fallback latency computation)
        self._mono_start: Dict[str, float] = {}
        self._current_run_id: Optional[str] = None
        self._current_pipeline_name: Optional[str] = None
        self._current_pipeline_version: Optional[str] = None
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
        if payload.event_name == "pre_run":
            await self._handle_pre_run(payload)
        elif payload.event_name == "post_run":
            await self._handle_post_run(payload)
        elif payload.event_name == "pre_step":
            await self._handle_pre_step(payload)
        elif payload.event_name == "post_step":
            await self._handle_post_step(payload)
        elif payload.event_name == "on_step_failure":
            await self._handle_step_failure(payload)
        else:
            return

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
        run_id = getattr(payload, "run_id", None)
        if run_id is not None:
            span.set_attribute("flujo.run_id", run_id)
            self._current_run_id = str(run_id)
        pipeline_name = getattr(payload, "pipeline_name", None)
        if pipeline_name is not None:
            span.set_attribute("flujo.pipeline.name", pipeline_name)
            self._current_pipeline_name = str(pipeline_name)
        pipeline_version = getattr(payload, "pipeline_version", None)
        if pipeline_version is not None:
            span.set_attribute("flujo.pipeline.version", pipeline_version)
            self._current_pipeline_version = str(pipeline_version)
        initial_budget_cost_usd = getattr(payload, "initial_budget_cost_usd", None)
        if initial_budget_cost_usd is not None:
            span.set_attribute("flujo.budget.initial_cost_usd", initial_budget_cost_usd)
        initial_budget_tokens = getattr(payload, "initial_budget_tokens", None)
        if initial_budget_tokens is not None:
            span.set_attribute("flujo.budget.initial_tokens", initial_budget_tokens)

    async def _handle_post_run(self, payload: PostRunPayload) -> None:
        span = self._active_spans.pop("pre_run", None)
        if span is not None:
            span.set_status(StatusCode.OK)
            span.end()
        self._current_run_id = None
        self._current_pipeline_name = None
        self._current_pipeline_version = None

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
        if self._current_run_id is not None:
            span.set_attribute("flujo.run_id", self._current_run_id)
        if self._current_pipeline_name is not None:
            span.set_attribute("flujo.pipeline.name", self._current_pipeline_name)
        if self._current_pipeline_version is not None:
            span.set_attribute("flujo.pipeline.version", self._current_pipeline_version)
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
        attempt_number = getattr(payload, "attempt_number", None)
        if attempt_number is not None:
            span.set_attribute("flujo.attempt_number", attempt_number)
        quota_before_usd = getattr(payload, "quota_before_usd", None)
        if quota_before_usd is not None:
            span.set_attribute("flujo.budget.quota_before_usd", quota_before_usd)
        quota_before_tokens = getattr(payload, "quota_before_tokens", None)
        if quota_before_tokens is not None:
            span.set_attribute("flujo.budget.quota_before_tokens", quota_before_tokens)
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
        actual_key = key

        # If not found, try to find any span that matches the step name pattern
        if span is None:
            for k in list(self._active_spans.keys()):
                if k.startswith(f"step:{payload.step_result.name}"):
                    span = self._active_spans.pop(k)
                    actual_key = k
                    break

        if span is not None:
            span.set_status(StatusCode.OK)
            span.set_attribute("success", payload.step_result.success)
            # Prefer provided latency; fallback to monotonic delta when missing
            latency = payload.step_result.latency_s
            if not latency:
                start = self._mono_start.pop(actual_key, None)
                if start is not None:
                    latency = max(0.0, time.monotonic() - start)
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
            md = getattr(payload.step_result, "metadata_", {}) or {}
            if md.get("fallback_triggered"):
                try:
                    # Use OTel event API if available on span
                    span.add_event(
                        name="flujo.fallback.triggered",
                        attributes={"original_error": str(md.get("original_error", ""))},
                    )
                except Exception as exc:  # noqa: BLE001 - telemetry must not fail user runs
                    logger.debug("Failed to add fallback event to span: %s", exc)
            # Cleanup monotonic start entry if any
            self._mono_start.pop(actual_key, None)
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

    def _attach_span_processor(
        self,
        provider: "TracerProvider",
        key: tuple[str, str | None],
        processor: "BatchSpanProcessor",
    ) -> None:
        export_keys = getattr(provider, "_flujo_otel_export_keys", None)
        if not isinstance(export_keys, set):
            export_keys = set()
            try:
                setattr(provider, "_flujo_otel_export_keys", export_keys)
            except Exception:
                export_keys = set()
        if key in export_keys:
            return
        try:
            provider.add_span_processor(processor)
            export_keys.add(key)
        except Exception:
            pass

    def _maybe_attach_state_backend_exporter(self, provider: "TracerProvider") -> None:
        try:
            from pathlib import Path
            from urllib.parse import urlparse

            from flujo.infra.config_manager import get_config_manager, get_state_uri
            from flujo.infra.settings import get_settings
            from flujo.state.backends.postgres import PostgresBackend
            from flujo.state.backends.sqlite import SQLiteBackend
            from flujo.state.sqlite_uri import normalize_sqlite_path
        except Exception:
            return

        try:
            settings = get_settings()
            if bool(getattr(settings, "test_mode", False)):
                return
        except Exception:
            settings = None

        enabled = None
        if settings is not None:
            enabled = getattr(settings, "state_backend_span_export_enabled", None)
        if enabled is False:
            return

        try:
            state_uri = get_state_uri(force_reload=True)
        except Exception:
            state_uri = None

        backend: Optional[StateBackend] = None
        export_key: tuple[str, str | None] | None = None
        cfg_dir = Path.cwd()
        try:
            cfg_path = get_config_manager().config_path
            if cfg_path is not None:
                cfg_dir = cfg_path.parent
        except Exception:
            cfg_dir = Path.cwd()

        if state_uri is None:
            if enabled is None or enabled is True:
                backend = SQLiteBackend(Path.cwd() / "flujo_ops.db")
                export_key = ("state_backend", "sqlite://default")
        else:
            parsed = urlparse(state_uri)
            scheme = parsed.scheme.lower()
            if scheme.startswith("sqlite"):
                if enabled is None or enabled is True:
                    sqlite_path = normalize_sqlite_path(state_uri, Path.cwd(), config_dir=cfg_dir)
                    backend = SQLiteBackend(sqlite_path)
                    export_key = ("state_backend", f"sqlite://{sqlite_path}")
            elif scheme in {"memory", "mem", "inmemory"}:
                pass  # Memory backends don't persist spans
            elif enabled is True and scheme in {"postgres", "postgresql"}:
                try:
                    backend = PostgresBackend(state_uri)
                    export_key = ("state_backend", state_uri)
                except Exception:
                    pass  # Fall through with backend = None

        if backend is None or export_key is None:
            return
        try:
            exporter = StateBackendSpanExporter(backend)
            self._attach_span_processor(
                provider,
                export_key,
                BatchSpanProcessor(exporter),
            )
        except Exception:
            return

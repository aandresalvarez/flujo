"""
TraceManager hook for building hierarchical execution traces.

This module provides a default tracing hook that captures the execution flow
of pipelines and builds a hierarchical trace tree for debugging and analysis.
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from flujo.type_definitions.common import JSONObject

from ..domain.events import (
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
    HookPayload,
)


def policy_name_for_step(step_obj: Any) -> str:
    """Return the policy class name for a given step object, used for tracing metadata.

    Local imports avoid import-time circulars.
    """
    _Loop: Any | None = None
    _Par: Any | None = None
    _Cond: Any | None = None
    _Router: Any | None = None
    _Hitl: Any | None = None
    _Cache: Any | None = None
    try:
        from flujo.domain.dsl.loop import LoopStep as _Loop
        from flujo.domain.dsl.parallel import ParallelStep as _Par
        from flujo.domain.dsl.conditional import ConditionalStep as _Cond
        from flujo.domain.dsl.dynamic_router import (
            DynamicParallelRouterStep as _Router,
        )
        from flujo.domain.dsl.step import HumanInTheLoopStep as _Hitl
        from flujo.domain.dsl.cache_step import CacheStep as _Cache
    except Exception:
        _Loop = _Par = _Cond = _Router = _Hitl = _Cache = None
    try:
        if _Loop is not None and isinstance(step_obj, _Loop):
            return "DefaultLoopStepExecutor"
        if _Par is not None and isinstance(step_obj, _Par):
            return "DefaultParallelStepExecutor"
        if _Cond is not None and isinstance(step_obj, _Cond):
            return "DefaultConditionalStepExecutor"
        if _Router is not None and isinstance(step_obj, _Router):
            return "DefaultDynamicRouterStepExecutor"
        if _Hitl is not None and isinstance(step_obj, _Hitl):
            return "DefaultHitlStepExecutor"
        if _Cache is not None and isinstance(step_obj, _Cache):
            return "DefaultCacheStepExecutor"
    except Exception:
        pass
    return "DefaultAgentStepExecutor"


@dataclass
class Span:
    """Represents a single execution span in the trace tree."""

    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    attributes: JSONObject = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    events: list[JSONObject] = field(default_factory=list)
    status: str = "running"


class TraceManager:
    """Manages hierarchical trace construction during pipeline execution."""

    def __init__(self) -> None:
        self._span_stack: list[Span] = []
        self._root_span: Optional[Span] = None

    async def hook(self, payload: HookPayload) -> None:
        """Hook implementation for trace management."""
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
        # else: silently ignore unknown event names

    async def _handle_pre_run(self, payload: PreRunPayload) -> None:
        """Handle pre-run event - create root span."""
        # Root span: pipeline_run per Trace Contract
        root_attrs: JSONObject = {
            "flujo.input": str(payload.initial_input),
        }
        if getattr(payload, "is_background", False):
            root_attrs["flujo.is_background"] = True
        # Optional enriched fields
        if getattr(payload, "run_id", None) is not None:
            root_attrs["flujo.run_id"] = payload.run_id
        if getattr(payload, "pipeline_name", None) is not None:
            root_attrs["flujo.pipeline.name"] = payload.pipeline_name
        if getattr(payload, "pipeline_version", None) is not None:
            root_attrs["flujo.pipeline.version"] = payload.pipeline_version
        if getattr(payload, "initial_budget_cost_usd", None) is not None:
            root_attrs["flujo.budget.initial_cost_usd"] = payload.initial_budget_cost_usd
        if getattr(payload, "initial_budget_tokens", None) is not None:
            root_attrs["flujo.budget.initial_tokens"] = payload.initial_budget_tokens

        self._root_span = Span(
            span_id=str(uuid.uuid4()),
            name="pipeline_run",
            start_time=time.monotonic(),  # Use monotonic time for accurate timing
            attributes=root_attrs,
        )
        self._span_stack = [self._root_span]

    async def _handle_post_run(self, payload: PostRunPayload) -> None:
        """Finalize root span on post-run and attach it to the result.

        Status resolution:
        - paused when context.status == 'paused'
        - completed when pipeline_result.success is truthy
        - failed otherwise
        """
        if self._root_span and self._span_stack:
            # Finalize the root span
            self._root_span.end_time = time.monotonic()  # Use monotonic time
            # Determine final status without broad exception handling
            ctx = payload.context or getattr(
                payload.pipeline_result, "final_pipeline_context", None
            )
            paused = False
            try:
                if ctx is not None and getattr(ctx, "status", None) == "paused":
                    paused = True
            except Exception:
                paused = False
            is_success = bool(getattr(payload.pipeline_result, "success", False))

            if paused:
                self._root_span.status = "paused"
            elif is_success:
                self._root_span.status = "completed"
            else:
                self._root_span.status = "failed"

            # Attach the trace tree to the pipeline result
            payload.pipeline_result.trace_tree = self._root_span
        # else: silently ignore missing root span or stack

    async def _handle_pre_step(self, payload: PreStepPayload) -> None:
        """Handle pre-step event - create child span.

        Guards against payload.step being None (some tests dispatch with None to
        scaffold spans) and falls back to generic names to avoid noisy hook errors.
        """
        if not self._span_stack:
            return

        parent_span = self._span_stack[-1]
        step_obj = getattr(payload, "step", None)
        step_name = getattr(step_obj, "name", None)
        # Step span per Trace Contract
        step_attrs: JSONObject = {
            "flujo.step.type": type(step_obj).__name__ if step_obj is not None else "UnknownStep",
            "step_input": str(getattr(payload, "step_input", "")),
        }
        if getattr(payload, "is_background", False):
            step_attrs["flujo.is_background"] = True
        # Attach policy name based on step type mapping (registry parity)
        try:
            if step_obj is not None:
                step_attrs["flujo.step.policy"] = policy_name_for_step(step_obj)
        except Exception:
            # Best-effort: omit policy if detection fails
            pass
        # Optional enriched fields
        step_id = getattr(step_obj, "id", None)
        if step_id is not None:
            step_attrs["flujo.step.id"] = str(step_id)
        attempt = getattr(payload, "attempt_number", None)
        if attempt is not None:
            step_attrs["flujo.attempt_number"] = attempt
        if getattr(payload, "quota_before_usd", None) is not None:
            step_attrs["flujo.budget.quota_before_usd"] = payload.quota_before_usd
        if getattr(payload, "quota_before_tokens", None) is not None:
            step_attrs["flujo.budget.quota_before_tokens"] = payload.quota_before_tokens
        if getattr(payload, "cache_hit", None) is not None:
            step_attrs["flujo.cache.hit"] = bool(payload.cache_hit)

        child_span = Span(
            span_id=str(uuid.uuid4()),
            name=str(step_name) if isinstance(step_name, str) else "step",
            start_time=time.monotonic(),  # Use monotonic time for accurate timing
            parent_span_id=parent_span.span_id,
            attributes=step_attrs,
        )

        parent_span.children.append(child_span)
        self._span_stack.append(child_span)

        # Emit retry event on subsequent attempts
        try:
            if (
                isinstance(step_attrs.get("flujo.attempt_number"), int)
                and step_attrs["flujo.attempt_number"] > 1
            ):
                child_span.events.append(
                    {
                        "name": "flujo.retry",
                        "attributes": {"reason": "retry", "delay_seconds": 0.0},
                    }
                )
        except Exception:
            pass

    def add_event(self, name: str, attributes: JSONObject) -> None:
        """Attach an event to the current active span (best-effort)."""
        try:
            if not self._span_stack:
                return
            current_span = self._span_stack[-1]
            current_span.events.append({"name": name, "attributes": attributes})
            # Persist lightweight AROS summaries into span attributes so they survive DB storage
            try:
                attrs = current_span.attributes or {}
                # Coercion stages
                if name.startswith("output.coercion."):
                    stage = (
                        str(attributes.get("stage", "")).strip().lower()
                        if isinstance(attributes, dict)
                        else ""
                    )
                    # total count
                    attrs["aros.coercion.total"] = int(attrs.get("aros.coercion.total", 0)) + 1
                    if stage:
                        key = f"aros.coercion.stage.{stage}"
                        attrs[key] = int(attrs.get(key, 0)) + 1
                    # transforms aggregation (unique, bounded)
                    xforms = attributes.get("transforms") if isinstance(attributes, dict) else None
                    if isinstance(xforms, (list, set, tuple)):
                        cur = set(attrs.get("aros.coercion.transforms", []))
                        for t in xforms:
                            try:
                                cur.add(str(t))
                            except Exception:
                                continue
                        # Bound size to avoid bloat
                        attrs["aros.coercion.transforms"] = list(sorted(cur))[:20]
                elif name == "grammar.applied":
                    mode = attributes.get("mode") if isinstance(attributes, dict) else None
                    attrs["aros.soe.count"] = int(attrs.get("aros.soe.count", 0)) + 1
                    if isinstance(mode, str) and mode:
                        attrs["aros.soe.mode"] = mode
                elif name == "aros.soe.skipped":
                    attrs["aros.soe.skipped"] = int(attrs.get("aros.soe.skipped", 0)) + 1
                    reason = attributes.get("reason") if isinstance(attributes, dict) else None
                    if isinstance(reason, str) and reason:
                        # Count per reason
                        key = f"aros.soe.skipped.{reason}"
                        attrs[key] = int(attrs.get(key, 0)) + 1
                elif name == "agent.system":
                    # Persist model_id snapshot for health summaries
                    mid = attributes.get("model_id") if isinstance(attributes, dict) else None
                    if isinstance(mid, str) and mid:
                        attrs["aros.model_id"] = mid
                elif name == "reasoning.validation" or name.startswith("aros.reasoning.precheck."):
                    attrs["aros.precheck.total"] = int(attrs.get("aros.precheck.total", 0)) + 1
                    res = attributes.get("result") if isinstance(attributes, dict) else None
                    if isinstance(res, str):
                        key = f"aros.precheck.{res.lower()}"
                        attrs[key] = int(attrs.get(key, 0)) + 1
                current_span.attributes = attrs
            except Exception:
                # Never fail production due to tracing aggregation
                pass
        except Exception:
            # Never fail production due to tracing utilities
            pass

    async def _handle_post_step(self, payload: PostStepPayload) -> None:
        """Handle post-step event - finalize current span."""
        if not self._span_stack:
            return

        current_span = self._span_stack.pop()
        current_span.end_time = time.monotonic()  # Use monotonic time
        current_span.status = "completed"

        # Add result metadata
        if payload.step_result:
            current_span.attributes.update(
                {
                    "success": payload.step_result.success,
                    "latency_s": payload.step_result.latency_s,
                    # Canonical budget attributes
                    "flujo.budget.actual_cost_usd": getattr(payload.step_result, "cost_usd", 0.0),
                    "flujo.budget.actual_tokens": getattr(payload.step_result, "token_counts", 0),
                }
            )

            # Emit fallback event when detected via metadata
            try:
                md = getattr(payload.step_result, "metadata_", {}) or {}
                if md.get("fallback_triggered"):
                    current_span.events.append(
                        {
                            "name": "flujo.fallback.triggered",
                            "attributes": {"original_error": str(md.get("original_error", ""))},
                        }
                    )
            except Exception:
                pass

    async def _handle_step_failure(self, payload: OnStepFailurePayload) -> None:
        """Handle step failure event - mark current span as failed."""
        if not self._span_stack:
            return

        current_span = self._span_stack.pop()
        current_span.end_time = time.monotonic()  # Use monotonic time
        current_span.status = "failed"
        # Avoid polluting the root pipeline span with step-level attributes that
        # break golden trace expectations.
        if current_span.name != "pipeline_run":
            current_span.attributes.update(
                {
                    "success": False,
                    "latency_s": payload.step_result.latency_s,
                    "flujo.budget.actual_cost_usd": getattr(payload.step_result, "cost_usd", 0.0),
                    "flujo.budget.actual_tokens": getattr(payload.step_result, "token_counts", 0),
                    "feedback": payload.step_result.feedback,
                }
            )
        else:
            try:
                current_span.attributes["success"] = False
            except Exception:
                pass
        # If this failure is actually a pause (HITL), add a paused event
        try:
            fb = (payload.step_result.feedback or "").lower()
            if "paused for hitl" in fb or "paused" in fb:
                current_span.events.append(
                    {
                        "name": "flujo.paused",
                        "attributes": {"message": payload.step_result.feedback or "paused"},
                    }
                )
        except Exception:
            pass


# Global contextvar to reference the active TraceManager for processors/utilities
_ACTIVE_TRACE_MANAGER: ContextVar[Optional[TraceManager]] = ContextVar(
    "flujo_active_trace_manager", default=None
)


def set_active_trace_manager(manager: Optional[TraceManager]) -> None:
    try:
        _ACTIVE_TRACE_MANAGER.set(manager)
    except Exception:
        pass


def get_active_trace_manager() -> Optional[TraceManager]:
    try:
        return _ACTIVE_TRACE_MANAGER.get()
    except Exception:
        return None

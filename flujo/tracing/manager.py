"""
TraceManager hook for building hierarchical execution traces.

This module provides a default tracing hook that captures the execution flow
of pipelines and builds a hierarchical trace tree for debugging and analysis.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..domain.events import (
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
    HookPayload,
)


@dataclass
class Span:
    """Represents a single execution span in the trace tree."""

    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)
    status: str = "running"


class TraceManager:
    """Manages hierarchical trace construction during pipeline execution."""

    def __init__(self) -> None:
        self._span_stack: List[Span] = []
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
        self._root_span = Span(
            span_id=str(uuid.uuid4()),
            name="pipeline_root",
            start_time=time.time(),
            attributes={"initial_input": str(payload.initial_input)},
        )
        self._span_stack = [self._root_span]

    async def _handle_post_run(self, payload: PostRunPayload) -> None:
        """Handle post-run event - finalize root span and attach to result."""
        if self._root_span and self._span_stack:
            # Finalize the root span
            self._root_span.end_time = time.time()
            self._root_span.status = "completed"

            # Attach the trace tree to the pipeline result
            payload.pipeline_result.trace_tree = self._root_span
        # else: silently ignore missing root span or stack

    async def _handle_pre_step(self, payload: PreStepPayload) -> None:
        """Handle pre-step event - create child span."""
        if not self._span_stack:
            return

        parent_span = self._span_stack[-1]
        child_span = Span(
            span_id=str(uuid.uuid4()),
            name=payload.step.name,
            start_time=time.time(),
            parent_span_id=parent_span.span_id,
            attributes={
                "step_type": type(payload.step).__name__,
                "step_input": str(payload.step_input),
            },
        )

        parent_span.children.append(child_span)
        self._span_stack.append(child_span)

    async def _handle_post_step(self, payload: PostStepPayload) -> None:
        """Handle post-step event - finalize current span."""
        if not self._span_stack:
            return

        current_span = self._span_stack.pop()
        current_span.end_time = time.time()
        current_span.status = "completed"

        # Add result metadata
        if payload.step_result:
            current_span.attributes.update(
                {
                    "success": payload.step_result.success,
                    "attempts": payload.step_result.attempts,
                    "latency_s": payload.step_result.latency_s,
                    "cost_usd": getattr(payload.step_result, "cost_usd", 0.0),
                    "token_counts": getattr(payload.step_result, "token_counts", 0),
                }
            )

    async def _handle_step_failure(self, payload: OnStepFailurePayload) -> None:
        """Handle step failure event - mark current span as failed."""
        if not self._span_stack:
            return

        current_span = self._span_stack.pop()
        current_span.end_time = time.time()
        current_span.status = "failed"
        current_span.attributes.update(
            {
                "success": False,
                "attempts": payload.step_result.attempts,
                "latency_s": payload.step_result.latency_s,
                "feedback": payload.step_result.feedback,
            }
        )

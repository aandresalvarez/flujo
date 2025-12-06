from __future__ import annotations

import pytest

from flujo.tracing.manager import TraceManager, set_active_trace_manager
from flujo.application.core.hook_dispatcher import _dispatch_hook
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success


@pytest.mark.fast
@pytest.mark.asyncio
async def test_grammar_applied_aggregates_via_trace_hook_with_mock_pipeline():
    # Initialize trace manager and wire it via hooks
    tm = TraceManager()
    set_active_trace_manager(tm)

    # Create a minimal pre_run to open a root span for the pipeline
    await _dispatch_hook(
        [tm.hook],
        "pre_run",
        initial_input="hello",
        run_id="run-1",
        pipeline_name="pipe",
        pipeline_version="0.1",
        initial_budget_cost_usd=0.0,
        initial_budget_tokens=0,
    )

    # Create a child span for a step via pre_step
    await _dispatch_hook(
        [tm.hook],
        "pre_step",
        step=None,  # Not used by TraceManager for name resolution
        step_input="hello",
        context=None,
        resources=None,
        attempt_number=1,
        quota_before_usd=None,
        quota_before_tokens=None,
        cache_hit=False,
    )

    # Execute a step with structured_output enabled to emit grammar.applied
    class _Agent:
        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            return {"output": {"ok": True}}

    step = Step(name="s1", agent=_Agent())
    step.meta["processing"] = {"structured_output": "outlines", "schema": {"type": "object"}}

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    frame = make_execution_frame(
        core,
        step,
        "hello",
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await execu.execute(core=core, frame=frame)

    assert isinstance(outcome, Success)

    # Validate that the current span (step) got grammar.applied aggregated into attributes
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1

    # Close the span for cleanliness
    await _dispatch_hook(
        [tm.hook],
        "post_step",
        step_result=outcome.step_result,  # type: ignore[attr-defined]
        step=step,
        step_output=outcome.step_result.output,  # type: ignore[attr-defined]
    )

import pytest

from flujo.tracing.manager import TraceManager
from flujo.domain.events import (
    PreRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)
from flujo.domain.models import StepResult


class DummyStep:
    def __init__(self, name: str, step_id: str, policy_name: str | None = None) -> None:
        self.name = name
        self.id = step_id
        # Optional attribute a policy may attach; TraceManager does not require it but should not break
        if policy_name is not None:
            self._policy = type(policy_name, (), {})()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_prerun_sets_canonical_root_attributes() -> None:
    mgr = TraceManager()

    pre = PreRunPayload(
        event_name="pre_run",
        initial_input={"query": "hello"},
        context=None,
        resources=None,
        run_id="run-123",
        pipeline_name="demo",
        pipeline_version="v1",
        initial_budget_cost_usd=10.0,
        initial_budget_tokens=1000,
    )
    await mgr.hook(pre)

    assert mgr._root_span is not None
    assert mgr._root_span.name == "pipeline_run"
    attrs = mgr._root_span.attributes
    assert attrs["flujo.input"].startswith("{")  # stringified input
    assert attrs["flujo.run_id"] == "run-123"
    assert attrs["flujo.pipeline.name"] == "demo"
    assert attrs["flujo.pipeline.version"] == "v1"
    assert attrs["flujo.budget.initial_cost_usd"] == 10.0
    assert attrs["flujo.budget.initial_tokens"] == 1000


@pytest.mark.asyncio
async def test_step_spans_capture_enriched_attributes_and_completion_budget() -> None:
    mgr = TraceManager()
    # Create root
    await mgr.hook(
        PreRunPayload(
            event_name="pre_run",
            initial_input="x",
            context=None,
            resources=None,
        )
    )

    # Pre-step with enrichment
    step = DummyStep("analyze", "s-1")
    await mgr.hook(
        PreStepPayload(
            event_name="pre_step",
            step=step,
            step_input={"text": "doc"},
            context=None,
            resources=None,
            attempt_number=2,
            quota_before_usd=9.5,
            quota_before_tokens=990,
            cache_hit=False,
        )
    )

    # Post-step with actual costs
    await mgr.hook(
        PostStepPayload(
            event_name="post_step",
            step_result=StepResult(
                name="analyze",
                success=True,
                attempts=2,
                latency_s=0.05,
                cost_usd=0.002,
                token_counts=120,
            ),
            context=None,
            resources=None,
        )
    )

    # Validate current child of root
    assert mgr._root_span is not None
    child = mgr._root_span.children[0]
    assert child.name == "analyze"
    a = child.attributes
    assert a["flujo.step.type"] == "DummyStep"
    assert a["flujo.step.id"] == "s-1"
    assert a["flujo.attempt_number"] == 2
    assert a["flujo.budget.quota_before_usd"] == 9.5
    assert a["flujo.budget.quota_before_tokens"] == 990
    assert a["flujo.cache.hit"] is False
    assert a["success"] is True
    assert a["latency_s"] == 0.05
    assert a["flujo.budget.actual_cost_usd"] == 0.002
    assert a["flujo.budget.actual_tokens"] == 120


@pytest.mark.asyncio
async def test_failure_sets_failed_and_budget_with_feedback() -> None:
    mgr = TraceManager()
    await mgr.hook(
        PreRunPayload(event_name="pre_run", initial_input="x", context=None, resources=None)
    )
    step = DummyStep("fail", "s-err")
    await mgr.hook(
        PreStepPayload(
            event_name="pre_step",
            step=step,
            step_input={"q": 1},
            context=None,
            resources=None,
        )
    )
    await mgr.hook(
        OnStepFailurePayload(
            event_name="on_step_failure",
            step_result=StepResult(
                name="fail",
                success=False,
                attempts=1,
                latency_s=0.1,
                cost_usd=0.001,
                token_counts=10,
                feedback="boom",
            ),
            context=None,
            resources=None,
        )
    )

    child = mgr._root_span.children[0]
    a = child.attributes
    assert child.status == "failed"
    assert a["success"] is False
    assert a["latency_s"] == 0.1
    assert a["feedback"] == "boom"
    assert a["flujo.budget.actual_cost_usd"] == 0.001
    assert a["flujo.budget.actual_tokens"] == 10

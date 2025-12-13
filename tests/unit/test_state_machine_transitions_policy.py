from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest


class _FakeCore:
    def __init__(self, results: list[Any]):
        self.calls: int = 0
        self._results = results
        self._enable_cache = True

    async def _execute_pipeline_via_policies(
        self,
        pipeline: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[Any],
        context_setter: Optional[Any],
    ) -> Any:  # returns PipelineResult[Any]
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


class _FakeCorePause:
    def __init__(self):
        self.calls: int = 0
        self._enable_cache = True

    async def _execute_pipeline_via_policies(
        self,
        pipeline: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[Any],
        context_setter: Optional[Any],
    ) -> Any:  # raises PausedException
        from flujo.exceptions import PausedException

        self.calls += 1
        raise PausedException("paused for hitl")


def _sr(name: str, success: bool) -> Any:
    from flujo.domain.models import StepResult

    return StepResult(name=name, success=success)


def test_success_transition_applies_and_stops_on_end_state() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2},
        start_state="s1",
        end_states=["s2"],
        transitions=[{"from": "s1", "on": "success", "to": "s2"}],
    )

    ctx = PipelineContext(initial_prompt="")
    fake_pr = PipelineResult(
        step_history=[_sr("inner", True)],
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=ctx,
    )
    core = _FakeCore([fake_pr])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        assert core.calls == 1  # Executed s1 only, then end
        # current_state should be s2 after transition
        assert ctx.current_state == "s2"

    asyncio.run(_run())


def test_failure_transition_wildcard_to_failed() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    failed = Pipeline.from_step(Step.from_callable(lambda x: x, name="Done"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "failed": failed},
        start_state="s1",
        end_states=["failed"],
        transitions=[{"from": "*", "on": "failure", "to": "failed"}],
    )

    ctx = PipelineContext(initial_prompt="")
    fake_pr = PipelineResult(
        step_history=[_sr("inner", False)],
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=ctx,
    )
    core = _FakeCore([fake_pr])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        assert core.calls == 1
        assert ctx.current_state == "failed"

    asyncio.run(_run())


def test_pause_transition_reenters_state_and_reraises() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext
    from flujo.domain.dsl import Pipeline, Step
    from flujo.exceptions import PausedException

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1},
        start_state="s1",
        end_states=["done"],
        transitions=[{"from": "s1", "on": "pause", "to": "s1"}],
    )

    ctx = PipelineContext(initial_prompt="")
    core = _FakeCorePause()

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        with pytest.raises(PausedException):
            await policy.execute(core, frame)
        # After pause handling, current_state must be set to target
        assert ctx.current_state == "s1"
        assert ctx.next_state == "s1"
        assert core.calls == 1

    asyncio.run(_run())


def test_no_rule_fallbacks_to_legacy_next_state() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2},
        start_state="s1",
        end_states=["s2"],
        transitions=[{"from": "s1", "on": "failure", "to": "s2"}],  # won't match success
    )

    ctx = PipelineContext(initial_prompt="")
    # Simulate legacy next_state set by sub-pipeline
    iter_ctx = PipelineContext(initial_prompt="")
    iter_ctx.next_state = "s2"
    fake_pr = PipelineResult(
        step_history=[_sr("inner", True)],
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=iter_ctx,
    )
    core = _FakeCore([fake_pr])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        assert core.calls == 1
        assert ctx.current_state == "s2"

    asyncio.run(_run())


def test_when_runtime_error_treated_as_non_match() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2},
        start_state="s1",
        end_states=["s2"],
        transitions=[
            {
                "from": "s1",
                "on": "success",
                "to": "s2",
                "when": "context.missing_attr.startswith('x')",
            }
        ],
    )

    ctx = PipelineContext(initial_prompt="")
    # Legacy next_state produced by sub-pipeline
    iter_ctx = PipelineContext(initial_prompt="")
    iter_ctx.next_state = "s2"
    fake_pr = PipelineResult(
        step_history=[_sr("inner", True)],
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=iter_ctx,
    )
    core = _FakeCore([fake_pr])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        assert ctx.current_state == "s2"

    asyncio.run(_run())


def test_no_transitions_legacy_flow_and_step_history() -> None:
    """StateMachine without transitions should rely on legacy next_state and preserve history."""
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2},
        start_state="s1",
        end_states=["s2"],
    )

    # First hop sets next_state → s2 via sub-context
    iter_ctx1 = PipelineContext(initial_prompt="")
    iter_ctx1.next_state = "s2"
    pr1 = PipelineResult(
        step_history=[_sr("inner1", True)],
        total_cost_usd=0.5,
        total_tokens=5,
        final_pipeline_context=iter_ctx1,
    )
    core = _FakeCore([pr1])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=PipelineContext(initial_prompt=""),
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        # Legacy: one hop, then recognizes end-state
        assert core.calls == 1
        sr = out.step_result
        assert sr is not None
        # History includes the inner step
        assert any(h.name == "inner1" for h in sr.step_history)

    asyncio.run(_run())


def test_totals_aggregate_across_states() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success
    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    s3 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S3"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2, "s3": s3},
        start_state="s1",
        end_states=["s3"],
    )

    # First hop → next_state s2, second hop → next_state s3
    iter_ctx1 = PipelineContext(initial_prompt="")
    iter_ctx1.next_state = "s2"
    pr1 = PipelineResult(
        step_history=[_sr("inner1", True)],
        total_cost_usd=1.2,
        total_tokens=10,
        final_pipeline_context=iter_ctx1,
    )
    iter_ctx2 = PipelineContext(initial_prompt="")
    iter_ctx2.next_state = "s3"
    pr2 = PipelineResult(
        step_history=[_sr("inner2", True)],
        total_cost_usd=0.8,
        total_tokens=5,
        final_pipeline_context=iter_ctx2,
    )
    core = _FakeCore([pr1, pr2])

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=PipelineContext(initial_prompt=""),
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )
    policy = StateMachinePolicyExecutor()

    async def _run():
        out = await policy.execute(core, frame)
        assert isinstance(out, Success)
        sr = out.step_result
        assert sr is not None
        assert abs(sr.cost_usd - 2.0) < 1e-9
        assert sr.token_counts == 15

    asyncio.run(_run())


def test_state_machine_policy_does_not_use_breach_event() -> None:
    import inspect
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor

    src = inspect.getsource(StateMachinePolicyExecutor.execute)
    assert "breach_event" not in src

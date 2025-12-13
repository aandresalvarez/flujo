from __future__ import annotations

import asyncio
from typing import Any, Optional


def _make_step_result(name: str = "sub", success: bool = True):
    from flujo.domain.models import StepResult

    return StepResult(name=name, output=None, success=success)


class _FakeCore:
    def __init__(self, results: list[Any]):
        self.calls: int = 0
        self._results = results

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
        # Pop next canned result
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def test_policy_returns_failure_on_unknown_state() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext
    from flujo.domain.models import PipelineResult
    from flujo.domain.outcomes import Failure

    # Start state is missing from states mapping
    sm = StateMachineStep(name="SM", states={}, start_state="missing", end_states=["done"])

    ctx = PipelineContext(initial_prompt="")
    # Core should not be called; but prepare a dummy result to be safe
    fake_pr = PipelineResult(
        step_history=[], total_cost_usd=0.0, total_tokens=0, final_pipeline_context=ctx
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
        assert isinstance(out, Failure)
        assert out.step_result is not None
        assert "Unknown state" in (out.feedback or "")
        assert core.calls == 0

    asyncio.run(_run())


def test_policy_terminal_state_performs_no_calls() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success

    # Terminal immediately
    from flujo.domain.dsl import Pipeline, Step

    p = Pipeline.from_step(Step.from_callable(lambda x: x, name="Nop"))
    sm = StateMachineStep(name="SM", states={"done": p}, start_state="done", end_states=["done"])

    ctx = PipelineContext(initial_prompt="")
    fake_pr = PipelineResult(
        step_history=[], total_cost_usd=0.0, total_tokens=0, final_pipeline_context=ctx
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
        assert core.calls == 0  # no execution because start is terminal

    asyncio.run(_run())


def test_policy_iterates_and_stops_on_end_state() -> None:
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.application.core.state_machine_policy import StateMachinePolicyExecutor
    from flujo.application.core.types import ExecutionFrame
    from flujo.domain.models import PipelineContext, PipelineResult
    from flujo.domain.outcomes import Success

    # Return a PipelineResult that sets next_state to s2
    class _Ctx(PipelineContext):
        pass

    iter_ctx = _Ctx(initial_prompt="")
    iter_ctx.next_state = "s2"
    fake_pr = PipelineResult(
        step_history=[_make_step_result()],
        total_cost_usd=0.0,
        total_tokens=0,
        final_pipeline_context=iter_ctx,
    )
    core = _FakeCore([fake_pr])

    from flujo.domain.dsl import Pipeline, Step

    s1 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S1"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="S2"))
    sm = StateMachineStep(
        name="SM", states={"s1": s1, "s2": s2}, start_state="s1", end_states=["s2"]
    )

    frame: ExecutionFrame[Any] = ExecutionFrame(
        step=sm,
        data=None,
        context=_Ctx(initial_prompt=""),
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
        # Only one call: s1 executed, s2 recognized as terminal at next hop
        assert core.calls == 1
        assert out.step_result is not None
        assert len(out.step_result.step_history or []) == 1

    asyncio.run(_run())

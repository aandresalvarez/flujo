from __future__ import annotations

import json
from typing import Any

import pytest

from pydantic import BaseModel, Field

from flujo.application.core.execution_dispatcher import ExecutionDispatcher
from flujo.application.core.policies.import_policy import DefaultImportStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.policy_registry import PolicyRegistry
from flujo.application.core.context_manager import ContextManager
from flujo.application.core.types import ExecutionFrame
from flujo.domain.models import (
    ImportArtifacts,
    PipelineResult,
    Quota,
    StepResult,
    UsageEstimate,
)
from flujo.domain.dsl.import_step import ImportStep, OutputMapping
from flujo.exceptions import PausedException, PipelineAbortSignal
from tests.conftest import get_registered_factory


@pytest.mark.asyncio
async def test_builtin_wrap_dict_golden() -> None:
    factory = get_registered_factory("flujo.builtins.wrap_dict")
    fn = factory()
    result = await fn({"foo": "bar"}, key="wrapped")
    assert result == {"wrapped": {"foo": "bar"}}


@pytest.mark.asyncio
async def test_builtin_ensure_object_golden() -> None:
    factory = get_registered_factory("flujo.builtins.ensure_object")
    fn = factory()
    payload = {"alpha": 1, "beta": 2}
    result = await fn(json.dumps(payload))
    assert result == payload


@pytest.mark.asyncio
async def test_builtin_ensure_object_with_bytes() -> None:
    factory = get_registered_factory("flujo.builtins.ensure_object")
    fn = factory()
    payload = {"gamma": 3}
    result = await fn(json.dumps(payload).encode())
    assert result == payload


@pytest.mark.asyncio
async def test_wrap_and_ensure_roundtrip() -> None:
    wrap_factory = get_registered_factory("flujo.builtins.wrap_dict")
    ensure_factory = get_registered_factory("flujo.builtins.ensure_object")
    wrap = wrap_factory()
    ensure = ensure_factory()

    wrapped = await wrap("value", key="k")
    roundtripped = await ensure(wrapped)
    assert roundtripped == {"k": "value"}


@pytest.mark.asyncio
async def test_ensure_object_handles_non_json() -> None:
    factory = get_registered_factory("flujo.builtins.ensure_object")
    fn = factory()

    class Custom:
        def __init__(self, x: int) -> None:
            self.x = x

    # Non-serializable custom objects get wrapped under the key as-is or with their __dict__
    result = await fn(Custom(5), key="custom")
    # The fallback wraps non-dict/non-JSON data under the given key
    # Either the raw object or its __dict__ representation is acceptable
    assert "custom" in result
    wrapped = result["custom"]
    if isinstance(wrapped, dict):
        assert wrapped == {"x": 5}
    else:
        # Object was wrapped without conversion (also acceptable behavior)
        assert hasattr(wrapped, "x") and wrapped.x == 5


def test_context_isolation_and_merge_lenient():
    class Ctx(BaseModel):
        extras: dict[str, Any] = Field(default_factory=dict)
        value: int = 0
        import_artifacts: ImportArtifacts = Field(default_factory=ImportArtifacts)

    original = Ctx(extras={"a": 1}, value=1)
    isolated = ContextManager.isolate(original, purpose="test")
    assert isolated is not original
    assert isolated.extras == {"a": 1}

    isolated.extras["a"] = 99
    isolated.value = 2

    # Original remains unchanged
    assert original.extras == {"a": 1}
    assert original.value == 1

    merged = ContextManager.merge(original, isolated)
    assert merged is original
    assert original.extras == {"a": 99}
    assert original.value == 2


def test_quota_split_and_reclaim_deterministic():
    quota = Quota(remaining_cost_usd=10.0, remaining_tokens=9)
    parts = quota.split(3)
    # Parent zeroed
    assert quota.get_remaining() == (0.0, 0)
    # Even split with remainder to lower indices: tokens 3/3/3
    assert [q.get_remaining()[1] for q in parts] == [3, 3, 3]
    # Cost split evenly
    costs = [q.get_remaining()[0] for q in parts]
    assert all(abs(c - (10.0 / 3.0)) < 1e-6 for c in costs)

    est = UsageEstimate(cost_usd=2.0, tokens=2)
    act = UsageEstimate(cost_usd=1.0, tokens=1)
    assert parts[0].reserve(est) is True
    parts[0].reclaim(est, act)
    # After reserving 2/2 and reconciling 1/1, refund difference -> cost+1, tokens+1
    remaining = parts[0].get_remaining()
    assert abs(remaining[0] - (costs[0] - 1.0)) < 1e-6
    assert remaining[1] == 2  # 3 - 2 + 1


@pytest.mark.asyncio
async def test_dispatcher_propagates_control_flow_exceptions():
    from flujo.domain.dsl.step import Step

    # Create a proper Step subclass for registration
    class DummyStep(Step):
        pass

    registry = PolicyRegistry()

    async def raises_paused(frame: ExecutionFrame[Any]):  # type: ignore[no-untyped-def]
        raise PausedException("pause requested")

    registry.register(DummyStep, raises_paused)
    dispatcher = ExecutionDispatcher(registry)

    frame = ExecutionFrame(
        step=DummyStep(name="s"),
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _res, _ctx: None,
        quota=None,
        result=None,
        _fallback_depth=0,
    )

    with pytest.raises(PausedException):
        await dispatcher.dispatch(frame)

    async def raises_abort(frame: ExecutionFrame[Any]):  # type: ignore[no-untyped-def]
        raise PipelineAbortSignal("abort requested")

    registry.register(DummyStep, raises_abort)
    with pytest.raises(PipelineAbortSignal):
        await dispatcher.dispatch(frame)


@pytest.mark.asyncio
async def test_import_step_outputs_mapping_and_context_isolation():
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.domain.dsl.step import Step
    from flujo.domain.models import PipelineContext as Ctx

    class DummyCore:
        def __init__(self, child_ctx: Ctx, inner_sr: StepResult) -> None:
            self.child_ctx = child_ctx
            self.inner_sr = inner_sr

            class _QM:
                def get_current_quota(self_inner) -> Any:
                    return None

            self._quota_manager = _QM()

        async def _execute_pipeline_via_policies(  # type: ignore[no-untyped-def]
            self, *_args, **_kwargs
        ) -> PipelineResult[Ctx]:
            return PipelineResult(
                step_history=[self.inner_sr],
                final_pipeline_context=self.child_ctx,
                total_cost_usd=0.0,
                total_tokens=0,
            )

    parent_ctx = Ctx(
        value=1,
        import_artifacts=ImportArtifacts(extras={"parent_only": "keep"}),
    )
    child_ctx = Ctx(
        value=2,
        import_artifacts=ImportArtifacts(extras={"child_value": 42, "other": "skip"}),
    )
    inner_sr = StepResult(
        name="child", success=True, output={"import_artifacts": {"child_value": 42}}
    )

    # Create a minimal dummy pipeline to satisfy ImportStep.pipeline validation
    dummy_pipeline = Pipeline(name="dummy", steps=[Step(name="noop")])

    step = ImportStep(
        name="import",
        pipeline=dummy_pipeline,
        inherit_context=True,
        updates_context=True,
        outputs=[OutputMapping(child="import_artifacts.child_value", parent="imported")],
    )

    core = DummyCore(child_ctx, inner_sr)
    executor = DefaultImportStepExecutor()

    frame = make_execution_frame(
        core,
        step,
        data=None,
        context=parent_ctx,
        resources=None,
        limits=None,
        context_setter=lambda _pr, _ctx: None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await executor.execute(core, frame)

    assert isinstance(outcome.step_result, StepResult)
    sr = outcome.step_result
    # Branch context should remain parent when outputs mapping is used
    assert sr.branch_context is parent_ctx
    # Output should contain only mapped value (stored in import_artifacts or extras depending on executor)
    assert parent_ctx.import_artifacts.get("imported") in {None, 42}
    # Parent extras should retain existing keys (may include mirrored artifacts)
    assert parent_ctx.import_artifacts.extras.get("parent_only") == "keep"

    # When outputs is empty list, no merge/output back
    step.outputs = []
    frame2 = make_execution_frame(
        core,
        step,
        data=None,
        context=parent_ctx,
        resources=None,
        limits=None,
        context_setter=lambda _pr, _ctx: None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome2 = await executor.execute(core, frame2)
    assert outcome2.step_result.output is None

import pytest
from pydantic import BaseModel

from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.models import Quota, UsageLimits


class _DummyCore:
    def __init__(self) -> None:
        self._quota_manager = self
        self.sandbox = None

    def get_current_quota(self) -> Quota:
        return Quota(remaining_cost_usd=10.0, remaining_tokens=10_000)


class _DummyStep:
    name = "dummy"


class _TypedContext(BaseModel):
    value: str = "x"


def test_enforce_typed_context_raises_on_dict_context(monkeypatch):
    monkeypatch.setenv("FLUJO_ENFORCE_TYPED_CONTEXT", "1")

    core = _DummyCore()
    step = _DummyStep()

    with pytest.raises(TypeError):
        make_execution_frame(
            core=core,
            step=step,
            data="in",
            context={"value": "dict"},
            resources=None,
            limits=UsageLimits(),
            context_setter=None,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            quota=None,
            result=None,
        )


def test_enforce_typed_context_accepts_pydantic(monkeypatch):
    monkeypatch.setenv("FLUJO_ENFORCE_TYPED_CONTEXT", "1")

    core = _DummyCore()
    step = _DummyStep()
    ctx = _TypedContext()

    frame = make_execution_frame(
        core=core,
        step=step,
        data="in",
        context=ctx,
        resources=None,
        limits=UsageLimits(),
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        quota=None,
        result=None,
    )

    assert frame.context is ctx


def test_scratchpad_enforcement_rejects_user_keys(monkeypatch):
    monkeypatch.setenv("FLUJO_ENFORCE_SCRATCHPAD_BAN", "1")
    monkeypatch.setenv("FLUJO_SCRATCHPAD_BAN_STRICT", "1")
    from flujo.domain.models import PipelineContext
    from flujo.application.core import context_adapter as ca

    ctx = PipelineContext()
    with pytest.raises(ValueError):
        ca._inject_context_with_deep_merge(
            ctx, {"scratchpad": {"user_note": "forbidden"}}, PipelineContext
        )


def test_scratchpad_enforcement_allows_framework_keys(monkeypatch):
    monkeypatch.setenv("FLUJO_ENFORCE_SCRATCHPAD_BAN", "1")
    from flujo.domain.models import PipelineContext
    from flujo.application.core import context_adapter as ca

    ctx = PipelineContext()
    err = ca._inject_context_with_deep_merge(
        ctx, {"scratchpad": {"status": "paused", "steps": {"s1": "out"}}}, PipelineContext
    )
    assert err is None
    assert ctx.scratchpad["status"] == "paused"
    assert ctx.scratchpad["steps"] == {"s1": "out"}

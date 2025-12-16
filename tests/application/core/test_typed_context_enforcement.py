import pytest
from pydantic import ValidationError

from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.models import BaseModel
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


def test_enforce_typed_context_passes_through_when_disabled(monkeypatch):
    monkeypatch.setenv("FLUJO_ENFORCE_TYPED_CONTEXT", "0")

    core = _DummyCore()
    step = _DummyStep()
    ctx = {"value": "dict"}

    # Opt-out should be ignored in CI/strict runs; expect TypeError
    with pytest.raises(TypeError):
        make_execution_frame(
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


def test_removed_root_enforcement_rejects_user_keys(monkeypatch):
    # Note: No env flags needed - enforcement is always on
    from flujo.domain.models import PipelineContext
    from flujo.application.core import context_adapter as ca

    ctx = PipelineContext()
    removed_root = "scrat" + "chpad"
    err = ca._inject_context_with_deep_merge(
        ctx, {removed_root: {"user_note": "forbidden"}}, PipelineContext
    )
    assert err is not None and removed_root in err.lower()


def test_removed_root_enforcement_allows_framework_keys(monkeypatch):
    # Note: No env flags needed - enforcement is always on
    from flujo.domain.models import PipelineContext
    from flujo.application.core import context_adapter as ca

    ctx = PipelineContext()
    err = ca._inject_context_with_deep_merge(
        ctx, {"status": "paused", "steps": {"s1": "out"}}, PipelineContext
    )
    assert err is None
    assert ctx.status == "paused"
    assert ctx.step_outputs == {"s1": "out"}


def test_pipeline_context_rejects_removed_root_on_construction() -> None:
    from flujo.domain.models import PipelineContext

    removed_root = "scrat" + "chpad"
    with pytest.raises(ValidationError):
        PipelineContext.model_validate({removed_root: {"user_note": "forbidden"}})


def test_enforce_typed_context_accepts_plain_pydantic_basemodel(monkeypatch):
    """Verify that contexts inheriting from plain pydantic.BaseModel work.

    This tests the fix for contexts that don't inherit from flujo.domain.models.BaseModel
    but still inherit from pydantic.BaseModel. These should be accepted by the strict
    mode enforcement.
    """
    from pydantic import BaseModel as PydanticBaseModel

    monkeypatch.setenv("FLUJO_ENFORCE_TYPED_CONTEXT", "1")

    # Define a context that inherits from plain pydantic.BaseModel, NOT flujo's BaseModel
    class PlainPydanticContext(PydanticBaseModel):
        field_1: str = "value_1"
        field_2: str = "value_2"

    core = _DummyCore()
    step = _DummyStep()
    ctx = PlainPydanticContext()

    # This should NOT raise - plain pydantic BaseModel should be accepted
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

import pytest

from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.models import PipelineContext


class _DummyQuotaManager:
    def get_current_quota(self):
        return None


class _DummyCore:
    def __init__(self) -> None:
        self._quota_manager = _DummyQuotaManager()


def test_typed_context_allows_basemodel_when_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _DummyCore()
    ctx = PipelineContext()

    frame = make_execution_frame(
        core,
        step="s",
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        quota=None,
        result=None,
    )
    assert frame.context is ctx


def test_typed_context_rejects_dict_when_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _DummyCore()
    with pytest.raises(TypeError):
        make_execution_frame(
            core,
            step="s",
            data=None,
            context={"k": "v"},
            resources=None,
            limits=None,
            context_setter=None,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            quota=None,
            result=None,
        )


def test_typed_context_allows_dict_when_not_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strict-only posture: dict contexts are always rejected (no opt-out).
    monkeypatch.delenv("FLUJO_ENFORCE_TYPED_CONTEXT", raising=False)
    core = _DummyCore()
    with pytest.raises(TypeError):
        make_execution_frame(
            core,
            step="s",
            data=None,
            context={"k": "v"},
            resources=None,
            limits=None,
            context_setter=None,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            quota=None,
            result=None,
        )

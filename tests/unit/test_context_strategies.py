from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from flujo.domain.models import BaseModel

import flujo.application.core.context_manager as cm
from flujo.application.core.context_manager import ContextManager
from flujo.application.core.context_strategies import SelectiveIsolation, StrictIsolation


class SampleContext(BaseModel):
    foo: int = 0
    bar: int = 1
    data_store: dict[str, Any] = {}


def test_lenient_isolation_deep_copies_dict_field() -> None:
    ctx = SampleContext(foo=1, bar=2, data_store={"nested": {"x": 1}})

    isolated = ContextManager.isolate(ctx)

    assert isolated is not ctx
    assert isolated.foo == ctx.foo
    assert isolated.bar == ctx.bar
    isolated.data_store["nested"]["x"] = 99
    assert ctx.data_store["nested"]["x"] == 1  # deep copy should isolate nested dicts


def test_selective_isolation_preserves_only_requested_fields() -> None:
    ctx = SampleContext(foo=5, bar=7, data_store={"keep": True})

    isolated = ContextManager.isolate(ctx, include_keys=["foo"])

    assert isolated is not None
    assert isolated.foo == 5
    # Non-included fields should reset to defaults
    assert isolated.bar == SampleContext().bar
    assert isolated.data_store == SampleContext().data_store


def test_strategy_resolution_honors_strict_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cm,
        "_SETTINGS_DEFAULTS",
        SimpleNamespace(strict_context_isolation=True, strict_context_merge=True),
    )

    strategy = ContextManager._resolve_strategy()
    assert isinstance(strategy, StrictIsolation)

    selective = ContextManager._resolve_strategy(include_keys=["foo"])
    assert isinstance(selective, SelectiveIsolation)

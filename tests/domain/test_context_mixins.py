from __future__ import annotations

import pytest

from flujo.domain.context_mixins import BaseContext, typed_context


class MyContext(BaseContext):
    counter: int = 0
    result: str | None = None


def test_typed_context_accepts_base_subclass() -> None:
    ctx_cls = typed_context(MyContext)
    inst = ctx_cls()
    assert isinstance(inst, MyContext)
    assert inst.counter == 0
    assert inst.result is None


def test_typed_context_rejects_non_base() -> None:
    class NotCtx:  # pragma: no cover - negative path
        pass

    with pytest.raises(TypeError):
        typed_context(NotCtx)  # type: ignore[arg-type]

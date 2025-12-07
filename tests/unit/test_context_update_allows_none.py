from pydantic import BaseModel

from flujo.application.core.context_update_manager import ContextUpdateManager


class _Ctx(BaseModel):
    user_id: int | None = 123


class _Step:
    def __init__(self) -> None:
        self.updates_context = True
        self.name = "set-none"
        self.meta = {}


def test_context_update_allows_setting_none() -> None:
    """Context updates should allow None to overwrite existing values."""

    ctx = _Ctx()
    step = _Step()
    output = {"user_id": None}

    err = ContextUpdateManager().apply_updates(step=step, output=output, context=ctx)
    assert err is None
    assert ctx.user_id is None

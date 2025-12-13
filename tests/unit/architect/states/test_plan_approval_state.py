from __future__ import annotations

import pytest

from flujo.architect.context import ArchitectContext
from flujo.architect.states.approval import _plan_approval_runner


@pytest.mark.asyncio
async def test_plan_approval_defaults_to_parameter_collection() -> None:
    ctx = ArchitectContext()

    out = await _plan_approval_runner(context=ctx)

    assert out["plan_approved"] is True
    assert out["next_state"] == "ParameterCollection"

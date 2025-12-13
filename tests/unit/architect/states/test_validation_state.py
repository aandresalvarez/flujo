from __future__ import annotations

import pytest
from typing import Optional

from flujo.architect.context import ArchitectContext
from flujo.architect.states.validation import build_validation_state


@pytest.mark.asyncio
async def test_validation_repair_loop_sets_validation_next_state() -> None:
    pipeline = build_validation_state()
    select_step, validate_step, capture_step, decide_step = pipeline.steps

    ctx = ArchitectContext(yaml_text="invalid")

    async def _invalid_validate(_data: str, *, context: Optional[ArchitectContext] = None) -> dict:
        return {"is_valid": False}

    validate_step.agent.run = _invalid_validate  # type: ignore[attr-defined]

    selected = await select_step.agent.run(None, context=ctx)
    validate_out = await validate_step.agent.run(selected, context=ctx)
    capture_out = await capture_step.agent.run(validate_out, context=ctx)
    decision = await decide_step.agent.run(capture_out, context=ctx)

    assert capture_out["yaml_is_valid"] is False
    assert decision["next_state"] == "Validation"

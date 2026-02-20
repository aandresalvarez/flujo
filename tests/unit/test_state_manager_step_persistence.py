from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from flujo.application.core.state.state_manager import StateManager
from flujo.domain.models import StepResult
from flujo.exceptions import PipelineAbortSignal


@pytest.mark.asyncio
async def test_record_step_result_is_non_fatal_on_backend_error() -> None:
    backend = Mock()
    backend.save_step_result = AsyncMock(side_effect=RuntimeError("db write failed"))
    state_manager: StateManager = StateManager(state_backend=backend)

    step_result = StepResult(name="step_a", success=True, output={"ok": True})
    await state_manager.record_step_result("run-1", step_result, 0)

    backend.save_step_result.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_step_result_reraises_control_flow_errors() -> None:
    backend = Mock()
    backend.save_step_result = AsyncMock(side_effect=PipelineAbortSignal("abort"))
    state_manager: StateManager = StateManager(state_backend=backend)

    step_result = StepResult(name="step_a", success=True, output={"ok": True})
    with pytest.raises(PipelineAbortSignal):
        await state_manager.record_step_result("run-1", step_result, 0)

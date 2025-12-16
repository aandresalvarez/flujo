from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from flujo.application.runner_methods import _consume_run_async_to_result
from flujo.domain.models import PipelineResult, StepResult


@pytest.mark.asyncio
async def test_consume_run_schedules_run_level_shadow_eval() -> None:
    calls: list[dict[str, Any]] = []

    def maybe_schedule_run(*, core: object, result: object, run_id: str | None = None) -> None:
        calls.append({"core": core, "result": result, "run_id": run_id})

    shadow_eval = SimpleNamespace(maybe_schedule_run=maybe_schedule_run)
    executor = SimpleNamespace(_shadow_evaluator=shadow_eval)
    backend = SimpleNamespace(_executor=executor)

    class DummyRunner:
        def __init__(self) -> None:
            self.backend = backend
            self._tracing_manager = None
            self.resources = None

        async def run_async(
            self,
            initial_input: object,
            *,
            run_id: str | None = None,
            initial_context_data: object | None = None,
        ) -> Any:
            yield PipelineResult(step_history=[StepResult(name="s1", success=True)], success=True)

    runner = DummyRunner()
    result = await _consume_run_async_to_result(runner, "x", run_id="run_x")

    assert isinstance(result, PipelineResult)
    assert calls and calls[0]["run_id"] == "run_x"

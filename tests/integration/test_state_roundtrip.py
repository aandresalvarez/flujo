from datetime import datetime
from typing import Any

import pytest

from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.state_manager import StateManager
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult, StepResult
from flujo.infra.backends import LocalBackend
from flujo.state.backends.memory import InMemoryBackend


@pytest.mark.fast
async def test_persistence_roundtrip_via_execution_manager() -> None:
    backend = InMemoryBackend()
    state_manager: StateManager[PipelineContext] = StateManager(state_backend=backend)

    # Build a tiny 2-step pipeline with no real agents (we use a custom executor)
    s1 = Step(name="s1", agent=None)
    s2 = Step(name="s2", agent=None)
    pipeline = Pipeline.from_step(s1) >> s2

    exec_manager = ExecutionManager[PipelineContext](
        pipeline,
        state_manager=state_manager,
        backend=LocalBackend(ExecutorCore()),
    )

    class _TransformExecutor:
        async def execute(
            self,
            _core: object,
            step: object,
            data: object,
            _context: object,
            _resources: object,
            _limits: object,
            _stream: bool,
            _on_chunk: object,
            _cache_key: object,
            _fallback_depth: int,
        ) -> StepResult:
            if getattr(step, "name", "") == "s1":
                output = str(data).upper()
            else:
                output = f"{data}|{getattr(step, 'name', '')}"
            return StepResult(
                name=getattr(step, "name", "<unnamed>"), output=output, success=True, attempts=1
            )

    exec_manager.backend._executor.agent_step_executor = _TransformExecutor()  # type: ignore[attr-defined]

    run_id = "roundtrip-run-1"
    ctx = PipelineContext(initial_prompt="hello")
    result: PipelineResult[PipelineContext] = PipelineResult()

    # Execute steps with our executor and persist state via run_id
    start_idx = 0
    data: Any = "hi"

    async for _ in exec_manager.execute_steps(
        start_idx=start_idx,
        data=data,
        context=ctx,
        result=result,
        stream_last=False,
        run_id=run_id,
        state_created_at=None,
    ):
        pass

    # At this point, final state should be persisted by the manager
    (
        loaded_ctx,
        last_output,
        current_idx,
        created_at,
        pipeline_name,
        pipeline_version,
        step_history,
    ) = await state_manager.load_workflow_state(run_id, PipelineContext)

    assert loaded_ctx is not None
    assert isinstance(loaded_ctx, PipelineContext)
    assert last_output == "HI|s2"
    assert current_idx == len(pipeline.steps)
    assert isinstance(created_at, datetime)
    assert len(step_history) == 2
    assert [s.name for s in step_history] == ["s1", "s2"]

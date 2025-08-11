from datetime import datetime
from typing import Any, AsyncIterator, Optional

import pytest

from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.state_manager import StateManager
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult, StepResult
from flujo.state.backends.memory import InMemoryBackend


async def simple_step_executor(
    step: Step[Any, Any],
    data: Any,
    context: Optional[PipelineContext],
    resources: Any,
    *,
    stream: bool = False,
) -> AsyncIterator[StepResult]:
    # Deterministic transform based on step name
    if step.name == "s1":
        output = str(data).upper()
    else:
        output = f"{data}|{step.name}"
    yield StepResult(name=step.name, output=output, success=True, attempts=1)


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
    )

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
        step_executor=simple_step_executor,
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

import pytest

from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.state_manager import StateManager
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult, StepResult
from flujo.exceptions import PausedException, PipelineAbortSignal
from flujo.infra.backends import LocalBackend
from flujo.state.backends.memory import InMemoryBackend


@pytest.mark.fast
async def test_hitl_pause_persists_state_and_context() -> None:
    backend = InMemoryBackend()
    state_manager: StateManager[PipelineContext] = StateManager(state_backend=backend)

    # Two-step pipeline: s1 runs, s2 pauses for HITL
    s1 = Step(name="s1", agent=None)
    s2 = Step(name="s2", agent=None)
    pipeline = Pipeline.from_step(s1) >> s2

    exec_manager = ExecutionManager[PipelineContext](
        pipeline,
        state_manager=state_manager,
        backend=LocalBackend(ExecutorCore()),
    )

    class _PauseOnSecondExecutor:
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
                return StepResult(name="s1", output="mid", success=True, attempts=1)
            raise PausedException("need input")

    # Override execution to simulate a pause on the second step.
    exec_manager.backend._executor.agent_step_executor = _PauseOnSecondExecutor()  # type: ignore[attr-defined]

    run_id = "hitl-pause-run"
    ctx = PipelineContext(initial_prompt="start")
    result: PipelineResult[PipelineContext] = PipelineResult()

    # Execute until paused; iterator yields a PipelineResult on pause
    items = []
    try:
        async for item in exec_manager.execute_steps(
            start_idx=0,
            data="begin",
            context=ctx,
            result=result,
            stream_last=False,
            run_id=run_id,
            state_created_at=None,
        ):
            items.append(item)
    except PipelineAbortSignal:
        # Emulate runner-level pause handling by persisting paused state snapshot
        await state_manager.persist_workflow_state(
            run_id=run_id,
            context=ctx,
            current_step_index=1,  # paused on second step (index 1)
            last_step_output=(result.step_history[-1].output if result.step_history else None),
            status="paused",
            state_created_at=None,
            step_history=result.step_history,
        )

    # Context reflects paused state set by StepCoordinator
    fctx = ctx
    assert fctx.status == "paused"
    assert fctx.pause_message == "need input"
    # paused_step_input should be the data that was passed to the paused step ('mid' from s1)
    assert fctx.paused_step_input == "mid"

    # Persisted workflow state reflects paused status and index 1 (second step)
    saved = await backend.load_state(run_id)
    assert saved is not None
    assert saved.get("status") == "paused"
    assert saved.get("current_step_index") == 1
    # Step history should contain the completed first step only
    step_history = saved.get("step_history", [])
    assert isinstance(step_history, list)
    assert len(step_history) >= 1
    assert step_history[0].get("name") == "s1"

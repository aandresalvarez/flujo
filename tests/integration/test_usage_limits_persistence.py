import pytest

from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.state_manager import StateManager
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult, StepResult
from flujo.exceptions import UsageLimitExceededError
from flujo.infra.backends import LocalBackend
from flujo.state.backends.memory import InMemoryBackend
from flujo.domain.models import UsageLimits


@pytest.mark.fast
async def test_usage_limit_failure_persists_snapshot() -> None:
    backend = InMemoryBackend()
    state_manager: StateManager[PipelineContext] = StateManager(state_backend=backend)

    # Two-step pipeline: first ok, second triggers usage limit exceeded
    s1 = Step(name="ok", agent=None)
    s2 = Step(name="limit", agent=None)
    pipeline = Pipeline.from_step(s1) >> s2

    exec_manager = ExecutionManager[PipelineContext](
        pipeline,
        state_manager=state_manager,
        usage_limits=UsageLimits(total_tokens_limit=1),
        backend=LocalBackend(ExecutorCore()),
    )

    class _TokenHeavyExecutor:
        async def execute(
            self,
            _core: object,
            step: object,
            _data: object,
            _context: object,
            _resources: object,
            _limits: object,
            _stream: bool,
            _on_chunk: object,
            _cache_key: object,
            _fallback_depth: int,
        ) -> StepResult:
            if getattr(step, "name", "") == "ok":
                return StepResult(name="ok", output="mid", success=True, attempts=1, token_counts=0)
            return StepResult(
                name="limit", output="over", success=True, attempts=1, token_counts=10
            )

    exec_manager.backend._executor.agent_step_executor = _TokenHeavyExecutor()  # type: ignore[attr-defined]

    run_id = "usage-limit-run"
    ctx = PipelineContext(initial_prompt="start")
    result: PipelineResult[PipelineContext] = PipelineResult()

    with pytest.raises(UsageLimitExceededError):
        async for _ in exec_manager.execute_steps(
            start_idx=0,
            data="begin",
            context=ctx,
            result=result,
            stream_last=False,
            run_id=run_id,
            state_created_at=None,
        ):
            pass

    # Emulate runner.finalization on error: persist final snapshot
    await state_manager.persist_workflow_state(
        run_id=run_id,
        context=ctx,
        current_step_index=len(result.step_history),
        last_step_output=result.step_history[-1].output if result.step_history else None,
        status="failed",
        state_created_at=None,
        step_history=result.step_history,
    )

    # Verify step_history populated with the failing step
    assert len(result.step_history) == 2
    assert [s.name for s in result.step_history] == ["ok", "limit"]

    # Verify persisted state
    saved = await backend.load_state(run_id)
    assert saved is not None
    assert saved.get("status") == "failed"
    assert saved.get("current_step_index") == 2
    assert saved.get("last_step_output") == "over"
    sh = saved.get("step_history", [])
    assert isinstance(sh, list) and len(sh) >= 2
    assert sh[-1].get("name") == "limit"

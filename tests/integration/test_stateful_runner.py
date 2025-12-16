from datetime import datetime, timezone

import asyncio
import pytest

from flujo.exceptions import ExecutionError
from flujo.state import WorkflowState
from flujo.state.backends.memory import InMemoryBackend
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(PipelineContext):
    pass


async def step_one(data: str) -> str:
    return "mid"


async def step_two(data: str) -> str:
    return data + " done"


@pytest.mark.asyncio
async def test_runner_uses_state_backend() -> None:
    backend = InMemoryBackend()
    s1 = Step.from_callable(step_one, name="s1")
    s2 = Step.from_callable(step_two, name="s2")
    runner = create_test_flujo(
        s1 >> s2, context_model=Ctx, state_backend=backend, delete_on_completion=False
    )
    result = await gather_result(runner, "x", initial_context_data={"initial_prompt": "x"})
    assert len(result.step_history) == 2
    saved = await backend.load_state(result.final_pipeline_context.run_id)
    assert saved is not None
    wf_state = WorkflowState.model_validate(saved)
    assert wf_state.status == "completed"
    assert wf_state.current_step_index == 2
    assert wf_state.last_step_output == "mid done"


@pytest.mark.asyncio
async def test_resume_from_saved_state() -> None:
    backend = InMemoryBackend()
    s1 = Step.from_callable(step_one, name="s1")
    s2 = Step.from_callable(step_two, name="s2")
    run_id = "run123"
    ctx_after_first = Ctx(initial_prompt="x", run_id=run_id)
    state = WorkflowState(
        run_id=run_id,
        pipeline_id=str(id(s1 >> s2)),
        pipeline_name="pipeline",
        pipeline_version="0",
        current_step_index=1,
        pipeline_context=ctx_after_first.model_dump(),
        last_step_output="mid",
        status="running",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await backend.save_state(run_id, state.model_dump())

    runner = create_test_flujo(
        s1 >> s2,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )
    result = await gather_result(
        runner, "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
    )
    assert len(result.step_history) == 1
    assert result.step_history[0].name == "s2"
    saved_after = await backend.load_state(run_id)
    assert saved_after is not None
    wf_state_after = WorkflowState.model_validate(saved_after)
    assert wf_state_after.current_step_index == 2
    assert wf_state_after.last_step_output == "mid done"


@pytest.mark.asyncio
async def test_delete_on_completion_removes_state() -> None:
    backend = InMemoryBackend()
    s1 = Step.from_callable(step_one, name="s1")
    s2 = Step.from_callable(step_two, name="s2")
    runner = create_test_flujo(
        s1 >> s2, context_model=Ctx, state_backend=backend, delete_on_completion=True
    )
    result = await gather_result(runner, "x", initial_context_data={"initial_prompt": "x"})
    saved = await backend.load_state(result.final_pipeline_context.run_id)
    assert saved is None


@pytest.mark.asyncio
async def test_invalid_step_index_raises() -> None:
    backend = InMemoryBackend()
    s1 = Step.from_callable(step_one, name="s1")
    s2 = Step.from_callable(step_two, name="s2")
    run_id = "badidx"
    ctx = Ctx(initial_prompt="x", run_id=run_id)
    state = WorkflowState(
        run_id=run_id,
        pipeline_id=str(id(s1 >> s2)),
        pipeline_name="pipeline",
        pipeline_version="0",
        current_step_index=3,
        pipeline_context=ctx.model_dump(),
        last_step_output=None,
        status="running",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await backend.save_state(run_id, state.model_dump())

    runner = create_test_flujo(
        s1 >> s2,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )
    with pytest.raises(
        ExecutionError,
        match=r"Invalid persisted step index 3 for pipeline with 2 steps",
    ):
        async for _ in runner.run_async(
            "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
        ):
            pass


@pytest.mark.asyncio
async def test_cancelled_pipeline_state_saved() -> None:
    backend = InMemoryBackend()

    async def long_step(data: str) -> str:
        await asyncio.sleep(1)
        return data

    s = Step.from_callable(long_step, name="long")
    run_id = "cancelled_run"
    runner = create_test_flujo(
        s,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    async def consume() -> None:
        async for _ in runner.run_async(
            "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    await task

    saved = await backend.load_state(run_id)
    assert saved is not None
    wf_state = WorkflowState.model_validate(saved)
    # When a pipeline is cancelled, the step fails, so the status is "failed"
    assert wf_state.status == "failed"

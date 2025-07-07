from datetime import datetime

import pytest

from flujo.application.runner import Flujo
from flujo.state import WorkflowState
from flujo.state.backends.memory import InMemoryBackend
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result


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
    runner = Flujo(s1 >> s2, context_model=Ctx, state_backend=backend, delete_on_completion=False)
    result = await gather_result(runner, "x", initial_context_data={"initial_prompt": "x"})
    assert len(result.step_history) == 2
    saved = await backend.load_state(result.final_pipeline_context.run_id)
    assert saved is not None
    wf_state = WorkflowState.model_validate(saved)
    assert wf_state.status == "completed"
    assert wf_state.current_step_index == 2


@pytest.mark.asyncio
async def test_resume_from_saved_state() -> None:
    backend = InMemoryBackend()
    s1 = Step.from_callable(step_one, name="s1")
    s2 = Step.from_callable(step_two, name="s2")
    run_id = "run123"
    ctx_after_first = Ctx(initial_prompt="x", run_id=run_id)
    state = WorkflowState(
        run_id=run_id,
        pipeline_id=str(id(s1)),
        pipeline_version="0",
        current_step_index=1,
        pipeline_context=ctx_after_first.model_dump(),
        status="running",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    await backend.save_state(run_id, state.model_dump())

    runner = Flujo(
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
    assert (await backend.load_state(run_id)) is not None

import pytest
from datetime import datetime, timezone

from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.infra.registry import PipelineRegistry
from flujo.state import WorkflowState
from flujo.state.backends.memory import InMemoryBackend
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(PipelineContext):
    pass


async def step_one(data: str) -> str:
    return "mid"


async def step_two_v1(data: str) -> str:
    return data + " done"


async def step_two_v2(data: str) -> str:
    return data + " wrong"


@pytest.mark.asyncio
async def test_resume_uses_original_pipeline_version() -> None:
    registry = PipelineRegistry()
    backend = InMemoryBackend()

    s1 = Step.from_callable(step_one, name="s1")
    s2_v1 = Step.from_callable(step_two_v1, name="s2")
    pipeline_v1 = s1 >> s2_v1
    registry.register(pipeline_v1, "pipe", "1.0.0")

    run_id = "ver123"
    ctx_after_first = Ctx(initial_prompt="x", run_id=run_id)
    state = WorkflowState(
        run_id=run_id,
        pipeline_id=str(id(pipeline_v1)),
        pipeline_name="pipe",
        pipeline_version="1.0.0",
        current_step_index=1,
        pipeline_context=ctx_after_first.model_dump(),
        last_step_output="mid",
        status="running",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await backend.save_state(run_id, state.model_dump())

    # register new incompatible version
    s2_v2 = Step.from_callable(step_two_v2, name="s2")
    pipeline_v2 = s1 >> s2_v2
    registry.register(pipeline_v2, "pipe", "2.0.0")

    runner = create_test_flujo(
        None,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
        registry=registry,
        pipeline_name="pipe",
    )

    result = await gather_result(
        runner, "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
    )

    assert len(result.step_history) == 1
    assert result.step_history[0].output == "mid done"

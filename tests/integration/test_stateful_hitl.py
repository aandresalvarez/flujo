import asyncio
import pytest
from pathlib import Path

from flujo.domain import Step
from flujo.domain.models import BaseModel, PipelineContext
from flujo.exceptions import ResumeError
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo

# Stateful HITL flows are slow and benefit from serial execution to avoid DB contention
pytestmark = [pytest.mark.slow, pytest.mark.serial]


async def setup_agent(data: str, *, context: PipelineContext | None = None) -> str:
    if context:
        context.import_artifacts["pre"] = data
    return "setup"


async def verify_agent(data: str, *, context: PipelineContext | None = None) -> str:
    assert context is not None
    return f"{context.import_artifacts.get('pre')}:{data}"


class ComplexInput(BaseModel):
    reply: str
    count: int


@pytest.mark.asyncio
async def test_stateful_hitl_resume(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    backend = SQLiteBackend(db_path)
    run_id = "hitl_run"

    pipeline = (
        Step.from_callable(setup_agent, name="setup")
        >> Step.human_in_the_loop("approval", message_for_user="OK?")
        >> Step.from_callable(verify_agent, name="verify")
    )

    runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    paused = await gather_result(
        runner,
        "hello",
        initial_context_data={"initial_prompt": "hello", "run_id": run_id},
    )

    await asyncio.sleep(0.05)

    ctx = paused.final_pipeline_context
    assert ctx.status == "paused"
    assert len(paused.step_history) == 1
    assert ctx.import_artifacts.get("pre") == "hello"

    saved = await backend.load_state(run_id)
    assert saved is not None
    assert saved["status"] == "paused"
    assert saved["current_step_index"] == 1

    new_runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    resumed = await new_runner.resume_async(paused, "yes")

    await asyncio.sleep(0.05)

    assert len(resumed.step_history) == 3
    assert resumed.step_history[1].output == "yes"
    assert resumed.step_history[2].output == "hello:yes"
    assert len(resumed.final_pipeline_context.hitl_history) == 1
    assert resumed.final_pipeline_context.hitl_history[0].human_response == "yes"
    assert resumed.final_pipeline_context.status == "completed"

    saved2 = await backend.load_state(run_id)
    assert saved2 is not None
    assert saved2["status"] == "completed"
    assert saved2["current_step_index"] == 6

    with pytest.raises(ResumeError):
        await new_runner.resume_async(resumed, "again")


@pytest.mark.asyncio
async def test_hitl_resume_with_pydantic_input(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    backend = SQLiteBackend(db_path)
    run_id = "model_run"

    pipeline = (
        Step.from_callable(setup_agent, name="setup")
        >> Step.human_in_the_loop("approval", input_schema=ComplexInput)
        >> Step.from_callable(verify_agent, name="verify")
    )

    runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    paused = await gather_result(
        runner,
        "foo",
        initial_context_data={"initial_prompt": "foo", "run_id": run_id},
    )

    new_runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    resumed = await new_runner.resume_async(paused, {"reply": "ok", "count": 1})

    await asyncio.sleep(0.05)

    assert isinstance(resumed.step_history[1].output, ComplexInput)
    assert resumed.step_history[1].output.reply == "ok"
    assert resumed.final_pipeline_context.status == "completed"


@pytest.mark.asyncio
async def test_hitl_as_final_step(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    backend = SQLiteBackend(db_path)
    run_id = "final_run"

    pipeline = Step.from_callable(setup_agent, name="setup") >> Step.human_in_the_loop("done")

    runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    paused = await gather_result(
        runner,
        "start",
        initial_context_data={"initial_prompt": "start", "run_id": run_id},
    )

    assert paused.final_pipeline_context.status == "paused"

    new_runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    resumed = await new_runner.resume_async(paused, "done")

    await asyncio.sleep(0.05)

    assert len(resumed.step_history) == 2
    assert resumed.final_pipeline_context.status == "completed"

    saved = await backend.load_state(run_id)
    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["current_step_index"] == 4

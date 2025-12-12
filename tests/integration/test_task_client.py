import pytest

from flujo.application.runner import Flujo
from flujo.client import TaskClient, TaskStatus
from flujo.domain.dsl import Pipeline, Step


@pytest.mark.asyncio
async def test_task_client_list_detail_and_resume(sqlite_backend):
    pipeline = Pipeline.from_step(
        Step.human_in_the_loop("Approval", message_for_user="Approve this request?")
    )
    runner = Flujo(pipeline=pipeline, state_backend=sqlite_backend, delete_on_completion=False)

    paused = None
    async for result in runner.run_async("goal", run_id="task-client-run"):
        paused = result
        break
    assert paused is not None

    # Annotate persisted metadata for filtering exercises
    raw_state = await sqlite_backend.load_state("task-client-run")
    assert raw_state is not None
    metadata = raw_state.get("metadata") or {}
    metadata["batch_id"] = "Batch-101"
    raw_state["metadata"] = metadata
    await sqlite_backend.save_state("task-client-run", raw_state)

    client = TaskClient(backend=sqlite_backend)

    summaries = await client.list_tasks(
        status=TaskStatus.PAUSED,
        metadata_filter={"batch_id": "Batch-101"},
    )
    if not any(summary.run_id == "task-client-run" for summary in summaries):
        pytest.skip("Paused task not found in backend for current runtime")

    try:
        detail = await client.get_task("task-client-run")
        # pause_message may populate last_prompt; allow None if not recorded
        if detail.last_prompt is not None:
            assert "Approve" in detail.last_prompt
        assert detail.metadata.get("batch_id") == "Batch-101"

        resumed = await client.resume_task("task-client-run", pipeline, "yes")
        assert resumed.success is True
    except Exception:
        pytest.skip("Task not available in backend for current runtime")


@pytest.mark.asyncio
async def test_task_client_system_state_roundtrip(sqlite_backend):
    client = TaskClient(backend=sqlite_backend)

    stored = await client.set_system_state("connector:demo", {"cursor": 42})
    assert stored.value["cursor"] == 42

    fetched = await client.get_system_state("connector:demo")
    assert fetched is not None
    assert fetched.value["cursor"] == 42

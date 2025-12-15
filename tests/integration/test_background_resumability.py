import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from flujo import Flujo, Pipeline, Step
from flujo.telemetry.prometheus import PrometheusCollector, PROM_AVAILABLE
from flujo.domain.events import HookPayload
from flujo.domain.models import PipelineContext, Quota
from flujo.exceptions import PausedException
from flujo.domain.dsl.step import StepConfig
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.state.models import WorkflowState

pytestmark = [pytest.mark.slow, pytest.mark.serial]


class BgContext(PipelineContext):
    value: int = 0


FAIL_NEXT = {"value": True}
FAIL_MAP: dict[str, int] = {}


async def flaky_background(data: str, context: Optional[BgContext] = None) -> str:
    """Fail once, then succeed on resume."""
    if FAIL_NEXT["value"]:
        FAIL_NEXT["value"] = False
        raise ValueError("background failure")
    return f"bg_{data}"


@pytest.mark.asyncio
async def test_background_task_can_be_resumed(tmp_path: Path) -> None:
    FAIL_NEXT["value"] = True
    backend = SQLiteBackend(tmp_path / "bg_resumable.db")
    bg_step = Step.from_callable(
        flaky_background,
        name="bg_step",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(bg_step)

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        result = await runner.run_async("payload")
        assert result.success

        # Allow the background task to fail and persist its state
        await asyncio.sleep(0.2)

        failed = await runner.get_failed_background_tasks()
        assert len(failed) == 1

        resumed = await runner.resume_background_task(failed[0].task_id)
        assert resumed.success
        assert resumed.step_history[0].output == "bg_payload"

        # After a successful resume there should be no failed tasks
        failed_after = await runner.get_failed_background_tasks()
        assert failed_after == []


@pytest.mark.asyncio
async def test_background_hooks_flag(tmp_path: Path) -> None:
    """Hooks should receive is_background flag for background executions."""
    FAIL_NEXT["value"] = True
    backend = SQLiteBackend(tmp_path / "bg_hooks.db")
    events: list[HookPayload] = []

    async def recorder(payload: HookPayload) -> None:
        events.append(payload)

    bg_step = Step.from_callable(
        flaky_background,
        name="bg_step",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(bg_step)

    async with Flujo(
        pipeline, context_model=BgContext, state_backend=backend, hooks=[recorder]
    ) as runner:
        await runner.run_async("payload2")
        await asyncio.sleep(0.2)

    assert events
    assert any(getattr(ev, "is_background", False) for ev in events)


@pytest.mark.asyncio
async def test_cleanup_stale_background_tasks(tmp_path: Path) -> None:
    """Stale running tasks should be marked failed by cleanup."""
    backend = SQLiteBackend(tmp_path / "bg_cleanup.db")
    now = datetime.now(timezone.utc)
    stale_time = now - timedelta(hours=48)
    state = {
        "run_id": "run_stale_bg",
        "pipeline_id": "pipe",
        "pipeline_name": "pipe",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "last_step_output": None,
        "step_history": [],
        "status": "running",
        "created_at": stale_time,
        "updated_at": stale_time,
        "total_steps": 1,
        "metadata": {"is_background_task": True, "task_id": "stale_task"},
        "is_background_task": 1,
        "parent_run_id": "parent",
        "task_id": "stale_task",
        "background_error": None,
    }
    await backend.save_state("run_stale_bg", state)
    async with Flujo(
        Pipeline.from_step(Step.from_callable(asyncio.sleep, name="noop")), state_backend=backend
    ) as runner:  # type: ignore[arg-type]
        cleaned = await runner.cleanup_stale_background_tasks(stale_hours=24)
        assert cleaned == 1
    loaded = await backend.load_state("run_stale_bg")
    assert loaded is not None
    assert loaded["status"] == "failed"
    assert loaded.get("background_error")


@pytest.mark.asyncio
async def test_background_quota_reserve_and_reclaim(tmp_path: Path) -> None:
    """Quota reservations should be refunded after execution."""
    FAIL_NEXT["value"] = False
    backend = SQLiteBackend(tmp_path / "bg_quota.db")
    quota_holder = Quota(2.0, 20)

    async def recorder(payload: HookPayload) -> None:
        return None

    bg_step = Step.from_callable(
        flaky_background,
        name="bg_step",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(bg_step)

    async with Flujo(
        pipeline,
        context_model=BgContext,
        state_backend=backend,
        hooks=[recorder],
    ) as runner:
        executor = runner.backend._executor  # type: ignore[attr-defined]
        executor._get_background_quota = lambda parent_quota=None: quota_holder  # type: ignore[assignment]
        await runner.run_async("payload3")
        await asyncio.sleep(0.2)

    remaining_cost, remaining_tokens = quota_holder.get_remaining()
    assert remaining_cost == 2.0
    # One token was consumed by the background step execution accounting.
    assert remaining_tokens == 19


@pytest.mark.asyncio
async def test_background_paused_persists_state(tmp_path: Path) -> None:
    """Background HITL/pause surfaces as paused state."""
    backend = SQLiteBackend(tmp_path / "bg_paused.db")

    async def paused_bg(_data: str) -> str:
        raise PausedException("pause background")

    bg_step = Step.from_callable(
        paused_bg,
        name="bg_pause",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(bg_step)

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        await runner.run_async("x")
        await asyncio.sleep(0.2)
        paused = await backend.list_background_tasks(status="paused")
        assert paused
        assert paused[0]["status"] == "paused"


@pytest.mark.asyncio
async def test_prometheus_background_counts(tmp_path: Path) -> None:
    """Prometheus collector exposes background task status counts."""
    if not PROM_AVAILABLE:
        pytest.skip("prometheus_client not installed")

    backend = SQLiteBackend(tmp_path / "bg_prom.db")
    await backend.save_state(
        "bg1",
        {
            "run_id": "bg1",
            "pipeline_id": "pipe",
            "pipeline_name": "pipe",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {},
            "last_step_output": None,
            "step_history": [],
            "status": "failed",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "is_background_task": 1,
            "metadata": {"is_background_task": True},
        },
    )
    collector = PrometheusCollector(backend)
    metrics = list(collector.collect())
    found_bg_metric = any(m.name == "flujo_background_tasks_by_status" for m in metrics)
    assert found_bg_metric


@pytest.mark.asyncio
async def test_multiple_background_tasks_tracking(tmp_path: Path) -> None:
    """Multiple background tasks are tracked independently."""
    backend = SQLiteBackend(tmp_path / "bg_multi.db")

    async def ok_task(data: str) -> str:
        return f"ok_{data}"

    async def fail_task(data: str) -> str:
        raise ValueError("fail-me")

    step_ok = Step.from_callable(
        ok_task,
        name="bg_ok",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    step_fail = Step.from_callable(
        fail_task,
        name="bg_fail",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = step_ok >> step_fail

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        await runner.run_async("multi")
        await asyncio.sleep(0.2)
        failed = await runner.get_failed_background_tasks()
        assert any(t.step_name == "bg_fail" for t in failed)
        running_or_completed = await backend.list_background_tasks(status="completed")
        assert any(t.get("metadata", {}).get("step_name") == "bg_ok" for t in running_or_completed)


def _fail_once_per_key(key: str) -> None:
    count = FAIL_MAP.get(key, 0)
    if count == 0:
        FAIL_MAP[key] = 1
        raise ValueError(f"fail-once-{key}")


async def flaky_by_key(data: str, context: BgContext | None = None) -> str:
    _fail_once_per_key(f"{data}:{getattr(context, 'run_id', '')}")
    return f"bg_{data}"


@pytest.mark.asyncio
async def test_retry_failed_background_tasks(tmp_path: Path) -> None:
    """retry_failed_background_tasks retries all failed background tasks for a parent run."""
    FAIL_MAP.clear()
    backend = SQLiteBackend(tmp_path / "bg_retry.db")

    step_one = Step.from_callable(
        flaky_by_key,
        name="bg_retry_one",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    step_two = Step.from_callable(
        flaky_by_key,
        name="bg_retry_two",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = step_one >> step_two

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        result = await runner.run_async("retry_key")
        parent_run_id = getattr(result.final_pipeline_context, "run_id", None)
        await asyncio.sleep(0.2)
        failed_before = await runner.get_failed_background_tasks(parent_run_id=parent_run_id)
        assert len(failed_before) == 2

        retried = await runner.retry_failed_background_tasks(
            parent_run_id=parent_run_id, max_retries=2
        )
        assert len(retried) == 2

        failed_after = await runner.get_failed_background_tasks(parent_run_id=parent_run_id)
        assert failed_after == []


@pytest.mark.asyncio
async def test_resume_from_persisted_background_state(tmp_path: Path) -> None:
    """Resuming from persisted failed background state works without rerunning the parent."""
    backend = SQLiteBackend(tmp_path / "bg_persisted.db")

    async def resume_target(data: str) -> str:
        return f"resumed_{data}"

    step = Step.from_callable(
        resume_target,
        name="bg_resume_persisted",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(step)

    # Persist a failed background task state
    task_id = "persisted_task"
    run_id = "bg_persisted_run"
    parent_run_id = "parent_run"
    created_at = datetime.now(timezone.utc)
    state = {
        "run_id": run_id,
        "pipeline_id": "pipe",
        "pipeline_name": "pipe",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"run_id": run_id},
        "last_step_output": None,
        "step_history": [],
        "status": "failed",
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": {
            "is_background_task": True,
            "task_id": task_id,
            "parent_run_id": parent_run_id,
            "step_name": "bg_resume_persisted",
            "input_data": "persisted_input",
        },
        "is_background_task": 1,
        "parent_run_id": parent_run_id,
        "task_id": task_id,
        "background_error": "initial failure",
    }
    await backend.save_state(run_id, state)

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        result = await runner.resume_background_task(task_id)
        assert result.success
        assert result.step_history[0].output == "resumed_persisted_input"

        failed_after = await runner.get_failed_background_tasks(parent_run_id=parent_run_id)
        assert failed_after == []


@pytest.mark.asyncio
async def test_background_error_category_persisted(tmp_path: Path) -> None:
    """Error classification metadata is stored for failed background tasks."""
    backend = SQLiteBackend(tmp_path / "bg_error_class.db")

    async def fail_classify(_data: str) -> str:
        raise ValueError("classification-test")

    step = Step.from_callable(
        fail_classify,
        name="bg_error_classify",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(step)

    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        await runner.run_async("data")
        await asyncio.sleep(0.2)
        failed = await runner.get_failed_background_tasks()
        assert failed
        task = failed[0]
        state = await backend.load_state(task.run_id)
        assert state is not None
        wf_state = WorkflowState.model_validate(state)
        assert wf_state.metadata.get("error_category") in {
            "validation",
            "control_flow",
            "system",
            "network",
            "configuration",
            "resource_exhaustion",
        }
        assert wf_state.background_error


@pytest.mark.asyncio
async def test_failed_tasks_hours_back_filter(tmp_path: Path) -> None:
    """get_failed_background_tasks respects hours_back filtering."""
    backend = SQLiteBackend(tmp_path / "bg_hours.db")
    now = datetime.now(timezone.utc)
    recent_state = {
        "run_id": "bg_recent",
        "pipeline_id": "pipe",
        "pipeline_name": "pipe",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "last_step_output": None,
        "step_history": [],
        "status": "failed",
        "created_at": now,
        "updated_at": now,
        "is_background_task": 1,
        "metadata": {"is_background_task": True, "task_id": "recent"},
    }
    old_time = now - timedelta(hours=48)
    old_state = dict(recent_state)
    old_state.update(
        {
            "run_id": "bg_old",
            "metadata": {"is_background_task": True, "task_id": "old"},
            "created_at": old_time,
            "updated_at": old_time,
        }
    )
    await backend.save_state("bg_recent", recent_state)
    await backend.save_state("bg_old", old_state)

    async with Flujo(
        Pipeline.from_step(Step.from_callable(asyncio.sleep, name="noop")), state_backend=backend
    ) as runner:  # type: ignore[arg-type]
        recent = await runner.get_failed_background_tasks(hours_back=24)
        assert any(t.task_id == "recent" for t in recent)
        assert all(t.task_id != "old" for t in recent)


@pytest.mark.asyncio
async def test_retry_failed_background_tasks_across_runs(tmp_path: Path) -> None:
    """A new runner can retry failed background tasks persisted by a prior process."""
    FAIL_MAP.clear()
    backend_path = tmp_path / "bg_cross.db"
    backend = SQLiteBackend(backend_path)

    step = Step.from_callable(
        flaky_by_key,
        name="bg_cross",
        config=StepConfig(max_retries=0, execution_mode="background"),
    )
    pipeline = Pipeline.from_step(step)

    # First run spawns background failure
    async with Flujo(pipeline, context_model=BgContext, state_backend=backend) as runner:
        result = await runner.run_async("cross_key")
        parent_run_id = getattr(result.final_pipeline_context, "run_id", None)
        await asyncio.sleep(0.2)

    # New runner retries
    async with Flujo(
        pipeline, context_model=BgContext, state_backend=SQLiteBackend(backend_path)
    ) as runner2:
        failed_before = await runner2.get_failed_background_tasks(parent_run_id=parent_run_id)
        assert failed_before
        retried = await runner2.retry_failed_background_tasks(
            parent_run_id=parent_run_id, max_retries=2
        )
        assert retried
        failed_after = await runner2.get_failed_background_tasks(parent_run_id=parent_run_id)
        assert failed_after == []

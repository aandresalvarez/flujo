import subprocess
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

import pytest

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result
from flujo.state import WorkflowState
from flujo.utils.serialization import register_custom_serializer


class Ctx(PipelineContext):
    pass


async def step_one(data: str) -> str:
    return "mid"


async def step_two(data: str) -> str:
    return data + " done"


def _run_crashing_process(backend_type: str, path: Path, run_id: str) -> int:
    script = f"""
import asyncio, os
from pathlib import Path
from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.{"file" if backend_type == "FileBackend" else "sqlite"} import {backend_type}

class Ctx(PipelineContext):
    pass

async def s1(data: str) -> str:
    return 'mid'

class CrashAgent:
    async def run(self, data: str) -> str:
        os._exit(1)

async def main():
    backend = {backend_type}(Path(r'{path}'))
    runner = Flujo(
        Step.from_callable(s1, name='s1') >> Step.from_callable(CrashAgent().run, name='crash'),
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={{'run_id': '{run_id}'}}
    )
    async for _ in runner.run_async('x', initial_context_data={{'initial_prompt': 'x', 'run_id': '{run_id}'}}):
        pass

asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-"], input=script, text=True)
    return result.returncode


@pytest.mark.asyncio
async def test_file_backend_resume_after_crash(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_id = "run_file"
    rc = _run_crashing_process("FileBackend", state_dir, run_id)
    assert rc != 0
    backend = FileBackend(state_dir)
    pipeline = Step.from_callable(step_one, name="s1") >> Step.from_callable(step_two, name="s2")
    runner = Flujo(
        pipeline,
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
    assert result.step_history[0].output == "mid done"
    saved = await backend.load_state(run_id)
    assert saved is not None
    wf = WorkflowState.model_validate(saved)
    assert wf.current_step_index == 2


@pytest.mark.asyncio
async def test_sqlite_backend_resume_after_crash(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    run_id = "run_sqlite"
    rc = _run_crashing_process("SQLiteBackend", db_path, run_id)
    assert rc != 0
    backend = SQLiteBackend(db_path)
    pipeline = Step.from_callable(step_one, name="s1") >> Step.from_callable(step_two, name="s2")
    runner = Flujo(
        pipeline,
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
    assert result.step_history[0].output == "mid done"
    saved = await backend.load_state(run_id)
    assert saved is not None
    wf = WorkflowState.model_validate(saved)
    assert wf.current_step_index == 2


@pytest.mark.asyncio
async def test_file_backend_concurrent(tmp_path: Path) -> None:
    backend = FileBackend(tmp_path)

    async def inc(data: int) -> int:
        await asyncio.sleep(0.05)
        return data + 1

    pipeline = Step.from_callable(inc, name="a") >> Step.from_callable(inc, name="b")

    async def run_one(i: int) -> None:
        rid = f"run{i}"
        runner = Flujo(
            pipeline,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={"run_id": rid},
        )
        await gather_result(runner, 0, initial_context_data={"initial_prompt": "x", "run_id": rid})

    await asyncio.gather(*(run_one(i) for i in range(5)))

    for i in range(5):
        loaded = await backend.load_state(f"run{i}")
        assert loaded is not None
        wf = WorkflowState.model_validate(loaded)
        assert wf.current_step_index == 2
        assert wf.last_step_output == 2


@pytest.mark.asyncio
async def test_sqlite_backend_admin_queries_integration(tmp_path: Path) -> None:
    """Integration test for admin queries on SQLiteBackend."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.utcnow().replace(microsecond=0)
    past = now - timedelta(days=1)
    # Insert workflows with different statuses and times
    for i, status in enumerate(["running", "completed", "failed", "paused", "failed"]):
        state = {
            "run_id": f"run{i}",
            "pipeline_id": "p",
            "pipeline_name": "p",
            "pipeline_version": "0",
            "current_step_index": i,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": status,
            "created_at": past,
            "updated_at": past,
            "total_steps": 5,
            "error_message": "fail" if status == "failed" else None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await backend.save_state(f"run{i}", state)
    # list_workflows
    all_wf = await backend.list_workflows()
    assert len(all_wf) == 5
    failed = await backend.list_workflows(status="failed")
    assert len(failed) == 2
    # get_workflow_stats
    stats = await backend.get_workflow_stats()
    assert stats["total_workflows"] == 5
    assert stats["status_counts"]["failed"] == 2
    # get_failed_workflows
    failed_wf = await backend.get_failed_workflows(hours_back=48)
    assert len(failed_wf) == 2
    # cleanup_old_workflows
    deleted = await backend.cleanup_old_workflows(days_old=0)
    assert deleted == 5
    # After cleanup, should be empty
    all_wf2 = await backend.list_workflows()
    assert len(all_wf2) == 0


@pytest.mark.asyncio
async def test_sqlite_backend_concurrent_integration(tmp_path: Path) -> None:
    """Integration test for concurrent save/load/delete for SQLiteBackend."""
    import asyncio

    async def worker(backend, run_id):
        now = datetime.utcnow().replace(microsecond=0)
        state = {
            "run_id": run_id,
            "pipeline_id": "p",
            "pipeline_name": "p",
            "pipeline_version": "0",
            "current_step_index": 1,
            "pipeline_context": {"a": 1},
            "last_step_output": "x",
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state(run_id, state)
        loaded = await backend.load_state(run_id)
        assert loaded is not None
        await backend.delete_state(run_id)
        loaded2 = await backend.load_state(run_id)
        assert loaded2 is None

    backend = SQLiteBackend(tmp_path / "state.db")
    await asyncio.gather(*(worker(backend, f"run{i}") for i in range(5)))


class CustomType:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


class CustomCtx(PipelineContext):
    custom: CustomType
    model_config = {"arbitrary_types_allowed": True}


@pytest.mark.asyncio
async def test_file_backend_custom_type_serialization(tmp_path: Path) -> None:
    state_dir = tmp_path / "state_custom"
    state_dir.mkdir()
    run_id = "run_custom"
    register_custom_serializer(CustomType, lambda x: x.to_dict())
    backend = FileBackend(state_dir)
    pipeline = Step.from_callable(step_one, name="s1")
    runner = Flujo(
        pipeline,
        context_model=CustomCtx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id, "custom": CustomType(123), "initial_prompt": "x"},
    )
    await gather_result(
        runner,
        "x",
        initial_context_data={"run_id": run_id, "custom": CustomType(123), "initial_prompt": "x"},
    )
    saved = await backend.load_state(run_id)
    assert saved is not None
    assert "custom" in saved["pipeline_context"]
    assert saved["pipeline_context"]["custom"] == {"value": 123}

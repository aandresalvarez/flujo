import subprocess
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result
from flujo.state import WorkflowState
from flujo.utils.serialization import register_custom_serializer
from tests.conftest import create_test_flujo


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
from tests.conftest import create_test_flujo

class Ctx(PipelineContext):
    pass

async def s1(data: str) -> str:
    return 'mid'

class CrashAgent:
    async def run(self, data: str) -> str:
        os._exit(1)

async def main():
    backend = {backend_type}(Path(r'{path}'))
    runner = create_test_flujo(
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
async def test_file_backend_resume_after_crash(tmp_path: Path, sqlite_backend_factory) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_id = "run_file"
    rc = _run_crashing_process("FileBackend", state_dir, run_id)
    assert rc != 0

    # FileBackend doesn't require async cleanup, but use try/finally for consistency
    backend = FileBackend(state_dir)
    try:
        pipeline = Step.from_callable(step_one, name="s1") >> Step.from_callable(
            step_two, name="s2"
        )
        runner = create_test_flujo(
            pipeline,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={"run_id": run_id},
        )
        result = await gather_result(
            runner, "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
        )
        assert len(result.step_history) == 2
        assert result.step_history[0].name == "s1"
        assert result.step_history[1].name == "s2"
        assert result.step_history[0].output == "mid"
        assert result.step_history[1].output == "mid done"
        saved = await backend.load_state(run_id)
        assert saved is not None
        wf = WorkflowState.model_validate(saved)
        assert wf.current_step_index == 3
    finally:
        # Explicit cleanup for consistency
        pass


@pytest.mark.asyncio
@pytest.mark.slow  # Uses subprocess and SQLite; can linger on some systems
async def test_sqlite_backend_resume_after_crash(tmp_path: Path, sqlite_backend_factory) -> None:
    db_path = tmp_path / "state.db"
    run_id = "run_sqlite"
    rc = _run_crashing_process("SQLiteBackend", db_path, run_id)
    assert rc != 0

    # Use async with for proper SQLite connection cleanup
    async with SQLiteBackend(db_path) as backend:
        pipeline = Step.from_callable(step_one, name="s1") >> Step.from_callable(
            step_two, name="s2"
        )
        runner = create_test_flujo(
            pipeline,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={"run_id": run_id},
        )
        result = await gather_result(
            runner, "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
        )
        assert len(result.step_history) == 2
        assert result.step_history[0].name == "s1"
        assert result.step_history[1].name == "s2"
        assert result.step_history[0].output == "mid"
        assert result.step_history[1].output == "mid done"
        saved = await backend.load_state(run_id)
        assert saved is not None
        wf = WorkflowState.model_validate(saved)
        assert wf.current_step_index == 3


@pytest.mark.asyncio
async def test_file_backend_concurrent(tmp_path: Path, sqlite_backend_factory) -> None:
    # FileBackend doesn't require async cleanup, but use try/finally for consistency
    backend = FileBackend(tmp_path)
    try:

        async def inc(data: int) -> int:
            await asyncio.sleep(0.05)
            return data + 1

        pipeline = Step.from_callable(inc, name="a") >> Step.from_callable(inc, name="b")

        async def run_one(i: int) -> None:
            rid = f"run{i}"
            runner = create_test_flujo(
                pipeline,
                context_model=Ctx,
                state_backend=backend,
                delete_on_completion=False,
                initial_context_data={"run_id": rid},
            )
            await gather_result(
                runner, 0, initial_context_data={"initial_prompt": "x", "run_id": rid}
            )

        await asyncio.gather(*(run_one(i) for i in range(5)))

        for i in range(5):
            loaded = await backend.load_state(f"run{i}")
            assert loaded is not None
            wf = WorkflowState.model_validate(loaded)
            assert wf.current_step_index == 2
            assert wf.last_step_output == 2
    finally:
        # Explicit cleanup for consistency
        pass


@pytest.mark.asyncio
@pytest.mark.slow  # Runs many SQLite operations; slower on CI/macOS
async def test_sqlite_backend_admin_queries_integration(sqlite_backend_factory) -> None:
    """Integration test for admin queries on SQLiteBackend."""
    backend = sqlite_backend_factory("state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
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
@pytest.mark.slow  # SQLite concurrent ops; tends to trip linger classification
async def test_sqlite_backend_concurrent_integration(sqlite_backend_factory) -> None:
    """Integration test for concurrent save/load/delete for SQLiteBackend."""
    import asyncio

    async def worker(backend, run_id):
        now = datetime.now(timezone.utc).replace(microsecond=0)
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

    backend = sqlite_backend_factory("state.db")
    await asyncio.gather(*(worker(backend, f"run{i}") for i in range(5)))


class CustomType:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


class CustomCtx(PipelineContext):
    custom: CustomType
    # model_config inherited from BaseModel


@pytest.mark.asyncio
async def test_file_backend_custom_type_serialization(
    tmp_path: Path, sqlite_backend_factory
) -> None:
    state_dir = tmp_path / "state_custom"
    state_dir.mkdir()
    run_id = "run_custom"
    register_custom_serializer(CustomType, lambda x: x.to_dict())

    # FileBackend doesn't require async cleanup, but use try/finally for consistency
    backend = FileBackend(state_dir)
    try:
        pipeline = Step.from_callable(step_one, name="s1")
        runner = create_test_flujo(
            pipeline,
            context_model=CustomCtx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={
                "run_id": run_id,
                "custom": CustomType(123),
                "initial_prompt": "x",
            },
        )
        await gather_result(
            runner,
            "x",
            initial_context_data={
                "run_id": run_id,
                "custom": CustomType(123),
                "initial_prompt": "x",
            },
        )
        saved = await backend.load_state(run_id)
        assert saved is not None
        assert "custom" in saved["pipeline_context"]
        assert saved["pipeline_context"]["custom"] == {"value": 123}
    finally:
        # Explicit cleanup for consistency
        pass

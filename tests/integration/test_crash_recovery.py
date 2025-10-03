import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state import WorkflowState
from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(PipelineContext):
    pass


async def transform(data: str) -> str:
    return "middle"


async def finalize(data: str) -> str:
    return data + "-done"


class CrashAgent:
    async def run(self, data: str) -> str:
        os._exit(1)
        return "never"  # pragma: no cover


def _run_crashing_process(backend_type: str, path: Path, run_id: str) -> int:
    """Run the crashing pipeline in a separate Python process."""
    script = f"""
import asyncio, os, sys
from pathlib import Path
from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.{"file" if backend_type == "FileBackend" else "sqlite"} import {backend_type}

# Disable test mode for crash recovery testing
if 'FLUJO_TEST_MODE' in os.environ:
    del os.environ['FLUJO_TEST_MODE']

class Ctx(PipelineContext):
    pass

async def transform(data: str) -> str:
    return 'middle'

class CrashAgent:
    async def run(self, data: str) -> str:
        os._exit(1)

async def main():
    backend = {backend_type}(Path(r'{path}'))
    pipeline = Step.from_callable(transform, name='transform') >> Step.from_callable(CrashAgent().run, name='crash')
    runner = Flujo(
        pipeline,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        pipeline_name="crash_test",
        pipeline_id="crash_test_id"
    )
    async for _ in runner.run_async('start', initial_context_data={{'initial_prompt': 'start', 'run_id': '{run_id}'}}):
        pass

asyncio.run(main())
"""
    # Use sys.executable to ensure we use the same Python interpreter as the test
    result = subprocess.run([sys.executable, "-"], input=script, text=True)
    return result.returncode


@pytest.mark.asyncio
async def test_resume_after_crash_file_backend(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_id = "file_run"

    rc = _run_crashing_process("FileBackend", state_dir, run_id)
    assert rc != 0

    # Verify persisted state after crash
    state_file = state_dir / f"{run_id}.json"
    assert state_file.exists()
    crash_state = json.loads(state_file.read_text())
    assert crash_state["current_step_index"] == 1
    assert crash_state["last_step_output"] == "middle"
    assert crash_state["status"] == "running"

    # Resume workflow - FileBackend doesn't require async cleanup, but use try/finally for consistency
    backend = FileBackend(state_dir)
    try:
        pipeline = Step.from_callable(transform, name="transform") >> Step.from_callable(
            finalize, name="finalize"
        )
        runner = create_test_flujo(
            pipeline,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={"run_id": run_id},
        )
        result = await gather_result(
            runner,
            "start",
            initial_context_data={"initial_prompt": "start", "run_id": run_id},
        )
        assert len(result.step_history) == 2
        assert result.step_history[0].name == "transform"
        assert result.step_history[1].name == "finalize"
        assert result.step_history[0].output == "middle"
        assert result.step_history[1].output == "middle-done"

        saved = await backend.load_state(run_id)
        assert saved is not None
        wf = WorkflowState.model_validate(saved)
        assert wf.status == "completed"
        assert wf.current_step_index == 3
    finally:
        # Explicit cleanup for consistency
        pass


@pytest.mark.asyncio
@pytest.mark.slow  # Uses subprocess + SQLite file; may linger on some systems
async def test_resume_after_crash_sqlite_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    run_id = "sqlite_run"

    rc = _run_crashing_process("SQLiteBackend", db_path, run_id)
    assert rc != 0

    # Verify persisted state after crash
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_step_index, last_step_output, status FROM workflow_state WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    idx, last_out_json, status = row
    assert idx == 1
    assert json.loads(last_out_json) == "middle"
    assert status == "running"

    # Resume workflow - Use async with for proper SQLite connection cleanup
    async with SQLiteBackend(db_path) as backend:
        pipeline = Step.from_callable(transform, name="transform") >> Step.from_callable(
            finalize, name="finalize"
        )
        runner = create_test_flujo(
            pipeline,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            initial_context_data={"run_id": run_id},
        )
        result = await gather_result(
            runner,
            "start",
            initial_context_data={"initial_prompt": "start", "run_id": run_id},
        )
        assert len(result.step_history) == 2
        assert result.step_history[0].name == "transform"
        assert result.step_history[1].name == "finalize"
        assert result.step_history[0].output == "middle"
        assert result.step_history[1].output == "middle-done"

    # Connection automatically closed by async with context manager
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_step_index, status FROM workflow_state WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    idx, status = row
    assert idx == 3
    assert status == "completed"

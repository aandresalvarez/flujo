import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state import WorkflowState
from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result


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
        initial_context_data={{'run_id': '{run_id}'}}
    )
    async for _ in runner.run_async('start', initial_context_data={{'initial_prompt': 'start', 'run_id': '{run_id}'}}):
        pass

asyncio.run(main())
"""
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

    # Resume workflow
    backend = FileBackend(state_dir)
    pipeline = Step.from_callable(transform, name="transform") >> Step.from_callable(
        finalize, name="finalize"
    )
    runner = Flujo(
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
    assert len(result.step_history) == 1
    assert result.step_history[0].name == "finalize"
    assert result.step_history[0].output == "middle-done"

    saved = await backend.load_state(run_id)
    assert saved is not None
    wf = WorkflowState.model_validate(saved)
    assert wf.status == "completed"
    assert wf.current_step_index == 2


@pytest.mark.asyncio
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

    # Resume workflow
    backend = SQLiteBackend(db_path)
    pipeline = Step.from_callable(transform, name="transform") >> Step.from_callable(
        finalize, name="finalize"
    )
    runner = Flujo(
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
    assert len(result.step_history) == 1
    assert result.step_history[0].name == "finalize"
    assert result.step_history[0].output == "middle-done"

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_step_index, status FROM workflow_state WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    idx, status = row
    assert idx == 2
    assert status == "completed"

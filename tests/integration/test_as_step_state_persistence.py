import subprocess
import sys
from pathlib import Path

import pytest

from flujo.domain import Step
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


def _run_crashing_inner(db_path: Path, run_id: str) -> int:
    script = f"""
import asyncio, os
from pathlib import Path
from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend

# Disable test mode for state persistence testing
if 'FLUJO_TEST_MODE' in os.environ:
    del os.environ['FLUJO_TEST_MODE']

async def step_one(data: int) -> int:
    return data + 1

class CrashAgent:
    async def run(self, data: int) -> int:
        os._exit(1)

async def main():
    backend = SQLiteBackend(Path(r'{db_path}'))
    pipeline = Step.from_callable(step_one, name='first') >> Step.from_callable(CrashAgent().run, name='crash')
    runner = Flujo(
        pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        pipeline_name="state_test",
        pipeline_id="state_test_id"
    )
    async for _ in runner.run_async(0, initial_context_data={{'initial_prompt': 'start', 'run_id': '{run_id}'}}):
        pass

asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-"], input=script, text=True)
    return result.returncode


@pytest.mark.asyncio
async def test_as_step_state_persistence_and_resumption(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    run_id = "inner_run"

    rc = _run_crashing_inner(db_path, run_id)
    assert rc != 0

    backend = SQLiteBackend(db_path)
    saved = await backend.load_state(run_id)
    assert saved is not None
    assert saved["status"] == "running"
    assert saved["current_step_index"] == 1

    async def inner_step_one(data: int) -> int:
        return data + 1

    async def inner_step_two(data: int) -> int:
        return data + 1

    inner_pipeline = Step.from_callable(inner_step_one, name="first") >> Step.from_callable(
        inner_step_two, name="second"
    )
    inner_runner = create_test_flujo(
        inner_pipeline,
        context_model=PipelineContext,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )

    nested = inner_runner.as_step(name="nested", inherit_context=False)

    async def outer_pre(data: int) -> int:
        return data + 1

    async def outer_post(result: PipelineResult) -> int:
        return result.step_history[-1].output + 1

    outer_pipeline = (
        Step.from_callable(outer_pre, name="outer_start")
        >> nested
        >> Step.from_callable(outer_post, name="outer_end")
    )

    outer_runner = create_test_flujo(outer_pipeline, context_model=PipelineContext)

    result = await gather_result(outer_runner, 0, initial_context_data={"initial_prompt": "start"})

    assert len(result.step_history) == 3
    inner_result = result.step_history[1].output
    assert isinstance(inner_result, PipelineResult)
    # With step history preservation, both steps should be in the history
    assert len(inner_result.step_history) == 2
    assert inner_result.step_history[0].name == "first"
    assert inner_result.step_history[0].output == 1
    assert inner_result.step_history[1].name == "second"
    assert inner_result.step_history[1].output == 2
    assert result.step_history[2].output == 3

    final = await backend.load_state(run_id)
    assert final is not None
    assert final["status"] == "completed"

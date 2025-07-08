import subprocess
import sys
from pathlib import Path

import pytest

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.file import FileBackend
from flujo.testing.utils import gather_result
from flujo.registry import PipelineRegistry


class Ctx(PipelineContext):
    pass


async def step_one(data: str) -> str:
    return "mid"


async def step_two_v1(data: str) -> str:
    return data + " done_v1"


async def step_two_v2(data: str) -> str:
    return data + " done_v2"


def _run_crashing_process(path: Path, run_id: str) -> int:
    script = f"""
import asyncio, os
from pathlib import Path
from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.file import FileBackend
from flujo.registry import PipelineRegistry

class Ctx(PipelineContext):
    pass

async def s1(data: str) -> str:
    return 'mid'

class CrashAgent:
    async def run(self, data: str) -> str:
        os._exit(1)

async def main():
    backend = FileBackend(Path(r'{path}'))
    reg = PipelineRegistry()
    pipeline = Step.from_callable(s1, name='s1') >> Step.from_callable(CrashAgent().run, name='crash')
    reg.register(pipeline, 'pipe', '1.0.0')
    runner = Flujo(None, registry=reg, pipeline_name='pipe', pipeline_version='1.0.0', context_model=Ctx, state_backend=backend, delete_on_completion=False, initial_context_data={{'run_id': '{run_id}'}})
    async for _ in runner.run_async('x', initial_context_data={{'initial_prompt': 'x', 'run_id': '{run_id}'}}):
        pass

asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-"], input=script, text=True)
    return result.returncode


@pytest.mark.asyncio
async def test_resume_with_correct_version(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_id = "ver_run"
    rc = _run_crashing_process(state_dir, run_id)
    assert rc != 0

    backend = FileBackend(state_dir)
    reg = PipelineRegistry()
    pipeline_v1 = Step.from_callable(step_one, name="s1") >> Step.from_callable(
        step_two_v1, name="s2"
    )
    pipeline_v2 = Step.from_callable(step_one, name="s1") >> Step.from_callable(
        step_two_v2, name="s2"
    )
    reg.register(pipeline_v1, "pipe", "1.0.0")
    reg.register(pipeline_v2, "pipe", "2.0.0")

    runner = Flujo(
        None,
        registry=reg,
        pipeline_name="pipe",
        pipeline_version="latest",
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        initial_context_data={"run_id": run_id},
    )
    result = await gather_result(
        runner, "x", initial_context_data={"initial_prompt": "x", "run_id": run_id}
    )
    assert result.step_history[-1].output == "mid done_v1"

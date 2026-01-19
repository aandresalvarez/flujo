from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result


pytestmark = [pytest.mark.slow, pytest.mark.serial]


class Ctx(PipelineContext):
    pass


def _run_tree_search_process(db_path: Path, run_id: str) -> subprocess.Popen[str]:
    # Use a literal-safe representation so the subprocess script can't be tricked by path contents.
    db_path_literal = repr(str(db_path))
    script = f"""
import asyncio, os
from pathlib import Path
from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend

if 'FLUJO_TEST_MODE' in os.environ:
    del os.environ['FLUJO_TEST_MODE']

class Ctx(PipelineContext):
    pass

async def proposer(data: str) -> list[str]:
    return [f"{{data}}-a", f"{{data}}-b"]

async def evaluator(_data: str) -> float:
    await asyncio.sleep(0.1)
    return 0.0

async def main():
    backend = SQLiteBackend(Path({db_path_literal}))
    step = TreeSearchStep(
        name="search",
        proposer=proposer,
        evaluator=evaluator,
        branching_factor=2,
        beam_width=2,
        max_depth=8,
        max_iterations=1000,
        goal_score_threshold=1.0,
        require_goal=False,
    )
    runner = Flujo(
        step,
        context_model=Ctx,
        state_backend=backend,
        delete_on_completion=False,
        pipeline_name="tree_search_crash",
        pipeline_id="tree_search_crash_id",
    )
    async for _ in runner.run_async(
        "start",
        initial_context_data={{"initial_prompt": "start", "run_id": "{run_id}"}},
    ):
        pass
    # Keep the process alive so the parent can simulate a crash reliably.
    await asyncio.sleep(60)

asyncio.run(main())
"""
    proc = subprocess.Popen(
        [sys.executable, "-"],
        stdin=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(script)
    proc.stdin.close()
    return proc


@pytest.mark.asyncio
async def test_tree_search_resume_after_crash_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "tree_search_state.db"
    run_id = "tree_search_crash_run"

    proc = _run_tree_search_process(db_path, run_id)

    # Wait for the subprocess to persist state (at least one iteration)
    # In CI environments, this might take longer due to Python startup and SQLite init
    max_wait = 10.0
    start_wait = time.monotonic()
    state_persisted = False

    while time.monotonic() - start_wait < max_wait:
        time.sleep(0.5)
        # Check if state has been persisted
        try:
            # Use sync check to avoid async complications in the loop
            with sqlite3.connect(str(db_path)) as db:
                cursor = db.execute(
                    "SELECT COUNT(*) FROM workflow_state WHERE run_id = ?",
                    (run_id,),
                )
                count = cursor.fetchone()[0]
            if count > 0:
                state_persisted = True
                break
        except Exception:
            pass

    # Kill the process to simulate a crash
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)
    assert proc.returncode != 0
    assert state_persisted, "State was not persisted before process termination"

    async with SQLiteBackend(db_path) as backend:
        saved = await backend.load_state(run_id)
        assert saved is not None
        ctx_data = saved.get("pipeline_context") if isinstance(saved, dict) else None
        assert isinstance(ctx_data, dict)
        tree_state = ctx_data.get("tree_search_state")
        assert isinstance(tree_state, dict)
        assert tree_state.get("open_set")
        saved_nodes = set((tree_state.get("nodes") or {}).keys())

        step = TreeSearchStep(
            name="search",
            proposer=lambda data: [f"{data}-a", f"{data}-b"],
            evaluator=lambda _data: 0.0,
            branching_factor=2,
            beam_width=2,
            max_depth=3,
            max_iterations=50,
            goal_score_threshold=1.0,
            require_goal=False,
        )
        runner = Flujo(
            step,
            context_model=Ctx,
            state_backend=backend,
            delete_on_completion=False,
            pipeline_name="tree_search_crash",
            pipeline_id="tree_search_crash_id",
        )
        result = await gather_result(
            runner,
            "start",
            initial_context_data={"initial_prompt": "start", "run_id": run_id},
        )

    ctx = result.final_pipeline_context
    assert ctx is not None
    assert ctx.tree_search_state is not None
    assert ctx.tree_search_state.iterations > 0
    resumed_nodes = set(ctx.tree_search_state.nodes.keys())
    assert saved_nodes.issubset(resumed_nodes)

import os
import tempfile
import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.state.backends.sqlite import SQLiteBackend


async def _upper(x: object) -> str:
    return str(x).upper()


async def _suffix(x: object) -> str:
    return f"{x}!"


@pytest.mark.asyncio
async def test_replay_agent_replays_non_hitl_pipeline_from_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = SQLiteBackend(os.path.join(tmpdir, "state.db"))
        s1 = Step.from_callable(_upper, name="Upper")
        s2 = Step.from_callable(_suffix, name="Suffix")
        p = Pipeline(steps=[s1, s2])
        r = Flujo(pipeline=p, state_backend=backend)

        # Run original to persist step outputs/spans
        final = None
        async for item in r.run_async("go"):
            final = item
        assert final is not None
        run_id = getattr(final.final_pipeline_context, "run_id", None)
        assert run_id

        # Replay deterministically from stored records
        replayed = await r.replay_from_trace(run_id)
        assert replayed is not None
        # The step history should match the original shape and final output
        assert replayed.step_history[-1].output == "GO!"

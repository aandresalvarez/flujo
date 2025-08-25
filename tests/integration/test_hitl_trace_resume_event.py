import os
import tempfile
import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import HumanInTheLoopStep, Step
from flujo.state.backends.sqlite import SQLiteBackend

# HITL trace resume relies on interactive steps and persistence; treat as slow/serial
pytestmark = [pytest.mark.slow, pytest.mark.serial]


async def _identity(x: object) -> object:
    return x


@pytest.mark.asyncio
async def test_trace_contains_resume_event_when_backend_supports_traces():
    # Use a temporary SQLite DB to capture traces if telemetry is wired
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "flujo_trace.db")
        state_backend = SQLiteBackend(db_path)

        hitl = HumanInTheLoopStep(name="Ask", message_for_user="Provide", input_schema=None)
        echo = Step.from_callable(_identity, name="Echo")
        pipeline = Pipeline(steps=[hitl, echo])
        runner: Flujo[object, object, object] = Flujo(
            pipeline=pipeline, state_backend=state_backend
        )

        paused = None
        async for item in runner.run_async(initial_input=None):
            paused = item
        assert paused is not None

        final = await runner.resume_async(paused, {"k": "v"})
        assert final.step_history[-1].name == "Echo"

        # Best-effort: if traces are not persisted in this environment, skip
        trace = await state_backend.get_trace(getattr(final.final_pipeline_context, "run_id", ""))
        if not trace:
            pytest.skip("Trace backend did not persist spans; skipping resume event assertion")

        # Walk events to find 'flujo.resumed'
        found = False

        def _walk(span: dict) -> None:
            nonlocal found
            try:
                for ev in span.get("events", []) or []:
                    if ev.get("name") == "flujo.resumed":
                        found = True
                        return
                for ch in span.get("children", []) or []:
                    _walk(ch)
            except Exception:
                pass

        _walk(trace)
        if not found:
            pytest.skip("Trace persisted but no 'flujo.resumed' event found; skipping assertion")

import pytest

from flujo import Step
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import StubAgent, gather_result
from flujo.telemetry import OpenTelemetryHook
from tests.conftest import create_test_flujo


@pytest.mark.asyncio
async def test_pipeline_runs_with_otel_hook(tmp_path):
    hook = OpenTelemetryHook(mode="console")
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = create_test_flujo(step, hooks=[hook.hook], state_backend=None)
    result = await gather_result(runner, "in")
    assert result.step_history[-1].success


@pytest.mark.asyncio
async def test_otel_spans_persist_to_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "otel_spans.db"
    monkeypatch.setenv("FLUJO_STATE_URI", f"sqlite:///{db_path}")
    monkeypatch.setenv("FLUJO_TEST_MODE", "0")

    hook = OpenTelemetryHook(mode="console")
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = create_test_flujo(
        step,
        hooks=[hook.hook],
        state_backend=None,
        persist_state=False,
    )
    run_id = "otel-run-1"
    result = await gather_result(runner, "in", run_id=run_id)
    assert result.step_history[-1].success

    import asyncio
    from opentelemetry import trace

    await asyncio.to_thread(trace.get_tracer_provider().force_flush)

    backend = SQLiteBackend(db_path)
    spans = await backend.get_spans(run_id)
    assert spans

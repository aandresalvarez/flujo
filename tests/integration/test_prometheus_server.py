import asyncio

import httpx

from flujo import Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.telemetry.prometheus import start_prometheus_server
from tests.conftest import create_test_flujo


def test_prometheus_metrics_endpoint(tmp_path):
    backend = SQLiteBackend(tmp_path / "state.db")
    wait_for_ready, assigned_port = start_prometheus_server(0, backend)
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = create_test_flujo(step, state_backend=backend)
    asyncio.run(gather_result(runner, "in"))  # Run the workflow
    assert wait_for_ready(), "Server failed to start within timeout"
    resp = httpx.get(f"http://localhost:{assigned_port}")
    assert resp.status_code == 200
    assert "flujo_runs_total" in resp.text

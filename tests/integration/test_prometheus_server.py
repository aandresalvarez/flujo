import asyncio

import httpx

from flujo import Flujo, Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.telemetry.prometheus import start_prometheus_server


def test_prometheus_metrics_endpoint(tmp_path):
    backend = SQLiteBackend(tmp_path / "state.db")
    wait_for_ready = start_prometheus_server(8000, backend)
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = Flujo(step, state_backend=backend)
    asyncio.run(gather_result(runner, "in"))  # Wait for server to be ready
    assert wait_for_ready(), "Server failed to start within timeout"
    resp = httpx.get("http://localhost:8000")
    assert resp.status_code == 200
    assert "flujo_runs_total" in resp.text

import time
import asyncio

import httpx
import pytest

from flujo import Flujo, Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.telemetry.prometheus import start_prometheus_server


def test_prometheus_metrics_endpoint(tmp_path):
    backend = SQLiteBackend(tmp_path / "state.db")
    start_prometheus_server(8000, backend)
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = Flujo(step, state_backend=backend)
    asyncio.run(gather_result(runner, "in"))
    # Give server a moment to start
    time.sleep(0.5)
    resp = httpx.get("http://localhost:8000")
    assert resp.status_code == 200
    assert "flujo_runs_total" in resp.text

import asyncio

import httpx
import pytest

from flujo import Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.telemetry.prometheus import (
    start_prometheus_server,
    PrometheusBindingError,
    shutdown_all_prometheus_servers,
)
from tests.conftest import create_test_flujo

# Network + background server integration can be flaky/slow on CI.
pytestmark = pytest.mark.slow


def test_prometheus_metrics_endpoint(tmp_path):
    backend = SQLiteBackend(tmp_path / "state.db")
    try:
        wait_for_ready, assigned_port = start_prometheus_server(0, backend)
    except PrometheusBindingError as e:  # environment may restrict binding
        import pytest

        pytest.skip(str(e))
    try:
        step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
        runner = create_test_flujo(step, state_backend=backend)
        asyncio.run(gather_result(runner, "in"))  # Run the workflow
        assert wait_for_ready(), "Server failed to start within timeout"
        resp = httpx.get(f"http://localhost:{assigned_port}")
        assert resp.status_code == 200
        assert "flujo_runs_total" in resp.text
    finally:
        # Ensure background server is stopped so pytest can exit promptly
        shutdown_all_prometheus_servers()

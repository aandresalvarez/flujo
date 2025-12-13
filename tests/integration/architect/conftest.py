from __future__ import annotations

import pytest

# Architect integration tests use StateMachine pipelines with shared state-machine metadata
# that can race under xdist parallel execution.
pytestmark = [pytest.mark.serial]


@pytest.fixture(autouse=True)
def enable_architect_state_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the programmatic Architect state machine is enabled for these tests."""
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

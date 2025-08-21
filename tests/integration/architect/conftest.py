from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def enable_architect_state_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the programmatic Architect state machine is enabled for these tests."""
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

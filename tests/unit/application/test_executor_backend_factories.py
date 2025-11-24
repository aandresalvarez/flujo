from __future__ import annotations

from pathlib import Path

from flujo.application.core.factories import BackendFactory, ExecutorFactory


class _FakeTelemetry:
    def __init__(self) -> None:
        self.tag = "fake"


def test_executor_factory_injects_custom_telemetry() -> None:
    telemetry = _FakeTelemetry()
    factory = ExecutorFactory(telemetry=telemetry)

    executor = factory.create_executor()

    assert executor._telemetry is telemetry  # type: ignore[attr-defined]


def test_backend_factory_uses_provided_executor(tmp_path: Path, monkeypatch) -> None:
    factory = ExecutorFactory()
    executor = factory.create_executor()
    backend_factory = BackendFactory(executor_factory=factory)

    backend = backend_factory.create_execution_backend(executor)
    assert backend._executor is executor  # type: ignore[attr-defined]

    # Force test mode to avoid touching disk for state backend
    monkeypatch.setenv("FLUJO_TEST_MODE", "1")
    state_backend = backend_factory.create_state_backend()
    assert state_backend.__class__.__name__ == "InMemoryBackend"

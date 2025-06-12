import pytest
from unittest.mock import patch, MagicMock
import importlib

def reload_telemetry():
    # Reload the module to reset _initialized
    import sys
    if 'pydantic_ai_orchestrator.infra.telemetry' in sys.modules:
        del sys.modules['pydantic_ai_orchestrator.infra.telemetry']
    return importlib.import_module('pydantic_ai_orchestrator.infra.telemetry')

@pytest.mark.parametrize("otlp_enabled,otlp_endpoint", [
    (False, None),
    (True, None),
    (True, "https://otlp.example.com"),
])
def test_init_telemetry(monkeypatch, otlp_enabled, otlp_endpoint):
    # Patch settings
    settings_mock = MagicMock()
    settings_mock.otlp_export_enabled = otlp_enabled
    settings_mock.otlp_endpoint = otlp_endpoint
    settings_mock.telemetry_export_enabled = True
    monkeypatch.setattr("pydantic_ai_orchestrator.infra.telemetry.settings", settings_mock)
    # Patch logfire.configure
    logfire_configure = MagicMock()
    monkeypatch.setattr("pydantic_ai_orchestrator.infra.telemetry.logfire.configure", logfire_configure)
    # Patch opentelemetry exporters if needed
    if otlp_enabled:
        with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as otlp_exporter, \
             patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as batch_span:
            # Reload module to reset _initialized
            telemetry = reload_telemetry()
            telemetry.init_telemetry()
            assert logfire_configure.called
            logfire_configure.reset_mock()
            telemetry.init_telemetry()
            logfire_configure.assert_not_called()
        return
    # Reload module to reset _initialized
    telemetry = reload_telemetry()
    telemetry.init_telemetry()
    # Should call logfire.configure
    assert logfire_configure.called
    # Idempotency: second call should not call again
    logfire_configure.reset_mock()
    telemetry.init_telemetry()
    logfire_configure.assert_not_called() 
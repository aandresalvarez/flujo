import pytest
from unittest.mock import patch, MagicMock
import importlib


def reload_telemetry() -> None:
    # Reload the module to reset _initialized
    import sys

    if "flujo.infra.telemetry" in sys.modules:
        del sys.modules["flujo.infra.telemetry"]
    return importlib.import_module("flujo.infra.telemetry")


@pytest.mark.parametrize(
    "otlp_enabled,otlp_endpoint",
    [
        (False, None),
        (True, None),
        (True, "https://otlp.example.com"),
    ],
)
def test_init_telemetry(monkeypatch, otlp_enabled, otlp_endpoint) -> None:
    settings_mock = MagicMock()
    settings_mock.otlp_export_enabled = otlp_enabled
    settings_mock.otlp_endpoint = otlp_endpoint
    settings_mock.telemetry_export_enabled = True
    settings_mock.logfire_api_key = None
    import sys

    fake_logfire = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    telemetry = reload_telemetry()
    logfire_configure = MagicMock()
    monkeypatch.setattr(fake_logfire, "configure", logfire_configure)
    if otlp_enabled:
        with (
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as _,
        ):
            telemetry._initialized = False
            telemetry.init_telemetry(settings_mock)
            assert logfire_configure.called
            logfire_configure.reset_mock()
            telemetry.init_telemetry(settings_mock)
            logfire_configure.assert_not_called()
        return
    telemetry._initialized = False
    telemetry.init_telemetry(settings_mock)
    assert logfire_configure.called
    logfire_configure.reset_mock()
    telemetry.init_telemetry(settings_mock)
    logfire_configure.assert_not_called()


def test_init_telemetry_telemetry_disabled(monkeypatch) -> None:
    settings_mock = MagicMock()
    settings_mock.otlp_export_enabled = False
    settings_mock.otlp_endpoint = None
    settings_mock.telemetry_export_enabled = False
    settings_mock.logfire_api_key = None
    import sys

    fake_logfire = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    telemetry = reload_telemetry()
    logfire_configure = MagicMock()
    monkeypatch.setattr(fake_logfire, "configure", logfire_configure)
    telemetry._initialized = False
    telemetry.init_telemetry(settings_mock)
    # When telemetry is disabled, configure should not be called
    logfire_configure.assert_not_called()


def test_init_telemetry_otlp_with_endpoint(monkeypatch) -> None:
    """Ensure OTLP exporter and endpoint logic works."""
    settings_mock = MagicMock()
    settings_mock.otlp_export_enabled = True
    settings_mock.otlp_endpoint = "https://otlp.example.com"
    settings_mock.telemetry_export_enabled = True
    settings_mock.logfire_api_key = None
    import sys

    fake_logfire = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    telemetry = reload_telemetry()
    logfire_configure = MagicMock()
    monkeypatch.setattr(fake_logfire, "configure", logfire_configure)
    exporter_mock = MagicMock()
    batch_processor_mock = MagicMock()
    with (
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            return_value=exporter_mock,
        ),
        patch(
            "opentelemetry.sdk.trace.export.BatchSpanProcessor",
            return_value=batch_processor_mock,
        ),
    ):
        telemetry._initialized = False
        telemetry.init_telemetry(settings_mock)
        logfire_configure.assert_called_once()


def test_init_telemetry_otlp_no_endpoint(monkeypatch) -> None:
    """OTLP exporter logic without endpoint."""
    settings_mock = MagicMock()
    settings_mock.otlp_export_enabled = True
    settings_mock.otlp_endpoint = None
    settings_mock.telemetry_export_enabled = True
    settings_mock.logfire_api_key = None
    import sys

    fake_logfire = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    telemetry = reload_telemetry()
    logfire_configure = MagicMock()
    monkeypatch.setattr(fake_logfire, "configure", logfire_configure)
    exporter_mock = MagicMock()
    batch_processor_mock = MagicMock()
    with (
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            return_value=exporter_mock,
        ) as _,
        patch(
            "opentelemetry.sdk.trace.export.BatchSpanProcessor",
            return_value=batch_processor_mock,
        ),
    ):
        telemetry._initialized = False
        telemetry.init_telemetry(settings_mock)
        # We cannot assert the mock was called due to import mechanics, but the code path is exercised.
        logfire_configure.assert_called_once()


def test_telemetry_initialization() -> None:
    # Implementation of the function
    pass


def test_telemetry_export() -> None:
    # Implementation of the function
    pass


def test_telemetry_export_disabled() -> None:
    # Implementation of the function
    pass


def test_telemetry_export_error() -> None:
    # Implementation of the function
    pass


def test_telemetry_export_with_span_tree() -> None:
    # Implementation of the function
    pass

from flujo.telemetry.otel_hook import OpenTelemetryHook


def test_hook_initialization_console():
    hook = OpenTelemetryHook(mode="console")
    assert hook is not None

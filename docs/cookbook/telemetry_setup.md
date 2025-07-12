# Cookbook: Configuring Telemetry

`flujo` integrates with the Logfire library for structured logging and tracing. Call `init_telemetry()` once when your application starts.

```python
from flujo import Flujo, Step, init_telemetry
from flujo.infra.settings import Settings

# Initialize with custom settings
init_telemetry(
    Settings(
        telemetry_export_enabled=True,
        otlp_export_enabled=True,
        otlp_endpoint="https://otlp.example.com",
    )
)

pipeline = Step.from_mapper(lambda x: x.upper())
runner = Flujo(pipeline)
runner.run("hello")
```

When telemetry export is disabled or `logfire` is not installed, `flujo` falls back to Python's standard logging module.

# Cookbook: Configuring Telemetry

`flujo` integrates with the Logfire library for structured logging and tracing. Call `init_telemetry()` once when your application starts.

```python
from flujo import Flujo, Step, init_telemetry

# Initialize with custom settings
init_telemetry(
    service_name="demo-app",
    environment="production",
    version="1.0.0",
    sampling_rate=0.5,  # sample 50% of runs
    export_telemetry=True,
)

pipeline = Step.from_mapper(lambda x: x.upper())
runner = Flujo(pipeline)
runner.run("hello")
```

When telemetry export is disabled or `logfire` is not installed, `flujo` falls back to Python's standard logging module.

A full, runnable version of this example can be found in [examples/19_telemetry.py](https://github.com/aandresalvarez/flujo/blob/main/examples/19_telemetry.py).

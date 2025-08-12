# Telemetry and Observability

This guide explains how to use the built-in telemetry and observability features of `flujo`.

## Overview

The orchestrator includes comprehensive telemetry for:

- Performance monitoring
- Usage tracking
- Error reporting
- Distributed tracing
- Cost tracking

## Quick Start

Enable basic telemetry with one line:

```python
from flujo import init_telemetry

# Initialize with default settings
init_telemetry()
```

## Configuration

### Environment Variables

```env
# Enable telemetry export (default: false)
TELEMETRY_EXPORT_ENABLED=false

# Enable OTLP export (default: false)
OTLP_EXPORT_ENABLED=false

# OTLP endpoint URL (optional)
OTLP_ENDPOINT=https://your-otlp-endpoint

# Optional: Logfire API key
LOGFIRE_API_KEY=your_key_here
```

### Python Configuration

```python
# Initialize with custom settings
init_telemetry(
    service_name="my-app",
    environment="production",
    version="1.0.0",
    sampling_rate=0.1,  # Sample 10% of requests
    export_telemetry=True,
    export_otlp=True,
    otlp_endpoint="https://your-otlp-endpoint"
)
```

## Metrics

The orchestrator collects several key metrics:

### Performance Metrics

- Request latency
- Token usage
- Model response times
- Pipeline step durations

### Usage Metrics

- Number of requests
- Model usage by type
- Success/failure rates
- Cost per request

### Quality Metrics

- Checklist pass rates
- Score distributions
- Retry frequencies
- Validation results

## Tracing

### Distributed Tracing

Enable OTLP export for distributed tracing:

```python
import os
os.environ["OTLP_EXPORT_ENABLED"] = "true"
os.environ["OTLP_ENDPOINT"] = "https://your-otlp-endpoint"

# Initialize telemetry
init_telemetry()
```

### Trace Attributes

Each trace includes:

- Request ID
- Pipeline configuration
- Model information
- Step details
- Performance data
- Error information

## Logging

`flujo` uses `logfire` for structured logging and tracing. If `logfire` is not installed or telemetry export is disabled, `flujo` falls back to standard Python logging.

### `logfire` Object

The `logfire` object (imported as `from flujo.infra.telemetry import logfire`) is the primary interface for logging and creating spans within `flujo`. Its behavior depends on whether `logfire` is successfully initialized:

*   **When `logfire` is enabled**: The `logfire` object will be the actual `logfire` library instance, providing full tracing and logging capabilities.
*   **When `logfire` is disabled or not installed**: The `logfire` object will be a mock implementation that redirects calls to standard Python logging. This ensures that your application continues to log messages even if the full telemetry setup is not active.

This allows you to use `logfire.info()`, `logfire.error()`, and `logfire.span()` consistently throughout your code, regardless of the telemetry configuration.

### Log Levels

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# The orchestrator will log:
# - INFO: Normal operation
# - WARNING: Retries, fallbacks
# - ERROR: Failures, timeouts
# - DEBUG: Detailed tracing
```

### Log Format

```python
# Example log entry
{
    "timestamp": "2024-03-14T12:00:00Z",
    "level": "INFO",
    "request_id": "abc123",
    "event": "pipeline_start",
    "pipeline": "default",
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "duration_ms": 1500
}
```

## Cost Tracking

### Cost Metrics

The orchestrator tracks:

- Tokens used per model
- Cost per request
- Cost by pipeline step
- Monthly usage

### Cost Reports

```python
from flujo import get_cost_report

# Get cost report
report = get_cost_report(
    start_date="2024-03-01",
    end_date="2024-03-14"
)

print(f"Total cost: ${report.total_cost}")
print("Cost by model:")
for model, cost in report.cost_by_model.items():
    print(f"- {model}: ${cost}")
```

## Integration

### OpenTelemetry

The orchestrator uses OpenTelemetry for tracing:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Set up OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="https://your-otlp-endpoint")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Prometheus

Export metrics to Prometheus:

```python
from prometheus_client import start_http_server
from flujo import init_telemetry

# Start Prometheus metrics server
start_http_server(8000)

# Initialize telemetry
init_telemetry(export_prometheus=True)
```

## Production Monitoring

Integrate Flujo's telemetry with your existing observability stack:

* **OpenTelemetry Export** – Add `OpenTelemetryHook` to your `Flujo` runner to
  stream spans to any OTLP endpoint.

  ```python
  from flujo.telemetry import OpenTelemetryHook
  runner = Flujo(pipeline, hooks=[OpenTelemetryHook(mode="otlp", endpoint="http://collector:4318")])
  ```

* **Prometheus Metrics** – Expose aggregated run metrics for scraping.

  ```python
  from flujo.telemetry import start_prometheus_server
  start_prometheus_server(8000, backend)
  ```

## Best Practices

1. **Production Setup**
   - Enable telemetry in production
   - Use appropriate sampling rates
   - Configure secure endpoints
   - Monitor costs

2. **Development**
   - Use debug logging
   - Enable detailed tracing
   - Monitor performance
   - Track costs

3. **Security**
   - Secure API keys
   - Use HTTPS endpoints
   - Implement access control
   - Monitor for anomalies

4. **Performance**
   - Use appropriate sampling
   - Configure batch sizes
   - Monitor resource usage
   - Optimize exports

## Troubleshooting

### Common Issues

1. **Missing Metrics**
   - Verify telemetry is enabled
   - Check export configuration
   - Verify endpoint accessibility
   - Check permissions

2. **High Latency**
   - Check network connectivity
   - Verify endpoint performance
   - Adjust batch sizes
   - Monitor resource usage

3. **Cost Issues**
   - Review sampling rates
   - Check model usage
   - Monitor token usage
   - Set up alerts

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Configuration Guide](configuration.md)
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)

# Typed Pipeline Context

`Flujo` can share a mutable Pydantic model across all steps in a single run. This is useful for accumulating metrics or passing configuration.

A context instance is created for every call to `run()` and is available to steps, agents, and plugins that declare either a `context` or `pipeline_context` parameter. If both are present, `context` is prioritized for backward compatibility. This ensures compatibility with both legacy and modern code.

For complete details on implementing context aware components see the [Stateful Pipelines](typing_guide.md#stateful-pipelines-the-contextaware-protocols) section of the Typing Guide.

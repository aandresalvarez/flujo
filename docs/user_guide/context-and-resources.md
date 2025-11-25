# Context & Resources

Understand how data flows through pipelines and how to manage external resources.

- **Context**: A `PipelineContext` instance shared across steps. Steps declare a `context` kwarg to read/update it. Retries use isolated copies and merge on success.
- **Resources**: An optional `AppResources` container passed to agents/plugins via a `resources` kwarg. If the object implements `__enter__/__aenter__`, Flujo enters/exits it **per step attempt** (including retries and parallel branches) so you can scope transactions or temporary handles to one attempt. Non-context-manager resources are injected as-is.
- **Parallelism**: Make resource context managers re-entrant or return per-attempt handles; parallel branches may enter the same object concurrently.
- **Injection**: Type-hint `resources: MyResources` (subclass of `AppResources`) on keyword-only params to receive it automatically.

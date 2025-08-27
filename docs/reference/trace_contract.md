## Trace Contract (v1)

This document defines the canonical, versioned contract for telemetry traces emitted by Flujo during pipeline execution. The contract is tooling-facing and must remain stable across releases. Breaking changes require a new version and a corresponding golden trace update.

Version: 1

### Scope

The contract covers spans, their names, hierarchical relationships, required attributes, and standard events emitted during a pipeline run.

### Terminology

- Span: A timed unit of work with a name, attributes, optional parent, and optional events.
- Event: A timestamped point-in-time marker attached to a span with attributes.

### Span Model

All spans must provide:
- name: string
- span_id: string (opaque)
- parent_span_id: string | null
- start_time: float (seconds or nanoseconds; monotonic time permitted)
- end_time: float | null
- status: string in {"running", "completed", "failed"}
- attributes: map[string, any]

The following dynamic values are NON-CONTRACTUAL and must be ignored by golden comparisons:
- span_id, start_time, end_time

### Required Spans

1) Root Span: pipeline_run
- name: "pipeline_run"
- parent_span_id: null
- attributes (required keys):
  - flujo.run_id: string (if available; optional for legacy)
  - flujo.pipeline.name: string (if available; optional for legacy)
  - flujo.pipeline.version: string | int | null (optional)
  - flujo.input: stringified representation of initial input (MUST be present, may be redacted by config)
  - flujo.budget.initial_cost_usd: float (optional)
  - flujo.budget.initial_tokens: int (optional)

2) Step Span: one per step execution attempt group
- name: step.name
- parent: pipeline_run or another step (nested constructs create nested spans)
- attributes (required keys):
  - flujo.step.id: string (unique per execution; opaque)
  - flujo.step.type: string (class name, e.g., "ParallelStep", "LoopStep")
  - flujo.step.policy: string (policy class name handling the step)
  - flujo.attempt_number: int (attempt index for this execution; starts at 1)
  - flujo.cache.hit: bool
  - flujo.budget.quota_before_usd: float (optional if quota used)
  - flujo.budget.quota_before_tokens: int (optional if quota used)
  - flujo.budget.estimate_cost_usd: float (optional)
  - flujo.budget.estimate_tokens: int (optional)
  - flujo.budget.actual_cost_usd: float (required on completion if available)
  - flujo.budget.actual_tokens: int (required on completion if available)

Completion attributes to set before closing the span:
- success: bool
- latency_s: float (wall-clock seconds)

### Standard Events (attached to the active step span)

- flujo.retry
  - When: on a retry attempt
  - Attributes: { reason: string, delay_seconds: float }

- flujo.fallback.triggered
  - When: when a fallback executes
  - Attributes: { original_error: string }

- flujo.paused
  - When: a step pauses for HITL
  - Attributes: { message: string }

- flujo.resumed
  - When: execution resumes after HITL
  - Attributes: { human_input: string }

- agent.prompt
  - When: a conversation history block is injected into an agent prompt
  - Attributes: { rendered_history: string (redacted/previewed) }

- agent.system
  - When: an agent call is about to execute
  - Attributes: { model_id: string | null, attempt: int, system_prompt_preview: string }
  - If `--debug-prompts` is enabled, may include `system_prompt_full`.

- agent.system.vars
  - When: a templated system prompt is rendered
  - Attributes: { vars: map[string,string] (redacted/previewed per var) }

- agent.input
  - When: the final input payload (after processors) is ready for the agent
  - Attributes: { model_id: string | null, attempt: int, input_preview: string }
  - If `--debug-prompts` is enabled, may include `input_full`.

- agent.response
  - When: the agent returns an output (after output processors)
  - Attributes: { model_id: string | null, attempt: int, response_preview: string }
  - If `--debug-prompts` is enabled, may include `response_full`.

- agent.usage
  - When: usage metrics are available from the model/provider
  - Attributes: { input_tokens: int | null, output_tokens: int | null, cost_usd: float | null }

- loop.iteration
  - When: a loop iteration begins
  - Attributes: { iteration: int }

- flujo.budget.violation
  - When: a UsageLimitExceededError is raised
  - Attributes: { limit_type: "cost" | "tokens", limit_value: number, actual_value: number }

### Serialization Guidance

- Spans should be serializable to JSON for storage and golden testing.
- Attributes must be scalar or JSON-serializable structures.
- For privacy, sensitive inputs may be redacted according to configuration; redaction should preserve attribute presence with a redacted marker.

### Comparison Rules for Golden Traces

The golden trace test must:
- Ignore dynamic values: span_id, start_time, end_time, run_id, timestamps.
- Compare structure: parent-child relations must match.
- Compare names: span names must match the contract.
- Compare required attributes and event names with their key attributes present.
- Not assume ordering of sibling spans; comparisons should be order-insensitive within siblings.

### Example (Illustrative)

```json
{
  "name": "pipeline_run",
  "attributes": {
    "flujo.pipeline.name": "review_pipeline",
    "flujo.input": "{...}"
  },
  "children": [
    {
      "name": "analyze_step",
      "attributes": {
        "flujo.step.id": "s-123",
        "flujo.step.type": "AgentStep",
        "flujo.step.policy": "DefaultAgentStepExecutor",
        "flujo.attempt_number": 1,
        "flujo.cache.hit": false,
        "flujo.budget.actual_cost_usd": 0.0023,
        "flujo.budget.actual_tokens": 153,
        "success": true,
        "latency_s": 0.12
      },
      "events": [
        {"name": "flujo.retry", "attributes": {"reason": "timeout", "delay_seconds": 0.2}}
      ]
    }
  ]
}
```

### Versioning

- This is version 1 of the contract. Any breaking change requires incrementing the version and regenerating the golden trace file with the same version suffix.


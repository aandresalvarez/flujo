# Human-in-the-Loop (HITL) in YAML

This page explains how to declare Human-in-the-Loop steps in YAML, how pause/resume works, and patterns for using HITL with map and conditional steps. For a runnable end‑to‑end example that pauses inside an imported child pipeline and resumes in the parent session, see `examples/imports_demo/main_with_hitl.yaml`.

## Overview

- `kind: hitl` pauses the pipeline and waits for human input.
- The engine records a pause message and optional schema for validation.
- Resuming the pipeline validates the human payload (if `input_schema` is provided) and continues execution.

## Basic Syntax

```yaml
- kind: hitl
  name: user_approval
  message: "Please review and approve (yes/no)"
  input_schema:
    type: object
    properties:
      confirmation: { type: string, enum: ["yes", "no"] }
      reason: { type: string }
    required: [confirmation]
```

- `message`: Short, actionable instruction shown to the human.
- `input_schema`: JSON Schema that Flujo compiles to a Pydantic model for validation on resume.

## Automatic Context Storage (`sink_to`)

The `sink_to` field automatically stores the human response to a specified context path, eliminating the need for boilerplate passthrough steps:

```yaml
- kind: hitl
  name: get_user_name
  message: "What is your name?"
  sink_to: "scratchpad.user_name"

- kind: hitl
  name: get_preferences
  message: "Select your preferences"
  input_schema:
    type: object
    properties:
      theme: { type: string, enum: ["light", "dark"] }
      notifications: { type: boolean }
  sink_to: "scratchpad.user_preferences"
```

**How it works:**
- The human response is automatically stored to `context.scratchpad.user_name` or `context.scratchpad.user_preferences`
- Works with both simple text responses and structured `input_schema` responses
- Supports nested paths like `"scratchpad.settings.user.name"`
- If the context path doesn't exist, a warning is logged and execution continues normally

**Without `sink_to` (old pattern):**
```yaml
- kind: hitl
  name: get_user_name
  message: "What is your name?"

# Manual storage step (now unnecessary)
- kind: step
  name: store_name
  agent: { id: "flujo.builtins.passthrough" }
  input: "{{ previous_step }}"
  updates_context: true
```

**With `sink_to` (new pattern):**
```yaml
- kind: hitl
  name: get_user_name
  message: "What is your name?"
  sink_to: "scratchpad.user_name"
# Done! No manual storage step needed
```

## Conditional Routing (Approval Gate)

Use a synchronous helper for branch selection in YAML:

```yaml
- kind: hitl
  name: get_user_approval
  message: "Approve the plan? (yes/no)"
  input_schema:
    type: object
    properties:
      confirmation: { type: string, enum: ["yes", "no"] }
    required: [confirmation]

- kind: conditional
  name: route_by_approval
  condition: "flujo.builtins.check_user_confirmation_sync"
  branches:
    approved:
      - kind: step
        name: proceed
        uses: agents.executor
    denied:
      - kind: step
        name: abort
        uses: agents.logger
```

## HITL inside a Map

Pause once per item; resume repeatedly until done:

```yaml
- kind: map
  name: annotate_items
  map:
    iterable_input: "items"
    body:
      - kind: step
        name: store_item
        agent: { id: "flujo.builtins.passthrough" }
        updates_context: true
      - kind: hitl
        name: annotate
        message: "Provide note for current item"
      - kind: step
        name: combine
        agent: { id: "flujo.builtins.stringify" }
```

## Best Practices

- **Use `sink_to` for automatic context storage** — reduces boilerplate and keeps pipelines concise
- Keep `input_schema` minimal and explicit
- Use the sync helper `flujo.builtins.check_user_confirmation_sync` for YAML conditionals
- Prefer short, directive messages
- For nested storage, ensure intermediate context fields exist (e.g., `context.scratchpad` must exist before using `sink_to: "scratchpad.field"`)

## Troubleshooting

- If a conditional branch with nested HITL fails instead of pausing, ensure you’re on a version where HITL pauses are propagated in conditional branches, and the condition callable is synchronous in YAML.
- If resume doesn’t validate, confirm the payload matches `input_schema` exactly (e.g., enum values like "yes"/"no").

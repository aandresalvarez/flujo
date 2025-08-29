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

- Keep `input_schema` minimal and explicit.
- Use the sync helper `flujo.builtins.check_user_confirmation_sync` for YAML conditionals.
- Prefer short, directive messages.
- Avoid relying on implicit context merges; pass needed data explicitly or use `updates_context` on a dedicated adapter step.

## Troubleshooting

- If a conditional branch with nested HITL fails instead of pausing, ensure you’re on a version where HITL pauses are propagated in conditional branches, and the condition callable is synchronous in YAML.
- If resume doesn’t validate, confirm the payload matches `input_schema` exactly (e.g., enum values like "yes"/"no").

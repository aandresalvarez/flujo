# StateMachineStep — A First‑Class DSL Primitive

`StateMachineStep` is a high‑level orchestration primitive that drives execution through named states. Each state maps to its own Pipeline. Transitions are controlled via typed context fields (`current_state`, `next_state`) and optional declarative `transitions`.

## When to Use

Use a `StateMachineStep` when you need explicit, named phases with clear transitions (e.g., iterate until analysis → refine → finalize).

## Python DSL

```python
from typing import Any
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.state_machine import StateMachineStep

async def set_next_state(_: Any, *, context=None) -> str:
    # Signal transition to "refine"
    context.next_state = "refine"
    return "ok"

analyze = Pipeline.from_step(Step.from_callable(set_next_state, name="Analyze"))
refine = Pipeline.from_step(Step.from_callable(lambda x: x, name="Refine"))

sm = StateMachineStep(
    name="SM",
    states={"analyze": analyze, "refine": refine},
    start_state="analyze",
    end_states=["refine"],
)
pipe = Pipeline.from_step(sm)
```

## YAML

```yaml
version: "0.1"
steps:
  - kind: StateMachine
    name: SM
    start_state: analyze
    end_states: [refine]
    states:
      analyze:
        - kind: step
          name: SetNext
      refine:
        - kind: step
          name: Done
```

The YAML loader converts each `states.<name>` block into a Pipeline and instantiates the model through the registry.

## Imports In States

StateMachine states can import child pipelines via `uses: imports.<alias>`. The loader compiles these into first‑class `ImportStep`s, preserving policy‑driven execution and ImportStep semantics (context inheritance and outputs mapping).

Example:

```yaml
version: "0.1"
imports:
  clarify: "./clarification/pipeline.yaml"
  refine: "./refinement/pipeline.yaml"
steps:
  - kind: StateMachine
    name: Orchestrate
    start_state: clarification
    end_states: [done]
    states:
      clarification:
        - name: Clarify
          uses: imports.clarify
          updates_context: true
          config:
            inherit_context: true
            outputs:
              - { child: "import_artifacts.cohort", parent: "import_artifacts.cohort" }
        - name: SetNext
          uses: flujo.builtins.stringify  # or any agent
          updates_context: true
      done:
        - kind: step
          name: Done
```

Notes:
- Use `config.inherit_context: true` to run the child with a deep copy of the parent context.
- Use `config.outputs` to map child context fields back into the parent when `updates_context: true`.
- Execution for state pipelines uses the core policy router; no separate runner is spawned.

## Execution Semantics

- The policy executor reads `current_state` from `context.current_state` when present; otherwise starts with `start_state`
- The selected state’s Pipeline is executed using the core’s policy path
- On completion, the executor checks `context.next_state`; if present, it becomes the next `current_state`
- Execution stops when `current_state` is in `end_states`, or after one hop without an explicit `next_state`

## Context Keys

- `current_state` (optional): force the starting state at runtime
- `next_state`: set by your state pipelines to transition to the next state

## Testing Tips

- Unit test your step pipelines independently
- For integration, build a `Pipeline` with `StateMachineStep` and run via `ExecutorCore` to exercise policy routing
- If a state is terminal (in `end_states`), the policy won’t execute its body

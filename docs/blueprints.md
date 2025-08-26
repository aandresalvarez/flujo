# Authoring Blueprints (Summary)

This page summarizes the declarative YAML features for loops, maps, and parallel steps.
For detailed examples, see docs/creating_yaml.md.

## Loop Declaratives
- `loop.init`: one-time init ops before the first iteration (idempotent on isolated context).
- `loop.propagation.next_input`: presets `context` | `previous_output` | `auto` (auto uses context when any body step sets `updates_context: true`).
- `loop.output_template` or `loop.output: { ... }`: final output mapping evaluated after successful exit.

Friendly presets for non-technical users:
- `conversation: true`
- `stop_when: agent_finished`
- `propagation: context|previous_output|auto`
- `output: text: conversation_history` or `output: fields: { ... }`

## MapStep Sugars
- `map.init`: pre-run ops applied once before mapping begins.
- `map.finalize`: post-aggregation mapping; sees the aggregated results as `previous_step`.

## Parallel Reduction
- `reduce: keys|values|union|concat|first|last` applied after branches complete:
  - `keys`: returns branch names in declared order.
  - `values`: returns outputs in branch order.
  - `union`: merges dicts with last-wins (branch order).
  - `concat`: concatenates list outputs (non-lists appended).
  - `first|last`: picks first/last available output by branch order.

## CLI Aids
- `flujo create --wizard` — emit a natural YAML using the presets above.
- `flujo explain <path>` — summarize a YAML in plain language.

For full examples and advanced flags, see docs/creating_yaml.md.

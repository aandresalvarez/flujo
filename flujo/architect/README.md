# Flujo Architect — Usage Guide

This guide explains how to use and extend the programmatic Architect that powers `flujo create`.
It covers the agentic planning/generation flow, state machine, environment toggles, and testing.

## Overview

- The Architect runs as a `StateMachineStep` inside a Flujo pipeline.
- The CLI `flujo create` builds/executes this pipeline with an `ArchitectContext`.
- Default mode uses the conversational state machine that plans, selects tools, generates, and validates `pipeline.yaml`.
- A minimal mode is available via `FLUJO_ARCHITECT_MINIMAL=1`.

### Agentic Hooks
- `flujo.architect.planner`: Decomposes a user goal into named steps with purpose.
- `flujo.architect.tool_matcher`: Picks the best skill per planned step (runs via MapStep).
- `flujo.architect.yaml_writer`: Assembles a complete `pipeline.yaml` from tool selections.
- If any agent is missing or disabled, robust fallbacks keep the flow working.

## Quick Start

- Non-interactive (CI-friendly):
  - `uv run flujo create --goal "Build a simple pipeline" --non-interactive --output-dir ./output`
- Interactive (collect required params from skills):
  - `uv run flujo create --goal "Fetch a webpage and process it" --output-dir ./output`
- Validate later:
  - `uv run flujo dev validate --strict`

Notes:
- With `--non-interactive`, pass `--output-dir` to avoid accidental overwrites.
- Optional flags: `--name`, `--budget`, `--force`.

## State Machine Flow

- GatheringContext: Discover skills, analyze project (safe/no network), map framework schema. → GoalClarification
- GoalClarification: For now, forwards to Planning (future: clarifying Q&A). → Planning
- Planning: `PlannerAgent` builds a structured plan; fallback to heuristics.
  - Visualize plan (Mermaid).
  - Estimate cost from registry metadata.
  - → PlanApproval
- PlanApproval: Auto-approve by default; HITL hooks available. → ParameterCollection
- ParameterCollection: Interactive only; prompt for required params (`input_schema.required`). → Generation
- Generation: Prepare items → `MapToolMatcher` (parallel) → collect selections → `YamlWriterAgent`.
  - Fallback to legacy generator when agents disabled/unavailable.
  - → Validation
- Validation: Validate YAML; attempt conservative repair on failure. → DryRunOffer or loop
- DryRunOffer: Default forwards to Finalization; HITL dry run optional.
- DryRunExecution: In-memory run with side effects mocked. → Finalization
- Finalization: Terminal; returns `{"generated_yaml", "yaml_text"}` as final output.
- Failure: Reserved terminal for future error routing.

All transitions are driven by `context.next_state`.

## Agent Contracts (Pydantic Models)

- Planner → `ExecutionPlan`:
  - Input: `user_goal`, `available_skills`, `project_summary`, `flujo_schema`
  - Output: `{ plan_summary: str, steps: [{ step_name: str, purpose: str }, ...] }`
- Tool Matcher → `ToolSelection` (per plan step via MapStep):
  - Input: `step_name`, `purpose`, `available_skills`
  - Output: `{ step_name: str, chosen_agent_id: str, agent_params: dict }`
- YAML Writer → `GeneratedYaml`:
  - Input: `user_goal`, `tool_selections`, `flujo_schema`
  - Output: `{ generated_yaml: str }` (clean YAML string; no fences)

See `flujo/architect/models.py` for the canonical schema definitions.

## Context Model

`flujo/architect/context.py` defines `ArchitectContext` (extends `PipelineContext`):

- Inputs: `user_goal`, `project_summary`, `refinement_feedback`.
- Discovered: `available_skills`, `flujo_schema`.
- Plan: `execution_plan`, `plan_summary`, `plan_mermaid_graph`, `plan_estimates`.
- Structured (agentic): `execution_plan_structured`, `tool_selections`, `generated_yaml_structured`.
- Interaction: `plan_approved`, `dry_run_requested`, `sample_input`, `hitl_enabled`, `non_interactive`.
- Artifact & Validation: `generated_yaml`, `yaml_text`, `validation_report`, `yaml_is_valid`, `validation_errors`.
- Helpers: `prepared_steps_for_mapping`.

## Built-ins and Agents

- Built-in skills used by the Architect (registered in `flujo/builtins.py`):
  - `flujo.builtins.discover_skills`, `flujo.builtins.analyze_project`, `flujo.builtins.visualize_plan`,
    `flujo.builtins.estimate_plan_cost`, `flujo.builtins.validate_yaml`, `flujo.builtins.repair_yaml_ruamel`,
    `flujo.builtins.run_pipeline_in_memory`, `flujo.builtins.stringify`.
- Architect agent stubs (enabled by default; safe for local iteration):
  - `flujo.architect.planner`, `flujo.architect.tool_matcher`, `flujo.architect.yaml_writer`.
  - Stubs follow the contracts and provide deterministic fallback behavior.

## Environment Toggles

- `FLUJO_ARCHITECT_MINIMAL=1`: Bypass state machine; emit minimal YAML.
- `FLUJO_ARCHITECT_STATE_MACHINE=0`: Legacy toggle to disable state machine.
- `FLUJO_ARCHITECT_AGENTIC_PLANNER=0|false|no|off`: Force heuristic planning even if planner agent is available.
- `FLUJO_ARCHITECT_AGENTIC_TOOLMATCHER=0|false|no|off`: Disable tool matcher agent; use safe defaults.
- `FLUJO_ARCHITECT_AGENTIC_YAMLWRITER=0|false|no|off`: Disable YAML writer agent; use fallback assembler.

## Interactive vs Non-Interactive

- Non-interactive (`--non-interactive`): Skips prompts; suitable for CI. Ensure `--output-dir` is set.
- Interactive (default): Prompts for missing required params from skill `input_schema.required`.
- Future: add HITL plan approval, dry-run prompts, and refinement.

## Output & Validation

- Finalization always returns `{"generated_yaml": str, "yaml_text": str}`.
- Side-effect skills can be gated; use `--allow-side-effects` for non-interactive runs.
- Post-run: `uv run flujo dev validate --strict` to revalidate the saved YAML.

## Extending the Architect

- Swap in your own planner/tool-matcher/yaml-writer by registering skills with the same IDs.
- Improve parameter collection (typing-aware prompts, detectors).
- Add HITL in `PlanApproval` with `flujo.builtins.ask_user` + `flujo.builtins.check_user_confirmation`.
- Add new states by extending `StateMachineStep.states` and driving transitions via `context.next_state`.

## Testing

- Unit:
  - Built-ins: analyze_project, visualize_plan, estimate_plan_cost, run_pipeline_in_memory.
  - Architect agent toggles/resilience:
    - `tests/unit/architect/test_planner_agent_toggle.py`
    - `tests/unit/architect/test_tool_matcher_resilience.py`
    - `tests/unit/architect/test_yaml_writer_and_finalize.py`
- Integration:
  - End-to-end with agent stubs: `tests/integration/architect/test_agentic_end_to_end.py`
  - Happy path: `tests/integration/architect/test_architect_happy_path.py`
  - Repair loop: `tests/integration/architect/test_architect_validation_repair.py`
  - Plan rejection/approval HITL flows and security validation suites.
- Run:
  - Fast: `make test-fast`
  - Integration: `pytest tests/integration/architect -q`
  - Full: `make test`

## Troubleshooting

- No YAML in output: Ensure Finalization ran; the CLI reads both step output and context.
- Validation loops: Check `validation_report`; repair is conservative by design.
- Skills not found: Ensure packaged entry points or a local skills catalog is discoverable.

## File Map

- `builder.py`: State machine and per-state helpers.
- `context.py`: ArchitectContext.
- `models.py`: Pydantic models for agent contracts.
- `README.md`: This guide.

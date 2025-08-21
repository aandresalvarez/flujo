# Flujo Architect – Usage Guide

This guide explains how to use and extend the new programmatic Architect that powers `flujo create`.
It summarizes the user experience, state machine flow, configuration knobs, and extension points.

## Overview

- The Architect is implemented as a `StateMachineStep` and runs entirely inside a Flujo pipeline.
- The CLI command `flujo create` builds and runs this pipeline using a dedicated `ArchitectContext`.
- The default experience is safe-by-default, non-interactive, and produces a minimal, valid
  `pipeline.yaml` tailored from your goal.

## Quick Start

- Non-interactive one shot (recommended for automation):
  - `uv run flujo create --goal "Build a simple pipeline" --non-interactive --output-dir ./output`
- Interactive prompts (parameter collection when required by skills):
  - `uv run flujo create --goal "Fetch a webpage and process it" --output-dir ./output`
- Validate the project pipeline later:
  - `uv run flujo dev validate --strict`

Notes:
- `--output-dir` is mandatory with `--non-interactive` to avoid accidental overwrites.
- You may pass `--name`, `--budget`, and `--force` to control naming, default budget, and overwriting.

## State Machine Flow

The Architect defines the following states and transitions:

- GatheringContext: Discovers available skills, analyzes the project tree (safe, no network), and
  fetches framework step schemas. Advances to GoalClarification.
- GoalClarification: Current implementation forwards to Planning. A future iteration can ask
  clarifying questions here.
- Planning: Builds a minimal execution plan from the goal using heuristics and available skills.
  - Heuristics:
    - If a URL is present in the goal (or mentions http), choose `flujo.builtins.http_get` and pre-fill `url`.
    - If the goal suggests search (“search”, “find”), choose `flujo.builtins.web_search` with `query` = goal.
    - Otherwise, default to `flujo.builtins.stringify`.
  - Visualizes the plan as a Mermaid graph (`plan_mermaid_graph`).
  - Estimates cost by summing `est_cost` metadata from the Skill Registry.
  - Advances to PlanApproval.
- PlanApproval: Currently auto-approves and proceeds. You can wire HITL here to ask the user.
- ParameterCollection: Prompts for missing required parameters based on each chosen skill’s
  `input_schema.required` when in interactive mode. Non-interactive mode skips prompts.
  Advances to Generation.
- Generation: Converts the plan into `pipeline.yaml` text and stores it in context (`yaml_text`).
- Validation: Validates YAML in-memory. If invalid, runs a conservative repair and re-validates.
  - On valid, goes to DryRunOffer; on invalid, loops to attempt repairs.
- DryRunOffer: Currently advances to Finalization. You can add a HITL branch to run an optional dry run.
- DryRunExecution: Available state that runs the pipeline in-memory with side-effect skills mocked.
  Advances to Finalization.
- Finalization: Terminal state used by the CLI to write the `pipeline.yaml` to disk and optionally
  update budgets.
- Failure: Terminal fallback reserved for future error reporting.

All transitions are driven by the StateMachine policy reading `context.scratchpad.next_state`.

## Context Model

`flujo/architect/context.py` defines `ArchitectContext`, extending `PipelineContext` with:

- Inputs: `user_goal`, `project_summary`, `refinement_feedback`.
- Discovered: `flujo_schema`, `available_skills`.
- Plan: `execution_plan`, `plan_summary`, `plan_mermaid_graph`, `plan_estimates`.
- Interaction: `plan_approved`, `dry_run_requested`, `sample_input`.
- Artifact & Validation: `generated_yaml`, `yaml_text`, `validation_report`, `yaml_is_valid`, `validation_errors`.
- CLI/Test helpers: `prepared_steps_for_mapping`.

## Built-in Skills Used

Registered in `flujo/builtins.py` and used by the Architect:

- `flujo.builtins.discover_skills`: Collects registered skills (catalog + entry points).
- `flujo.builtins.analyze_project`: Shallow, safe file system scan for project hints (no network).
- `flujo.builtins.get_framework_schema`: Generates JSON Schemas for registered DSL step kinds.
- `flujo.builtins.visualize_plan`: Renders a basic Mermaid diagram for a linear plan.
- `flujo.builtins.estimate_plan_cost`: Sums `est_cost` metadata from the Skill Registry.
- `flujo.builtins.validate_yaml`: Validates `yaml_text` into a `ValidationReport`.
- `flujo.builtins.repair_yaml_ruamel`: Conservative YAML repair for common shape issues.
- `flujo.builtins.run_pipeline_in_memory`: Compiles and runs YAML in-memory, mocking side effects.
- `flujo.builtins.stringify`: Safe default step; echoes input.

## Interactive vs Non-Interactive

- Non-interactive (`--non-interactive`):
  - No prompts are shown. Parameter collection is skipped if anything is missing.
  - Good for CI and repeatable generation.
- Interactive (default):
  - Missing required parameters (from `input_schema.required`) trigger Typer prompts.
  - Future: you can add interactive plan approval, dry run prompts, and refinement.

## Output & Validation

- The CLI writes `pipeline.yaml` to the chosen directory after validation and optional plan approval.
- Side-effect skills in the YAML are detected before writing. In non-interactive mode, use
  `--allow-side-effects` to proceed automatically.
- After writing, the CLI can update budgets in `flujo.toml` for the detected pipeline name.

## Extending the Architect

- Improve planning: Replace `_make_plan_from_goal` with a more sophisticated mapper using
  `available_skills`, model prompts, or a rule set.
- Add plan approval: Insert a HITL step in `PlanApproval` using `flujo.builtins.ask_user` and route
  responses with `flujo.builtins.check_user_confirmation`.
- Enhance parameter collection: Add typing-aware prompts or pre-fill from project detectors.
- Enable DryRunOffer: Ask the user to run a safe in-memory dry run via `run_pipeline_in_memory`.
- Add states: New states can be registered in the `StateMachineStep.states` mapping; use
  `scratchpad.next_state` to transition.

## Troubleshooting

- “No YAML found” after run: Ensure the build states set `yaml_text` in context; the CLI extracts it
  from the final step outputs and from context fields.
- Validation loop doesn’t finish: The repair step is conservative; check console output or inspect
  `validation_report` in context.
- Skills not found: Make sure packaged entry points or a local skills catalog is discoverable.

## Development Tips

- Run the full suite:
  - `make format && make lint && make typecheck && make test`
- Check what the Architect produced:
  - `uv run flujo dev show-steps ./output/pipeline.yaml`
  - `uv run flujo dev validate --strict`
- Inspect available skills:
  - Add debug prints or call `flujo.builtins.discover_skills` in a small pipeline.

## File Map

- `builder.py`: State machine definition and per-state helpers/adapters.
- `context.py`: ArchitectContext model.
- `README.md`: This guide.

If you want a deeper conversational flow (approvals, refinement, dry run prompts),
let us know and we’ll wire the HITL steps and policies accordingly.

## Testing

- Unit (built-ins): Validates the new FSD‑024 helpers.
  - `tests/unit/architect/test_builtins_analyze_project.py`: detects common files and handles empty dirs.
  - `tests/unit/architect/test_builtins_visualize_plan.py`: Mermaid rendering with correct node names.
  - `tests/unit/architect/test_builtins_estimate_plan_cost.py`: sums `est_cost` metadata.
  - `tests/unit/architect/test_builtins_run_pipeline_in_memory.py`: mocks side‑effects in sandbox; no file writes; non‑blocking execution.
- Integration (state machine): Runs the full flow end‑to‑end in‑process.
  - `tests/integration/architect/test_architect_happy_path.py`: produces valid YAML; confirms `GenerateYAML` executed.
  - `tests/integration/architect/test_architect_validation_repair.py`: invalid → repair → valid loop.
  - `tests/integration/architect/test_architect_plan_rejection.py`: context‑driven plan rejection → `Refinement` → re‑planning.
  - `tests/integration/architect/test_architect_plan_approval_hitl.py`: HITL plan approval via `ask_user` → denial triggers refinement.
- Running tests:
  - Fast pass: `make test-fast`
  - Integration only: `pytest tests/integration/architect -q`
  - Full: `make test`
- Enabling the full Architect state machine for tests/runs:
  - Set `FLUJO_ARCHITECT_STATE_MACHINE=1` (the default builder returns a minimal single‑step pipeline for legacy tests).

### Manual E2E Scenarios (CLI)

- Magic Moment:
  - `flujo init` → `flujo create` → enter goal: `Search the web for the price of a Tesla Model 3 and save it to price.txt` → approve.
  - Expect a valid `pipeline.yaml` using `flujo.builtins.web_search` and `flujo.builtins.fs_write_file`.
- Parameter Collection:
  - `flujo create` with a goal that requires required params (e.g., a hypothetical `create_jira_ticket`).
  - Interactive prompts fill `input_schema.required` fields.
- Dry Run:
  - `flujo create` → approve → opt‑in to dry run (when enabled) → provide sample input.
  - Expect printed output from the in‑memory run, then pipeline saved.

# Architect Flow: First‑Principles Design and Fixes

## Summary
This document describes the first‑principles design for the Architect flow and the changes implemented to make it deterministic, policy‑driven, and robust. It covers the root cause, guiding principles, concrete changes in policies/built‑ins/recipe/CLI, test expectations, and troubleshooting tips.

## Problem Statement
- The Architect CLI occasionally selected or emitted the wrong pipeline variant and did not consistently produce YAML.
- ConditionalStep behavior was modified in a way that broke its public contract (passing reshaped input to the condition callable), causing broad failures.
- The validity decision (valid/invalid) was not reliably tied to a single source of truth, leading to non‑deterministic branch selection.

## Design Goals
- Deterministic loop behavior and branch selection.
- Single source of truth for validity: `context.yaml_is_valid` set by validation.
- Strict policy contracts (no hidden reshaping of inputs or control‑flow swallowing).
- Clear, string final output for easy CLI extraction (no heuristics or re‑serialization).

## First‑Principles Decisions
1) ConditionalStep Contract
- Call `condition_callable(data, context)` exactly as provided.
- Merge branch context back using `ContextManager` semantics.

2) Source of Truth for Validity
- Use `context.yaml_is_valid` only.
- Update the flag via a dedicated adapter `validation_report_to_flag` after validation.
- Select branches using a `select_validity_branch` function that prefers the last step’s `{"yaml_is_valid": ...}` output and otherwise falls back to `context.yaml_is_valid`.

3) Deterministic Loop Shape
- Each iteration:
  - write/extract/store → validate → set validity flag → ValidityBranch.
  - If invalid: repair → apply repaired YAML → revalidate → reflag → record "valid" immediately via nested ValidityBranch.
  - Emit the current YAML as a string.
- Exit the loop based on the validity flag (not counters).

4) Output Clarity
- End each iteration by emitting the current YAML as a string.
- End the top‑level pipeline with a final emit step so the overall `StepResult.output` is the YAML string.

## Changes Implemented
- Policies
  - Restored ConditionalStep contract (pass `(data, context)` unchanged to the condition).
  - Added a unit guard for this contract:
    - `tests/unit/test_conditional_policy_contract.py`.

- Built‑ins
  - Added `capture_yaml_text` to normalize YAML outputs and update context.
  - `select_validity_branch`: uses last output dict or falls back to `context.yaml_is_valid`.
  - `validate_yaml` registration resolves dynamically so test monkeypatches take effect.

- Recipe (`flujo/recipes/architect_pipeline.yaml`)
  - Loop body (ValidateAndRepair):
    - `decide_yaml_source` → `YamlSource` (reuse present YAML or call Architect agent)
    - `extract_yaml_text` → `store_yaml_text` (updates context.yaml_text)
    - `validate_generated_yaml` → `check_validation_status` (updates context.yaml_is_valid)
    - `ValidityBranch` (main)
      - `valid`: noop
      - `invalid` branch:
        - `repair_yaml` → `extract_repaired_yaml` → `revalidate_generated_yaml` → `apply_repaired_yaml` → `recheck_validation_status`
        - Nested `ValidityBranch` (records "valid" immediately after repair)
    - `emit_current_yaml` (string)
  - Final top‑level: `final_emit_yaml` (string)

- CLI
  - No scaffolds or re‑serialization. YAML extraction walks step_history and context attributes and prefers the final emit.
  - Added `--debug` printout for step names and output types to accelerate triage.

## Testing Impact
- Conditional tests are stable (contract guarded).
- Architect integration and GPT‑5 tests become deterministic:
  - First iteration: `invalid` recorded (malformed inline steps).
  - After repair/revalidation: nested branch records `valid`.
  - Final output is a YAML string; CLI extraction is straightforward.

## Troubleshooting
- Enable CLI debug: `uv run flujo create --goal demo --non-interactive --output-dir /tmp/arch --debug` to view step outputs.
- Validate built‑in resolution: `validate_yaml` is resolved dynamically; monkeypatch before the recipe is compiled.
- If branch keys don’t show as expected:
  - Confirm `check_validation_status` sets `yaml_is_valid` prior to `ValidityBranch`.
  - Confirm the nested `ValidityBranch` exists after reflagging inside the invalid branch.

## Rationale & Alignment
- All behavioral logic lives in policies/recipe/built‑ins; no executor core changes.
- Context updates happen via `updates_context` and `ContextManager`.
- Control‑flow exceptions are propagated (no swallowing).
- Deterministic outputs eliminate CLI guesswork.

## Future Work
- Add a small optional env‑guarded trace in `select_validity_branch` for deep debugging (e.g., `FLUJO_DEBUG_COND=1`).
- Consider a minimal end‑to‑end test that asserts the presence of both "invalid" and "valid" keys using the nested ValidityBranch approach as a regression guard.


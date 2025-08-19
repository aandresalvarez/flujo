# Reflections on Architect Flow Fixes

## Summary of Work Done
- Removed deprecated `architect_mode` usage; the `create` command now always loads the canonical `flujo/recipes/architect_pipeline.yaml`.
- Made YAML extraction robust and non‑heuristic: scan step history and context for `generated_yaml`/`yaml_text`, prefer the pipeline’s final emit step, and avoid re‑serialization or scaffolds.
- Restored ConditionalStep policy contract: conditions receive `(data, context)` unchanged. Added a unit test to guard this.
- Implemented a deterministic Architect loop (ValidateAndRepair):
  - write/extract/store → validate → set validity flag → ValidityBranch.
  - Invalid path: repair → extract repaired → revalidate → reflag, then record a nested ValidityBranch decision immediately.
  - Emit the current YAML string; exit based on validity.
- Built‑ins:
  - Added `http_get` with graceful degradation when `httpx` is missing.
  - Added `fs_write_file` with an async fallback using thread offload when `aiofiles` is missing.
  - Added adapters: `capture_yaml_text`, `validation_report_to_flag`, `select_validity_branch`.
  - Made registry resolution for `validate_yaml` dynamic so monkeypatches in tests take effect.
- Wrote a design doc: `docs/architect_flow_first_principles.md` capturing the approach, rationale, and troubleshooting.
- Added a unit test `tests/unit/test_conditional_policy_contract.py` to lock in the policy contract for ConditionalStep.

## Why These Changes
- Align with policy‑driven architecture: move behavior into policies, built‑ins, and recipes (not core or CLI hacks).
- Single source of truth for validity: ensure decisions depend only on `context.yaml_is_valid` updated by validation.
- Determinism: ensure the loop structure and branch decisions are predictable and easy to reason about.
- Testability: dynamic resolution for built‑ins allows test monkeypatches to shape validation outcomes; added guard tests for critical contracts.

## Lessons Learned
- Small deviations in policy contracts (e.g., reshaping condition inputs) can cascade into many unrelated failures.
- Validity should be represented explicitly in context and set by a single adapter after validation, not inferred from content shape.
- Final pipeline outputs should be clearly emitted as strings to keep CLI behavior simple and robust.
- Keeping changes localized (policies/recipe/built‑ins) respects the core and reduces regression risk.

## Follow‑Ups
- Optional: add an env‑guarded debug trace in `select_validity_branch` (e.g., `FLUJO_DEBUG_COND=1`) for faster triage.
- Consider a small end‑to‑end regression test that asserts both `invalid` and `valid` branch keys are recorded during the repair path.


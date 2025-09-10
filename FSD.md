FSD-AROS-001: Robust Structured Output (AROS v2)

  - Owner: Flujo Core
  - Status: Draft (Proposed)
  - Target: Flujo 0.4.x minor release (backwards compatible)
  - Date: 2025-09-10

  Problem Statement

  - Structured-output steps intermittently fail on imperfect model outputs, especially with
  noisy or multilingual inputs.
  - Provider JSON-mode can throw early errors (e.g., UnexpectedModelBehavior) that bypass
  current repair.
  - The linter (V-C1/V-A8/V-A5) and runtime behavior are not fully aligned, creating friction
  for simple demos and robust production flows.
  - Streaming and complex schemas exacerbate failure modes.

  Goals

  - Dramatically improve success rates for JSON-structured steps while retaining simplicity
  for demos.
  - Catch and repair provider-level JSON-mode failures.
  - Align linter with runtime capabilities; offer auto-remediation.
  - Support streaming + structured output.
  - Keep API changes minimal and backwards compatible.

  Non-Goals

  - Provider-specific feature parity beyond OpenAI Responses JSON-mode.
  - Full JSON Schema draft-2020-12 compliance; we focus on a robust subset.

  User Stories

  - As a user, my step with a JSON schema succeeds even on imperfect outputs, thanks to
  automatic coercion/repair/fallback.
  - As a contributor, my simple demo works with plain string outputs without linter thrash.
  - As a maintainer, I can see precise telemetry (what failed, which repairs ran, what fallback
  path was used).

  Overview
  AROS v2 spans changes in five areas:

  1. Wrapper-level catch + repair for provider JSON-mode errors.
  2. Step-level enforcement parity and schema auto-promotion.
  3. Robust JSON coercion and post-stream aggregation.
  4. Adaptive fallback strategy (retry → stricter prompt → extract_from_text → degrade).
  5. Linter/CLI alignment and minimal built-ins to remove custom adapters.

  Detailed Design

  -
      1. Wrapper Catch + Repair
      - Files: flujo/agents/wrapper.py
      - Add try/catch around provider JSON-mode invocation:
      - Catch provider “unexpected JSON” errors (e.g., UnexpectedModelBehavior).
      - Convert them into the same repair path as pydantic ValidationError/ModelRetry.
  - Deterministic repair (existing) runs first; then LLM-based repair; then re-validate.
  - Config (from settings):
      - aros.repair.max_attempts (default 2)
      - aros.repair.enable_llm (default true)
      - aros.degrade_on_unexpected_behavior (default true) → defers to fallback policy (see 4).

  Pseudocode (wrapper):

  - result = call_provider()
  - if provider_error_is_json_mode:
        try deterministic_repair(result.raw_text)
        or try llm_repair
        revalidate
  -
  else return processed_output
  -
      2. Step-Level Enforcement Parity
      - Files: flujo/domain/blueprint/compiler.py, flujo/domain/blueprint/loader.py
      - When a step has processing.structured_output set (auto or openai_json) and agent has
  output_schema:
      - Auto-populate step.meta.processing.schema with the agent’s schema if empty.
      - If agent has schema but step lacks processing, auto-set processing.structured_output:
  openai_json (unless step opts out).
  -
  Add per-step override meta: processing.enforcement: off|validate|repair (default
  validate+repair).
  -
  CLI validate warns only if both are absent and a schema exists; --fix can auto-insert the
  step processing.
  -
      3. Robust JSON Coercion + Streaming Aggregation
      - New file: flujo/utils/json_normalizer.py
      - Functions:
        - normalize_json_text(str) -> str (strip code fences, normalize quotes, remove trailing
  commas, extract last balanced object).
        - parse_and_validate(str, schema?) -> object | raises
  - Streaming:
      - Files: flujo/application/core/step_coordinator.py (or wrapper)
      - Aggregate streamed chunks into a final string; after stream completion, run
  normalize_json_text + parse_and_validate when step processing requests structured JSON.
      - Config: aros.streaming.aggregate_structured (default true)

  -
      4. Adaptive Fallback Strategy
      - Files: flujo/application/core/step_policies.py and/or flujo/application/core/
  executor_core.py
      - Step-level fallback sequence for structured steps:
      1) Retry with stricter instruction (“Return only JSON. No extra text.”).
      2) Switch to built-in extractor flujo.builtins.extract_from_text with the same schema.
      3) Degrade to plain text summary (string), then convert to a minimal object only if the
  next step expects dict (by using built-in adapters; see 5).
  - Config:
      - aros.fallback.max_stages (default 3)
      - aros.fallback.enable_extractor (default true)
      - aros.fallback.degrade_to_string (default true)
  -
  Policy must preserve control-flow exceptions (non-data failures re-raise).
  -
      5. Linter/CLI Alignment + Minimal Built-ins
      - Linter (ContextLinter – V-C1):
      - Allow updates_context=True when step declares processing.structured_output with schema
  and step.agent.target_output_type is dict or a Pydantic model (mergeable_output=true).
      - Add a per-step meta override: mergeable_output: true (opt-in when the user knows output
  merges).
  - Linter (V-A8):
      - If agent has schema but step lacks processing.structured_output, warn and propose auto-
  fix.
  - CLI Fixers (--fix):
      - Fix V-A8 by adding processing.structured_output: openai_json and copying schema into
  step meta.
      - Fix V-C1 by inserting a dictionary wrapper adapter step or by setting
  mergeable_output=true if the output is known to be dict-like.
  - Built-ins (flujo/builtins.py):
      - Add wrap_dict (params: key) to wrap a string into {key: string}.
      - Add ensure_object to coerce Pydantic model/other to dict via safe serialization.
      - Both are side_effects=False and accept Any → dict.

  Configuration

  - Settings (flujo.infra.settings):
      - aros.enabled (bool, default true)
      - aros.repair.max_attempts (int, default 2)
      - aros.repair.enable_llm (bool, default true)
      - aros.degrade_on_unexpected_behavior (bool, default true)
      - aros.streaming.aggregate_structured (bool, default true)
      - aros.schema_simplify (bool, default false) – optional, see “Schema Simplifier”
      - aros.telemetry.verbose (bool, default false)

  Telemetry

  - Add events: aros.repair.started, aros.repair.success, aros.repair.failed
  - Add tags for provider_error, stage (deterministic|llm|extractor|degrade), and schema_hash
  - Emit agent.usage and aros.fallback.stage metrics

  Schema Simplifier (optional)

  - For complex schemas, a module that reduces to a flat object with string fields for LLM
  compliance
  - Used only when aros.schema_simplify=true and validation repeatedly fails

  Backward Compatibility

  - Default behavior remains unchanged unless processing.structured_output is set or the step is
  configured to use AROS fallback stages.
  - Built-in adapters do not alter existing behavior unless referenced.

  Security & Privacy

  - Repair prompts must not leak secrets; they only send the failing output and schema.
  - Streaming aggregator buffers in memory; no disk writes unless debug-export is on.

  Risks

  - Over-aggressive normalization could accept malformed JSON. We mitigate by validating after
  normalization.
  - Auto-fix could surprise users; keep --fix opt-in and print a diff summary.

  Testing Plan

  - Unit:
      - json_normalizer: fence stripping, quotes, trailing commas, balanced extraction.
      - Wrapper: simulate UnexpectedModelBehavior; validate repair runs.
      - Step policies: fallback stage transitions; ensure control-flow exceptions are re-raised.
      - Built-ins: wrap_dict, ensure_object.
      - Linters: V-C1/V-A8 logic paths, meta overrides.
  - Integration:
      - Structured step with multilingual inputs; flaky provider stubs returning near-JSON.
      - Streaming structured step -> aggregated JSON validation.
      - CLI --fix applies correct changes.
  - E2E:
      - Real provider (OpenAI) behind a small test (if keys available), else stubbed.

  Rollout

  - Phase 1: Wrapper catch+repair; built-ins; linter/--fix.
  - Phase 2: Streaming aggregator; adaptive fallback.
  - Phase 3: Schema simplifier (optional).
  - Provide migration notes in CHANGELOG and docs: “AROS v2”.

  Acceptance Criteria

  - A step configured with structured JSON and small schema succeeds ≥ 99% across noisy inputs
  with retries and repair enabled (in CI with provider stub).
  - Early provider errors are routed through the repair/fallback path.
  - Streaming structured steps validate at end-of-stream with aggregated output.
  - CLI --fix cleanly resolves V-A8/V-C1 in typical YAMLs.
  - No regressions in plain-string demo pipelines.

  Implementation Touch Points

  - flujo/agents/wrapper.py
      - Catch provider JSON-mode failures and route to repair; add config gates.
      - Post-stream aggregation hook (or coordinate with step_coordinator).
  - flujo/utils/json_normalizer.py (new)
      - normalize_json_text, parse_and_validate
  - flujo/application/core/step_coordinator.py (or wrapper)
      - Streaming chunk aggregation + post-aggregation validation when structured_output
  enabled.
  - flujo/application/core/step_policies.py
      - Adaptive fallback logic; respect control-flow exceptions; telemetry for stages.
  - flujo/domain/blueprint/compiler.py, loader.py
      - Auto-promote schema to step meta; auto-set structured_output when missing.
  - flujo/validation/linters.py
      - V-C1: allow mergeable_output true or dict-like outputs; V-A8 improved detection.
  - flujo/validation/fixers.py
      - --fix rules for V-A8 and V-C1; optional for V-A5.
  - flujo/builtins.py
      - Register wrap_dict and ensure_object built-ins.

  Developer Notes

  - Keep all new behavior behind explicit step processing toggles or AROS settings; default
  should remain low-risk.
  - Telemetry should be concise by default; verbose gated by a setting.
  - Avoid tight coupling between linter and runtime; use small, stable flags (mergeable_output)
  to reconcile.

  Open Questions

  - Should we treat all Pydantic models as mergeable for V-C1 by default? (Current proposal:
  opt-in via mergeable_output=true.)
  - How far to go with schema simplification in core vs. keep as an opt-in helper only?

  If you’d like, I can start with Phase 1 (wrapper catch+repair, built-ins, linter/--fix) and
  open a PR with code scaffolding and tests to accelerate adoption.

  Phase 1 — Implementation Status (PR: aros-phase1)

  - Wrapper: AsyncAgentWrapper now treats pydantic_ai.exceptions.UnexpectedModelBehavior
    like ValidationError/ModelRetry and routes through deterministic→LLM repair.
  - Built-ins: Added flujo.builtins.wrap_dict and flujo.builtins.ensure_object.
  - Linter/CLI: Added fixer V-A8 to enable processing.structured_output: openai_json when a
    schema is present; integrated with --fix/--fix-dry-run patch preview.
  - Tests: unit coverage for wrapper catch+repair, new built-ins, and V-A8 fixer.
  - Docs: this FSD section documents Phase 1 changes; CHANGELOG to be updated on merge.

## Engineering Specification: Flujo AROS (Adaptive Reasoning & Output System)

Version: 1.2 (policy-aligned + strengthened AOP)
Status: Proposed
Owner: Flujo Core Team

1) Overview & Vision

LLM agents can fail with malformed, stringified, or loosely typed outputs and occasionally flawed plans. AROS improves reliability by:
- Preventing format errors with model-native structured outputs when supported.
- Repairing or coercing minor output mismatches via the existing processor pipeline.
- Optionally pre-checking reasoning for high-assurance steps.
- Surfacing rich trace events for health analysis without adding storage tables.

This revision aligns fully with the Flujo Team Guide: policy-driven execution, control-flow exception safety, context idempotency, proactive quotas, centralized configuration, and agent factories.

2) Non-Negotiable Architectural Constraints

- Policy-driven: All step behavior lives in the AgentStep policy; no special-casing inside ExecutorCore.
- Exception safety: Never convert control-flow exceptions into data failures—re-raise them.
- Idempotency: All added logic must operate within per-attempt isolated context; only merge on success.
- Quotas: No reactive budget gating; continue Reserve → Execute → Reconcile patterns already enforced by policies.
- Central config: Read global defaults via ConfigManager; step-level toggles come from DSL step config. Never read env directly in domain/policies.
- Agent creation: Continue using `flujo.agents` factories/wrappers.

3) Components (Practical, Minimal, Robust)

3.1 Structured Output Enforcement (SOE)
- Purpose: Use pydantic‑ai model capabilities for structured outputs to reduce JSON-shape failures up front.
- Integration point: Wrapper/factory-level enablement; policies only signal intent. The wrapper inspects the pydantic‑ai model and attaches the appropriate response_format hint when supported (e.g., JSON Schema mode).
- Phase 1: Best-effort JSON object/JSON schema mode via pydantic‑ai; if unsupported, no-ops safely. Experimental Outlines/XGrammar adapters remain telemetry-only (no runtime effect) and off by default.
- Observability: Trace `grammar.applied {mode}` when enabled; otherwise omit events unless explicitly configured.

3.2 Adaptive Output Processors (AOP) via Existing Processor Pipeline
- Purpose: Deterministically normalize agent outputs after generation using the existing output-processor chain.
- Design:
  - Stage 0: Deterministic JSON extraction + bounded unescape
    - Add a streaming, stack-based brace/bracket matcher to extract the largest balanced `{...}` or `[...]` region from mixed text/code-fenced outputs. If both exist, choose the one matching the schema root (`object` vs `array`).
    - Add a bounded double-encoded fix: detect quoted/escaped JSON (e.g., `"{\"k\":1}"`) and iteratively unescape up to max depth (default 2). Guard with size/time caps.
  - Stage 1: Tiered tolerant decode (opt-in)
    - Default: `orjson.loads` when available, else Python `json.loads`.
    - If configured (`tolerant_level >= 1`): try `json5/pyjson5` for comments/single quotes/trailing commas.
    - If configured (`tolerant_level >= 2`): try `json-repair` and log a patch preview. Off by default for safety.
  - Stage 2: Syntactic JSON fixer
    - Reuse DeterministicRepairProcessor as last resort before retry/repair, only for JSON-shaped expectations.
  - Stage 3: Schema-aware Smart Coercion
    - Add `SmartTypeCoercionProcessor` that walks the decoded structure against JSON Schema and applies whitelisted, unambiguous conversions:
      - `"42"→42` (integer), `"3.14"→3.14` (number), `"true"/"false"/"0"/"1"→bool` (boolean)
      - String→list when schema is `array` and the string looks like `[...]`
      - Respect `anyOf/oneOf`: try branches in order, choose the first that fully validates; record chosen index in trace.
    - Fail fast on ambiguous cases; never guess semantics.
  - Custom coercion
    - Encourage users to implement custom output processors; avoid a bespoke rules DSL in core. AROS auto-injects built-ins when enabled; user processors run in the same chain.
- Safety:
  - Processors run inside the per-attempt isolated context; the policy merges context back only on success.
  - Final schema/type validation remains strict; if coercion doesn’t resolve, fail with precise feedback and let the normal retry loop handle it.
- Observability: Emit `aros.aop.syntactic.{success|skip|fail}` and `aros.aop.coercion.{success|skip|fail}` trace events with concise details.

3.2.1 Optional high-performance typed validation (hot path)
- For hot paths, optionally decode directly into typed structures (e.g., `msgspec.Struct` or Pydantic v2 `TypeAdapter`) to combine speed and validation in one pass. This remains optional and must be feature-gated to avoid mandatory dependencies.

3.3 Optional Reasoning Precheck (Preflight)
- Purpose: Catch obvious plan issues cheaply before a costly generation.
- Mode: Off by default. When enabled in step config, run a low-cost pre-validation inside the AgentStep attempt loop.
- Implementation:
  - Add a local checklist pre-gate: run a fast required-field/shape presence check from the schema. If basics are missing, skip validator and proceed to repair/coercion path.
  - Provide a `PlanCheckValidator` that takes a short plan representation and returns pass/fail plus feedback. Use it only when a plan is available; otherwise skip gracefully (no forced prompt changes).
  - Optional scoring integration: allow a simple scoring/threshold plugin; if score < threshold, treat as precheck fail and provide precise retry feedback.
  - Optional consensus gate: for high-stakes steps, sample two very short plans; if they disagree on key actions, run validator; else proceed.
  - If precheck fails, record an `aros.reasoning.precheck.fail` event and continue within the same attempt by turning validator/scorer feedback into retry guidance (no control-flow exception conversion). Keep it lightweight (small token budget) and explicitly optional.
- Observability: `aros.reasoning.precheck.{pass|fail|skipped}`.

3.4 Health Reporting via Traces (No New Tables in Phase 1)
- Purpose: Provide visibility into how often AROS intervenes, by agent/model/step.
- Storage: Use existing TraceManager events persisted through the SQLite backend (`spans`/`traces` tables). Do not add a new `correction_ledger` table in Phase 1.
- CLI: Add `flujo dev health-check` that aggregates AROS events from traces and reports:
  - Top steps by coercion frequency
  - Models with frequent SOE skips (unsupported)
  - Precheck failure rates
  - Suggested actions (e.g., enable SOE where supported; add a custom processor)

3.4.1 Future: Optional Correction Ledger (Phase 2)
- If trace querying proves insufficient, introduce a `correction_ledger` table with fields:
  - `id, timestamp, step_name, agent_id, provider_model, schema_hash, coercion_policy_version, stage, repair_depth, transforms_applied, latency_ms_saved`
  - Carefully gate writes; keep it optional and behind `[aros].correction_ledger_enabled`.

4) Execution Flow (AgentStep Policy)

Within the existing AgentStepExecutor attempt loop:
1. Determine AROS options from step config + global defaults (via ConfigManager).
2. If SOE is enabled and supported for the model, set structured-output hints via the agent wrapper/factory.
3. Call the agent as usual (respecting quotas, timeouts, retries).
4. Run output processors. If AOP is enabled, auto-inject built-ins in the processor chain order:
   - Stage 0 extractor/unescape
   - Tiered tolerant decode (per tolerant_level)
   - DeterministicRepairProcessor (syntactic JSON fix)
   - SmartTypeCoercionProcessor (semantic coercion + anyOf/oneOf disambiguation)
   User-defined processors run as configured.
5. Run validators as today. If precheck is enabled, run `PlanCheckValidator` as a preflight prior to generation or as an early short pass when plan text is available; on fail, continue within the attempt using the validator feedback as retry guidance.
6. On success, strict-validate final output shape; on failure, proceed with standard retry feedback. Never swallow control-flow exceptions.
7. Emit AROS trace events throughout. Merge context only on success.

5) Configuration (Centralized)

5.1 Global defaults (flujo.toml, accessed via ConfigManager)
```
[aros]
enabled = true                      # master switch; defaults to true but does nothing unless per-step opts are set
structured_output_default = "auto"  # off | auto | openai_json
enable_aop_default = "minimal"      # off | minimal | full
coercion_tolerant_level = 0          # 0=off, 1=json5, 2=json-repair (logs patch previews)
max_unescape_depth = 2               # depth limit for Stage 0b
anyof_strategy = "first-pass"        # branch selection strategy
enable_reasoning_precheck = false
```

5.2 Step-level toggles (DSL/YAML excerpt)
```
agents:
  my_agent_step:
    processing:
      structured_output: auto          # off | auto | openai_json | outlines | xgrammar
      aop: minimal                     # off | minimal | full
      coercion:
        tolerant_level: 0              # 0=off, 1=json5, 2=json-repair
        allow:
          integer: ["str->int"]
          number:  ["str->float"]
          boolean: ["str->bool"]
        max_unescape_depth: 2
        anyof_strategy: "first-pass"
      reasoning_precheck:
        enabled: false
        validator_agent: agents.plan_checker   # optional
        max_tokens: 200                        # budget cap for the precheck
        score_threshold: 0.7                   # optional scoring gate
        required_context_keys: ["initial_input"]
        inject_feedback: "prepend"             # prepend | context_key | (unset)
        retry_guidance_prefix: "Guidance: "    # for prepend mode
        context_feedback_key: "_aros_retry_guidance"  # for context_key mode
```

Notes:
- “minimal” AOP auto-injects Stage 0 extractor/unescape + fast JSON parse and limited whitelisted conversions; “full” also enables tolerant decoders and advanced anyOf/oneOf handling.
- User processors can still be provided explicitly; AROS injections prepend in a safe order.

Example schema-aware coercion (AOP full)
```
processing:
  aop: full
  coercion:
    allow:
      integer: ["str->int"]
      number: ["str->float"]
      boolean: ["str->bool"]
  schema:
    type: object
    required: ["count"]
    properties:
      count: { type: integer }
```

Example structured outputs via pydantic‑ai (OpenAI JSON)
```
processing:
  structured_output: openai_json
  schema:
    type: object
    properties:
      ok: { type: boolean }
```

6) Observability & Health Check

- Trace events (examples, aligned with Trace Contract):
  - `grammar.applied` `{ mode: "openai_json|outlines|xgrammar", schema_hash }`
  - `aros.soe.enabled|skipped`
  - `output.coercion.attempt` `{ stage: "syntactic|semantic|custom|extract|tolerant", reason, expected_type, actual_type }`
  - `output.coercion.success` `{ stage, transforms: ["json.loads","str->int"], branch_index? }`
  - `output.coercion.fail` `{ stage, error_preview }`
  - `reasoning.validation` `{ result: "pass|fail", score? }`
- CLI `flujo dev health-check`:
  - Reads spans/traces; aggregates AROS events per step/agent/model.
  - Outputs top-N offenders and recommendations (enable SOE, add custom processor, adjust prompts).
  - No schema changes required in SQLite backend.

7) Testing Strategy

7.1 Unit tests
- SOE/OpenAI mapping emits correct options via wrapper/factory and logs `aros.soe.enabled`.
- Stage 0 extractor + bounded unescape: property-based tests (Hypothesis) across fenced/double-encoded inputs.
- DeterministicRepairProcessor regression tests (existing) + integration with AOP flagging.
- SmartTypeCoercionProcessor: success cases (str→int/bool/float, T→[T]), nested objects, anyOf/oneOf disambiguation, and negative cases (ambiguous/unsafe).
- PlanCheckValidator: parsing optional plan snippets; pass/fail propagation as retry feedback.

7.2 Integration tests
- Happy-path agent bypasses AOP; trace shows no AROS interventions.
- Mixed/fenced/double-encoded JSON is extracted/unescaped and parsed; final object validated; events recorded.
- SOE enabled with OpenAI model adds response_format; trace shows `aros.soe.enabled`.
- Precheck fail produces retry feedback; no control-flow exceptions are swallowed; final retry path executes.
- Global disable `[aros].enabled=false` preserves legacy behavior.

7.3 Performance tests
- Overhead of AOP on correct outputs < 5 ms.
- Precheck adds bounded tokens/time; disabled by default.
 - Microbenchmarks: compare `json` vs `orjson` vs `msgspec` typed decode.
 - Shadow mode: monitor-only runs that log AOP decisions without mutation.

8) Rollout Plan

Phase 1 (this spec):
- SOE for OpenAI JSON via wrapper/factory integration; optional and auto-detected.
- AOP via existing processor pipeline: auto-inject Stage 0 extractor/unescape + fast parse + DeterministicRepairProcessor (minimal); add SmartTypeCoercionProcessor and tolerant decoders for “full”.
- Reasoning precheck: local checklist + optional validator/scorer; off by default.
- Trace-only health reporting; `flujo dev health-check` reads traces.

Phase 2:
- Add other providers’ native JSON modes.
- Consider regex/CFG grammars (Outlines/XGrammar) as experimental opt-ins (off by default) with strict tests.
- Expand health-check analytics (time-sliced trends, per-pipeline summarization).
 - Optional correction ledger table with the proposed schema fields if trace queries are insufficient.

9) Risks & Mitigations

- Over-coercion hiding real bugs → Keep final strict validation; “minimal” default safe conversions only; make tolerant decoders and broader coercions opt-in via `tolerant_level`.
- Provider coupling for SOE → Contain in wrapper/factory; policy reads intent only.
- Precheck cost/latency → Off by default; small token caps; skip when no plan available.
- Storage creep → Use traces first; add a dedicated ledger only if query/retention needs justify it later.

10) Backward Compatibility

- Defaults do not change behavior unless explicitly enabled per step or via sensible global defaults.
- No DB schema changes in Phase 1.
- All control-flow and idempotency guarantees preserved.

11) Change Log (v1.1 vs v1.0)

- Replaced generic “grammar enforcement” with provider-native SOE (OpenAI JSON in Phase 1).
 - Implemented AOP as standard processors; add Stage 0 extractor/unescape; reuse DeterministicRepair; add SmartTypeCoercionProcessor; add optional tolerant decoders.
 - Reasoning validation made optional, lightweight, and non-intrusive; no forced prompt markers.
 - Dropped new “correction_ledger” table; use trace events + health-check CLI.
 - Tightened exception and context semantics to match Team Guide.

13) Implementation Progress (live)

- AOP pipeline
  - Stage 0 extractor/unescape: Completed and tested (nested/fenced, double-encoded).
  - Tiered tolerant decode: Completed (orjson/json; json5/json-repair opt-ins) + tests.
  - DeterministicRepair ordering: Integrated.
  - SmartTypeCoercion: Completed (schema-aware, anyOf/oneOf; allowlisted coercions) + tests.
  - Schema auto-derivation from agent output type: Implemented.

- Structured Output Enforcement (SOE)
  - Wrapper-based (pydantic‑ai): Implemented best‑effort enabling for JSON object/JSON Schema when explicitly enabled per step; policies only signal intent. Defaults guarded.
  - Experimental adapters (Outlines/XGrammar): Telemetry-only stubs in place; no runtime effect. Deeper wiring is Phase 2.
  - Tests: Policy-path unit tests ensure `response_format` is attached when `processing.structured_output` is set; safe no-op when agent doesn't accept kwargs.
  - Provider-profile coverage: factory creates OpenAI Responses model for GPT‑5 family; wrapper tests ensure response_format hinting without network; policy emits `grammar.applied` for GPT‑5 profile under `auto`.
  - Schema hash: `grammar.applied` events include `schema_hash` (sha256 of JSON schema) for openai_json and outlines/xgrammar in both initial and retry paths.

- Reasoning Precheck
  - Local checklist pre-gate: Implemented (required_context_keys) with aros.reasoning.precheck.{pass|fail} events.
  - Validator agent + scoring: Implemented (telemetry-only) with optional score_threshold and opt‑in feedback injection (prepend/context_key).
  - Consensus gate: Implemented (telemetry-only) using N‑sample Jaccard; does not alter execution.

- Observability & Health
  - Trace events and on-span aggregation: Implemented (including aros.model_id and transforms).
  - CLI health-check: Implemented with since-hours, top steps/models/transforms, basic + targeted recommendations (top step/model hints), and JSON/CSV export. Added N-bucket trends via `--trend-buckets` with per-bucket coercion counts and per-stage breakdowns. Added overall per-step and per-model stage distributions; extended trend buckets to include per-step/per-model stage distributions as well. Stage-aware recommendations (tolerant/semantic/extract) and trend-based hints (top rising step/model and stage-specific rises) now included. Added `--step/--model` filters and JSON export metadata (`version`, `generated_at`), plus stdout export with `--output -`.

- Config & DSL
  - [aros] defaults wired via ConfigManager; defaults OFF by design.
  - Step-level processing accepted via step.meta["processing"].
  - Validation: ProcessingConfigModel validates structured_output/aop/coercion/reasoning_precheck; invalid configs raise BlueprintError.
  - Grammar enforcement flag: `processing.enforce_grammar` accepted (boolean).

- Tests & Golden
  - Unit tests for AOP (Stage 0, tolerant decode, SmartCoercion) added and passing.
  - SOE policy-path tests (openai_json, auto=JSON, outlines, xgrammar) and wrapper schema-name propagation.
  - Health-check CLI tests for JSON export bucket breakdowns, stage-aware recs, and edge cases (no runs, no since-hours).
  - Integration test: grammar.applied aggregates via TraceManager hook across a mock pipeline step.
  - Provider-profile tests: GPT‑5 (OpenAI Responses model) and GPT‑4o wrappers accept structured_output hints without network; policy emits grammar.applied in auto mode.
  - Precheck tests: skipped when no plan; max_tokens forwarded to validator.
  - Grammar enforcement tests: outlines enforcement pass/fail on matching/non-matching outputs.
  - anyOf coercion test: branch selection recorded with branch_choices.
  - Golden traces validated (defaults off prevent deltas).

Next up (Phase 2)
- SOE deeper integration via pydantic‑ai across supported models (JSON Schema), plus unit mocks; Outlines/XGrammar wiring (opt-in).
- Health‑check trend windows and richer, per-step/model recommendations; extended exports.
- Optional typed fast path (feature‑gated) and microbenchmarks.
- Additional negative/edge tests for processing validation and docs/examples polish.

12) Implementation Checklist

- Foundations: setup and config
  - Define `[aros]` in `flujo.toml` and wire via `ConfigManager` (`structured_output_default`, `enable_aop_default`, `coercion_tolerant_level`, `max_unescape_depth`, `anyof_strategy`, `enable_reasoning_precheck`).
  - Add step-level `processing` keys (`structured_output`, `aop`, `coercion.*`, `reasoning_precheck.*`) to DSL schema/validation.
  - Done when: Config loads without direct env reads; defaults applied; mypy strict passes.

- Structured Output Enforcement (SOE)
  - Add provider capability probe in agent factories/wrappers for OpenAI JSON mode.
  - Map step `processing.structured_output` to wrapper options; keep policy side declarative.
  - Emit `grammar.applied` and `aros.soe.{enabled|skipped}` trace events.
  - Tests: unit (option mapping), integration (payload shows JSON mode), negative (unsupported → skipped).
  - Done when: opt‑in SOE works; fallbacks safe; no policy/provider coupling leaks.

- AOP: Stage 0 extractor/unescape
  - Implement streaming brace/bracket matcher for largest balanced JSON/array; prefer schema root.
  - Add bounded double‑unescape (depth ≤ `max_unescape_depth`, size/time guards).
  - Integrate as auto‑injected processor when `aop != off`.
  - Emit `output.coercion.attempt|success|fail` with stage `extract`.
  - Tests: property‑based (fenced/double‑encoded), size/timeout guards.
  - Done when: plain `json`/`orjson` succeeds more often after Stage 0.

- AOP: Tiered tolerant decode (opt‑in)
  - Default parser: `orjson.loads` (fallback to `json.loads`).
  - If `tolerant_level >= 1`: try `json5/pyjson5`.
  - If `tolerant_level >= 2`: try `json-repair`; log patch preview (before/after snippet).
  - Emit events with stage `tolerant` and transformations.
  - Tests: unit for each tier; security: off by default; logs present.
  - Done when: tiers activate only by config; strict validation still final gate.

- AOP: DeterministicRepair integration
  - Reuse `DeterministicRepairProcessor` late in chain for JSON expectations.
  - Ensure ordering: Stage 0 → tolerant decode → DeterministicRepair → Smart Coercion.
  - Emit stage `syntactic` events.
  - Tests: regressions on malformed/near‑valid JSON.
  - Done when: repair never masks large semantic diffs; strict validation follows.

- AOP: Smart Type Coercion
  - Implement `SmartTypeCoercionProcessor` (schema‑aware, Ajv‑style rules).
  - Whitelist conversions per type; fail fast on ambiguity.
  - Support `anyOf/oneOf` branch trial; record chosen index.
  - Emit stage `semantic` events with `transforms` and `branch_index`.
  - Tests: unit (success/negative), nested objects, arrays, branch selection.
  - Done when: unambiguous coercions pass; ambiguous cases fail with precise feedback.

- Optional: Typed decode fast path
  - Feature‑gate direct decode to `msgspec.Struct` or Pydantic v2 `TypeAdapter`.
  - Benchmarks: small/medium/large payloads vs `orjson`.
  - Done when: measurable speedups; remains optional; no hard deps.

- Reasoning Precheck
  - Add local schema checklist for required fields/shape; cheap and local.
  - Optional `PlanCheckValidator` + scoring plugin + consensus gate; integrate in attempt loop.
  - Config: `reasoning_precheck.enabled`, `max_tokens`, `score_threshold`.
  - Emit `aros.reasoning.precheck.{pass|fail|skipped}` with scores.
  - Tests: unit (precheck logic), integration (retry feedback path).
  - Done when: off by default; no control‑flow exceptions converted; retries use feedback.

- Observability (Trace Contract)
  - Add standardized events:
    - `grammar.applied` {mode, schema_hash}
    - `output.coercion.{attempt|success|fail}` {stage, reason, transforms, branch_index?}
    - `aros.soe.{enabled|skipped}`, `reasoning.validation` {result, score?}
  - Wire event emission in processors/wrapper/policy at each stage.
  - Tests: golden traces; attributes stable; spans untouched on failures.
  - Done when: events appear with correct attributes across scenarios.

- Health‑Check CLI
  - Implement `flujo dev health-check`:
    - Aggregate AROS events from traces/spans.
    - Report: top steps by coercions/100 runs, SOE skips by model, precheck fail rates, recommendations.
  - Support filters (pipeline/step/time span).
  - Tests: unit (aggregators), smoke (end‑to‑end on sample runs).
  - Done when: actionable output; no new tables required (Phase 1).

- Policy & Idempotency
  - Ensure AOP injections run inside per‑attempt isolated context.
  - Only merge context on success; never catch/rewrite control‑flow exceptions.
  - Tests: retries do not “poison” context; fallbacks unchanged.
  - Done when: matches Team Guide invariants.

- Configuration & DSL
  - Document global and per‑step knobs; validate DSL schema.
  - Backward compatibility: defaults keep legacy behavior when disabled.
  - Tests: config precedence (global vs step), invalid config errors.
  - Done when: mypy strict; docs reflect accurate defaults.

- Dependencies & Packaging
  - Make tolerant decoders optional extras (e.g., `flujo[aop-extra]`).
  - Guard imports; degrade gracefully when not installed.
  - Update `pyproject.toml` and `Makefile` targets if needed.
  - Done when: core installs cleanly; extras opt‑in.

- Testing & Benchmarks
  - Unit: Stage 0, tolerant decoders, Smart Coercion, SOE mapping, precheck.
  - Integration: AOP happy/repair; SOE on/off; precheck retry; global disable.
  - Performance: overhead < 5ms on correct outputs; microbench comparisons; shadow mode toggle.
  - Done when: `make all` passes; coverage maintained.

- Docs & Examples
  - Update spec links, contributor guide sections, and usage docs.
  - Add example YAML showing `structured_output`, `aop`, `coercion.*`, `reasoning_precheck`.
  - Provide migration notes and “safe defaults” guidance.
  - Done when: examples run; docs consistent.

- Rollout & Safety
  - Phase 1 flags default: `structured_output=auto`, `aop=minimal`, `tolerant_level=0`, precheck off.
  - Add monitoring period (shadow mode) before enabling broader coercions.
  - Done when: feature flags verified; monitoring reports reviewed.

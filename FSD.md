# Flujo Validate — Functional Specification and Test Plan

This document specifies the behavior, rules, CLI interface, architecture, and tests for the `flujo validate` command. The validation system provides proactive, actionable feedback on Flujo pipelines defined in YAML or Python, catching structural, type, templating, import, and orchestration issues before runtime.

---

## 1. Goals & Non‑Goals

- Goals:
  - Provide fast, deterministic validation of pipelines prior to `flujo run`.
  - Catch common authoring mistakes (missing agents, type mismatches, import issues, template misuses, state machine misconfigurations) with precise locations and actionable suggestions.
  - Offer multiple output formats (human‑readable, JSON, SARIF) suitable for local use and CI.
  - Allow configuration, baselining, and suppression for teams adopting validation incrementally.

- Non‑Goals:
  - Execute or contact external providers (no network or agent calls during validate).
  - Replace runtime checks; some issues remain inherently runtime (e.g., dynamic data paths).

---

## 2. User Stories

- As a developer, I want `flujo validate pipeline.yaml` to catch template patterns that will certainly fail (e.g., `{{ previous_step.output }}`) and tell me exactly how to fix them.
- As a reviewer/CI pipeline, I want a machine‑readable report with rule IDs, severities, step references, and file/line when available.
- As a maintainer, I want rule configuration and suppression so I can stage adoption without blocking merges.

---

## 3. CLI Interface

Command: `flujo validate [PATH]`

- Arguments
  - `PATH` (optional): pipeline file (`.yaml`, `.yml`, or Python module path). Defaults to project pipeline (`pipeline.yaml`).

- Options
  - `--strict/--no-strict` (default `--strict`): exit non‑zero when errors found.
  - `--format {text,json,sarif}` (default `text`): output formatter.
  - `--explain`: include brief “why it matters” guidance per finding (text/json). In `text` format, rule IDs may include clickable URLs to docs.
  - `--fail-on-warn`: treat warnings as errors (non‑zero exit code).
  - `--fix`: apply safe auto‑fixes (e.g., V‑T1). Never modifies files in CI unless `--yes` is also provided.
  - `--yes`: non‑interactive confirm for `--fix` (intended for CI). Without `--yes` and running in a TTY, an interactive prompt summarizes fixable items and asks for confirmation.
  - `--baseline FILE`: path to previous validation report to compare; prints deltas.
  - `--update-baseline`: write current report to `--baseline` path.
  - `--rules PROFILE|FILE`: named profile from `flujo.toml` (e.g., `strict`, `ci`) or path to YAML/TOML rules. Profiles are configured under `[validation.profiles]`.
  - `--imports/--no-imports` (default `--imports`): when enabled, recursively validate imported blueprints declared under `imports:` with proper base_dir resolution. Findings from child blueprints are aggregated into the parent report with alias/file context. Disabling can speed up local iteration on large import graphs.

- Exit Codes
  - 0: valid (or only warnings with `--no-strict` and `--fail-on-warn` not set)
  - 2: validation failures (errors or warnings with `--fail-on-warn`)
  - 64/70: CLI usage/config/runtime errors (existing semantics)

---

## 4. Validation Model & Report Schema

Core model (existing):

- `ValidationFinding`:
  - `rule_id: str` (e.g., `V-A1`)
  - `severity: "error"|"warning"`
  - `message: str`
  - `step_name: Optional[str]`
  - `suggestion: Optional[str]`
  - Extended fields (new):
    - `location_path: Optional[str]` (e.g., `steps[3].input`)
    - `file: Optional[str]`
    - `line: Optional[int]`, `column: Optional[int]` (if YAML parser provides)

- `ValidationReport`:
  - `errors: list[ValidationFinding]`
  - `warnings: list[ValidationFinding]`
  - `is_valid: bool` (derived: `len(errors) == 0`)

Formatters:
- `text`: grouped by severity with file/line if available.
- `json`: full object graph for CI.
- `sarif`: SARIF v2.1 (minimal), mapping rule IDs to SARIF results.

---

## 5. Rule Set (Existing + New)

Existing rules (already implemented in `Pipeline.validate_graph()`):
- V-A1: Missing agent on simple steps.
- V-A2: Type mismatch between output of previous step and input of next.
- V-A3: Same Step instance reused (side‑effects risk).
- V-A4-ERR: Could not analyze agent signature (warning).
- V-A5: Unused output (previous step produces output not consumed by next and does not update context).
- V-F1: Fallback step input incompatible.
- V-P1 / V-P1-W: ParallelStep potential context merge conflicts (default strategy, no mapping).

New template rules:
- V-T1: `previous_step.output` misuse
  - Trigger: `{{ previous_step.output ... }}` found in templated input/output.
  - Message: `previous_step` is a raw value and has no `.output`.
  - Suggestion: Prefer the explicit step proxy: `{{ steps.<previous_step_name>.output | tojson }}`. For simple linear flows, the previous step name is the prior step; otherwise, use `{{ previous_step | tojson }}`.
  - Location: template field path.

- V-T2: `this` misuse
  - Trigger: `{{ this }}` outside of a map body.
  - Suggestion: Use in `map.body` only or bind a variable.

- V-T3: Unknown/disabled filters
  - Trigger: filter not in `[settings.enabled_template_filters]` or default allow list.
  - Suggestion: Add to `flujo.toml` or remove/misspelling fix.

- V-T4: Unknown step proxy name
  - Trigger: `{{ steps.<name>... }}` where `<name>` not in prior steps.
  - Suggestion: Correct the step name or move the reference after the producing step.

- V-T5: Missing field on typed previous output (best‑effort)
  - Trigger: prior output type is a Pydantic model with fields; template references missing field.
  - Severity: warning (static inference can be conservative).

- V-T6: JSON decode trap
  - Trigger: known JSON‑expecting locations (e.g., `loop.output_template`) are templated with a non‑JSON literal without `tojson`.
  - Suggestion: wrap value with `tojson` or ensure it renders valid JSON.

Schema & types rules:
- V-S1: JSON Schema sanity
  - `required` keys must exist in `properties` (object schemas).
  - `array` must have `items` (warn if missing).
  - Unknown `type` values -> warning.

- V-S2: Response‐format mismatch
  - If `model_settings.response_format == json_object` and step final rendering stringifies raw content, warn to use `tojson` or consume structure.

- V-S3: Primitive wrapper awareness
  - For `type: string|number|boolean` schemas in agents, warn that output model is `{value: ...}` to avoid later confusion.

Imports & composition rules:
- V-I1: Import existence
  - Imported YAML paths resolve from `base_dir`. Error if not found or unreadable.

- V-I2: ImportStep outputs mapping sanity
  - Validate dotted paths: `parent` usually under `scratchpad` or known context fields; warn on obviously invalid roots.

- V-I3: Cyclic imports
  - Detect simple cycles among imported YAMLs and error.

- V-I4: Child blueprint validation aggregation
  - When `--imports` is enabled, load each imported YAML and run full validation on the child pipeline (without agent execution). Aggregate child findings into the parent report, annotating `file`, `alias`, and `location_path` (e.g., `imports.clarification::steps[0].agent`).

- V-I5: Import input projection coherence
  - Check `ImportStep.config.input_to` vs likely child expectations:
    - `initial_prompt` used while the child's first concrete step expects a dict/object → warning to consider `scratchpad` or an adapter step.
    - `scratchpad` projection with a scalar `input_scratchpad_key` while the child’s first step expects a string → informational warning to confirm intended usage.
  - Heuristic, severity `warning`. Documented limitations.

- V-I6: Inherit context consistency
  - If `inherit_context=false` and outputs map attempts to read fields other than `scratchpad` from the child (e.g., `command_log`), warn: those fields may be uninitialized depending on the child’s behavior. Suggest enabling `inherit_context` or scoping mappings to `scratchpad`.

Orchestration rules:
- V-SM1: StateMachine transitions validity
  - `from`/`to` states must exist; compile `when` expressions; warn unreachable states and absence of path to `end_states`.

- V-L1: Loop exit coverage
  - Heuristic: warn if loop body cannot set exit condition or exit mapping is absent.

- V-P2: Parallel explicit conflicts
  - When branches with explicit `outputs` map to the same parent keys without merge strategy, warn.

Context & persistence rules:
- V-C1: updates_context without mergeable output
  - If `updates_context: true` but output is non‑dict and not a PipelineResult, warn.
  - Escalation: If the next step also does not consume this output (per V‑A5) and no `outputs` mapping is provided, escalate severity to `error` (output would be effectively dropped).

- V-C2: scratchpad shape conflicts
  - Warn when assigning non‑dict to `scratchpad`.

- V-C3: large context artifacts (performance)
  - Warn if template embeds extremely large strings/constants (threshold configurable).

Agent/Provider rules:
- V-A6: Unknown agent id/skill in `agent.id` or `uses` (Python object path)
  - Resolve importable objects without executing them (no side effects). Error if unresolved.

- V-A7: Invalid `max_retries`/`timeout` coercion
  - Warn if values cannot be coerced to expected types.

- V-A8: Structured output with non‑JSON response mode
  - Provider‑agnostic warning when obvious mismatch is declared.

Parallel input uniformity (new):
- V-P3: Parallel branch input uniformity
  - Trigger: In a `ParallelStep`, the first step in branches expects different input types (heterogeneous typing).
  - Severity: warning.
  - Message: Branches in `ParallelStep '{step.name}'` expect different input types; the same input is passed to all branches.
  - Suggestion: Ensure branches can handle the same input type or add adapter steps per branch.

Control‑flow safety (new):
- V-CF1: Unconditional infinite loop
  - Trigger: `LoopStep` with missing/constant‑false exit condition and no/very large `max_loops`.
  - Severity: error.
  - Suggestion: Provide an `exit_condition` or set a reasonable `max_loops`.

Rule configuration:
- Source: `flujo.toml` under `[validation]`, optional `--rules` file.
- Controls per rule or category: severity (`error|warning|off`).
- Inline suppression: YAML comments `# flujo: ignore V-T1,V-SM1` on the same line or step block.

---

## 6. Architecture & Components

- `Pipeline.validate_graph()` remains the orchestration entry. It:
  - Runs existing graph/type checks.
  - Delegates to pluggable linters:
    - `TemplateLinter`
    - `SchemaLinter`
    - `ImportLinter`
    - `OrchestrationLinter`
    - `ContextLinter`
    - `AgentLinter`
  - Aggregates `ValidationFinding`s into a single `ValidationReport`.

- Linters are pure analyzers with no side effects and accept:
  - `pipeline` (compiled DSL), optional `blueprint` model, `source_map` (step → YAML location), and `settings`.

- YAML source mapping (best‑effort):
  - Prefer `ruamel.yaml` when available to capture line/column for nodes. Fallback to PyYAML when not installed.
  - Enhance loader to track step array indices and fields; store as `step.meta["_yaml_loc"] = {file,line,column,path}`.
  - Linters use this to annotate `file/line/column/location_path`.

- Configuration: Access all settings and rule profiles via `infra.config_manager` (Team Guide: Centralized Configuration). The validation logic must not read `flujo.toml` directly.

- Side‑effects: Validation performs no network calls and no agent execution (Team Guide: Exception Safety & Separation). `--fix` only touches files when explicitly requested and never during run‑time policies.

- Import validation execution model:
  - Child blueprints are loaded via the same loader (`load_pipeline_blueprint_from_yaml`) with `base_dir` set to the directory of the parent YAML. Import resolution honors sandboxing and prevents path traversal. No agents are executed; only graph compilation and `validate_graph()` run for children.
  - Use a visited set keyed by the normalized absolute path to detect and prevent cycles; report V‑I3.
  - Cache compiled child `Pipeline` per path to avoid repeated work if referenced by multiple parents.

- CLI wires `--format` and report printing; supports baselining by reading/writing JSON reports.

---

## 7. Algorithms (Highlights)

- Template scanning (V-T1..T4):
  - For each templated field (e.g., `Step.meta["templated_input"]`, loop/map templates), collect `{{ ... }}` tokens using the same parser used by `AdvancedPromptFormatter`.
  - Identify misuse via simple regex/AST:
    - `previous_step\.output(\b|\s*\|)` → V-T1
    - `\bthis\b` outside map bodies → V-T2
    - Filters: extract `| filter` segments and validate membership against allow‑list → V-T3
    - `steps\.(?P<name>[A-Za-z0-9_]+)`; if name not in prior steps set → V-T4

- Prior type field existence (V-T5):
  - If `prev_out_type` is a Pydantic model, parse `previous_step\.(\w+)` attribute refs and check against `model_fields`.

- Import checks (V-I1..I3):
  - Resolve relative paths from `base_dir`.
  - Build a graph of imports → detect cycles with DFS.
  - For each child, run validation recursively (when `--imports` is enabled) and prefix findings with alias and file.

- StateMachine transitions (V-SM1):
  - Validate `from` and `to` against `states` and `end_states`.
  - Attempt reachability by exploring transitions graph; if no path from `start_state` to any end, warn.

- Parallel branch input uniformity (V-P3):
  - Inspect each branch’s first step input type; warn when heterogeneous.

- Infinite loop heuristic (V-CF1):
  - If `exit_condition` is missing or trivially false and `max_loops` is None or above a configurable threshold, flag error.

---

## 8. Performance & Safety

- No network or provider execution; only import resolution and file IO (for YAML).
- O(N) over steps/templates; state machine/parallel reachability linear in graph size.
- Short‑circuit linters if rule category disabled.

---

## 9. Telemetry (Optional)

- Emit aggregated counts per rule and per severity when `FLUJO_CLI_TELEMETRY=1`.
 - Optionally record which fixes were applied with `--fix` (counts only; no PII/code).

---

## 10. Test Plan

### 10.1 Unit Tests — Core Models & Formatters

- `ValidationFinding`/`ValidationReport`
  - Create, serialize to JSON; `is_valid` property behavior.
  - Include extended fields and ensure formatters include them.

- Formatters
  - `text`: groupings, severity headers, step/file/line formatting.
  - `json`: round‑trip.
  - `sarif`: minimal SARIF structure with ruleId mapping and message.

### 10.2 Unit Tests — Linters

- TemplateLinter
  - V-T1: Detect `{{ previous_step.output }}` in step input.
  - V-T2: `{{ this }}` outside map body → warning with location.
  - V-T3: Unknown filter `| foo` when not enabled.
  - V-T4: `{{ steps.Missing.output }}` when no prior step `Missing`.
  - V-T5: With known prior model, reference missing field → warning.
  - V-T6: Output template that produces non‑JSON where JSON expected → warning.

- SchemaLinter
  - V-S1: `required` outside `properties`; array with no `items` → warnings; unknown type → warning.
  - V-S2: `response_format=json_object` with obvious stringification downstream → warning.
  - V-S3: `type: string` → adds awareness warning.

- ImportLinter
  - V-I1: Missing import file path → error with `file` resolution.
  - V-I2: Outputs mapping to invalid root (e.g., `parent: badroot.value`) → warning.
  - V-I3: Cycle in imported YAMLs → error.
  - V-I4: Aggregated child findings: child pipeline missing agent/error surfaces under parent with alias and file.
  - V-I5: Input projection coherence heuristics → warnings with suggestions.
  - V-I6: Inherit context inconsistency → warning.

- OrchestrationLinter
  - V-SM1: Invalid transitions, unreachable states, no path to end → errors/warnings.
  - V-L1: Loop without exit heuristic → warning.
  - V-P2: Parallel explicit outputs conflict → warning.
  - V-P3: Parallel branch input uniformity → warning.
  - V-CF1: Unconditional infinite loop → error.

- ContextLinter
  - V-C1: `updates_context: true` with scalar output → warning.
  - V-C2: Assign non‑dict to `scratchpad` → warning.
  - V-C3: Extremely large literals in templates → warning; threshold configurable.

- AgentLinter
  - V-A6: Unknown `agent.id`/`uses` import path → error.
  - V-A7: Non‑int `max_retries`, invalid `timeout` → warning.
  - V-A8: Structured output with non‑JSON response mode mismatch → warning.

### 10.3 Unit Tests — Pipeline.validate_graph() Orchestration

- Aggregation behavior: multiple linters contribute findings; report combines, preserves ordering by step order, then rule id.
- Config controls: disable a rule → it doesn’t appear; severity overrides applied.
- Inline suppression: `# flujo: ignore V-T1` removes that finding for the specific location.
 - Suppression globs: `# flujo: ignore V-T*` suppresses all template rules.
 - Profiles: `--rules=strict` applies severities defined in `flujo.toml` under `[validation.profiles.strict]`.

### 10.4 Integration Tests — CLI

- `flujo validate` on a valid YAML → exit 0, no errors.
- `--strict` default: errors cause exit 2.
- `--no-strict`: errors still printed, exit 0.
- `--fail-on-warn`: warnings cause exit 2.
- `--format json/sarif`: output parses and contains expected rule IDs and locations.
- Baseline: with `--baseline` and `--update-baseline`, compare previous vs current; print added/removed findings.
- `--fix` (interactive): in TTY, shows summary and prompts; with `--yes`, applies changes non‑interactively; verify only fixable rules are modified (e.g., V‑T1).
 - Imports: with `--imports` (default), validation of a parent with `imports:` includes child errors; with `--no-imports`, only parent is validated.

### 10.5 Regression Tests

- Template null symptom: A YAML with `{{ previous_step.output | tojson }}` triggers V-T1 and suggests `previous_step | tojson`.
- Import existence: missing `imports.alias` file → V-I1 error with base_dir hint.
- StateMachine: transitions referencing unknown states → V-SM1.

---

## 11. Rollout Plan

- Phase 1 (MVP): Implement TemplateLinter V‑T1..T4, SchemaLinter V‑S1, ImportLinter V‑I1, OrchestrationLinter V‑SM1, JSON/SARIF formatters, `--explain`.
- Phase 1.5: Add clickable rule URLs, suppression globs, V‑P3, refined V‑T1/V‑C1, profiles support via `--rules` and `flujo.toml`.
- Phase 2: Add V‑T5..T6, V‑I2..I3, V‑P2, context/agent rules, baseline/suppression with deltas, `--fix` for V‑T1.
- Phase 3: Optional `ruamel.yaml` source mapping with line/column, telemetry, expand safe fixers beyond V‑T1.

---

## 12. Backward Compatibility

- Existing `validate_graph()` rules remain unchanged; new rules default to `warning` except safety‑critical ones (import existence, missing agent, type mismatch) which remain `error`.
- CLI maintains existing options; new options are additive.
 - Team Guide alignment: Validation uses `ConfigManager` (no direct file/env reads in domain logic), performs no agent execution, and maintains clear separation from runtime policy execution.

---

## 13. Open Questions

- Should `previous_step.output` be auto‑fixable with `--fix`? (Proposal: yes, safe rewrite.)
- SARIF rule metadata: maintain a centralized rule catalog for consistency across tools.

---

## 14. Feature‑Level Acceptance & Test Requirements

This section defines acceptance criteria and explicit tests for each feature/rule. All tests must pass under `make test-fast` and `make test` and adhere to team testing standards (markers, serial/slow as needed).

CLI & Framework Features
- --fix/--yes
  - Acceptance: Applies only safe fixes (initially V‑T1) with file backups; interactive prompt in TTY; `--yes` applies without prompt. No changes in `--no-strict` dry validations.
  - Unit: Fixer registry selects correct fixer; V‑T1 rewrite correct and idempotent; backup written then cleaned.
  - Integration: Run `flujo validate --fix` on sample YAML; confirm rewrite diff; with `--yes`, no prompt; with CI env, no write unless `--yes`.
  - Regression: Ensure non‑fixable rules are not modified; ensure multiple fixable instances handled.

- --rules profiles
  - Acceptance: Named profiles load from `flujo.toml` via ConfigManager; CLI profile overrides defaults; per‑rule severity honored.
  - Unit: Profile parser precedence; wildcard/glob matching of rule IDs.
  - Integration: `--rules=strict` produces different severities vs default.

- --imports recursive validation
  - Acceptance: Child findings aggregated with alias/file; cycle detection; caching avoids re‑validation.
  - Unit: Cycle graph detection; path resolution with base_dir; multi‑parent caching.
  - Integration: Parent YAML with imports; verify child errors surface; `--no-imports` excludes children.

- --format sarif / json / text, --explain
  - Acceptance: Output renders; SARIF objects valid minimal schema; text includes clickable rule URLs when available; `--explain` adds guidance.
  - Unit: Formatter snapshot tests; SARIF schema shape; explain toggling.
  - Integration: CI‑friendly JSON processable by jq.

- Baseline & deltas
  - Acceptance: Loads baseline; prints added/removed changes; `--update-baseline` writes file.
  - Unit: Delta calculator; file IO safe‑guarding.
  - Integration: Two runs with changing YAML produce expected deltas.

- Suppression globs
  - Acceptance: `# flujo: ignore V-T*` suppresses all template rules at location scope.
  - Unit: Glob → regex; local scoping rules.

Template Rules (V‑T1..V‑T6)
- V‑T1 previous_step.output
  - Acceptance: Finding created with location; suggestion prefers `steps.<prev>.output | tojson`.
  - Unit: Detector regex/AST; suggestion formatting.
  - Integration: Sample pipeline triggers finding; `--fix` rewrites when enabled.

- V‑T2 this misuse; V‑T3 unknown filters; V‑T4 unknown step proxy; V‑T5 missing field; V‑T6 JSON trap
  - Acceptance: Each rule triggers on designed cases with correct severity.
  - Unit: Dedicated detectors per case.
  - Integration: YAML samples covering each rule.

Schema Rules (V‑S1..V‑S3)
- Acceptance: Sanity checks produce warnings; primitive wrapper awareness message present.
- Unit: JSON schema parsing; property/required checks.
- Integration: Agent spec with minimal schemas.

Import Rules (V‑I1..V‑I6)
- Acceptance: Existence errors, cycle errors, mapping sanity warnings, child aggregation (V‑I4), input projection coherence (V‑I5), inherit context consistency (V‑I6).
- Unit: Path resolution, cycle detection, mapping validator.
- Integration: Parent + children fixture set; verify aggregated findings with `alias` context.

Orchestration Rules (V‑SM1, V‑L1, V‑P2, V‑P3, V‑CF1)
- Acceptance: Transitions validity/unreachable; loop exit heuristic; parallel conflicts and heterogeneous inputs; unconditional infinite loop flagged as error.
- Unit: Graph reachability; branch type comparison; loop heuristic.
- Integration: Minimal pipelines per rule triggering correct outputs.

Context & Persistence (V‑C1..V‑C3)
- Acceptance: Non‑mergeable output warnings; escalation to error when output otherwise dropped; scratchpad shape conflicts; large constant warnings.
- Unit: Severity escalation logic (with/without next step consumption).
- Integration: Pipelines demonstrating each case.

Agent/Provider (V‑A6..V‑A8)
- Acceptance: Unknown id/import path error; invalid coercions warning; structured/non‑JSON mismatch warning.
- Unit: Import path resolver isolated from side effects; type coercion checks.
- Integration: Pipelines referencing missing skills and malformed configs.

Source Mapping (ruamel.yaml optional)
- Acceptance: When ruamel available, findings include line/column; otherwise path only.
- Unit: Mapper extracts correct positions for steps; fallback logic.

Telemetry
- Acceptance: When enabled, emits counts per rule/severity; never includes code content.
- Unit: Counter aggregation.

All tests must respect Team Guide test markers and avoid network or agent execution during validation.

---

## 15. Documentation Update Requirements

For every implemented feature/rule, update or add documentation as follows. PRs must include these docs to be accepted.

- CLI Reference
  - File: `docs/cli/validate.md` (new) and `docs/cli/main.md` (if exists)
  - Content: All flags (`--fix`, `--yes`, `--rules`, `--imports`, `--format`, `--explain`, baseline options), examples, exit codes, sample outputs for text/json/sarif.

- Rule Catalog
  - File: `docs/validation_rules.md` (new)
  - Content: One section per rule (ID, severity, rationale, trigger patterns, examples with before/after, fixability, related config keys, suppression examples).

- Templating Guidance
  - Files: `docs/creating_yaml.md`, `docs/creating_yaml_best_practices.md`
  - Content: Explicit examples for `previous_step` vs `steps.<name>`, filter allow‑list, tojson usage, common mistakes (V‑T1..T4).

- Imports & Composition
  - File: `docs/blueprints.md`
  - Content: `imports:` resolution, base_dir semantics, recursive validation behavior, output mappings between child/parent (V‑I2, V‑I5, V‑I6) with examples.

- Examples
  - Directory: `examples/validation/`
  - Content: Minimal YAMLs per rule group demonstrating triggers and correct fixes.

- Release Notes
  - File: `CHANGELOG.md` or `docs/release_notes.md`
  - Content: New flags, rules, and migration notes; links to Rule Catalog.

- SARIF Integration
  - File: `docs/ci/sarif.md` (new)
  - Content: Using `--format sarif` in CI, uploading to GitHub code scanning, sample JSON.

All documentation changes must follow style/structure conventions and include runnable snippets where applicable. Cross‑reference Team Guide principles (centralized configuration, policy separation) where relevant.

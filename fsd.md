## FSD-020: Flujo Pipeline Architect (Self-Hosting Blueprint Generator)

### 1. Goal and Scope

Build a conversational, agent-driven “Architect” that generates runnable Flujo YAML blueprints from a natural-language goal, with guardrails and self-correction. The deliverable is a production-grade feature aligned with the Flujo Team Guide (policy-driven architecture, strict type safety, robust exception handling, and comprehensive testing).

- **In-scope**:
  - CLI: `flujo create` interactive command.
  - Architect pipeline (YAML) that orchestrates decomposition → tool matching → YAML generation → validation → self-repair.
  - Skill/Tool discovery via a formalized `SkillRegistry` contract and catalog loading.
  - Validation and self-repair loop without subprocesses, using internal validation APIs.
  - Security guardrails (safe imports allowlist, side-effect skill confirmations).
  - Documentation and tests (unit, integration, regression, security).
- **Out-of-scope**:
  - Creating new step types or modifying `ExecutorCore` behavior (respect policy-driven architecture).
  - Building a full web UI (CLI-first; web can follow later).

### 2. User Stories

- As a business analyst, I describe a goal (e.g., “Summarize a URL, translate to Spanish, post to Slack”) and receive a ready-to-run `pipeline.yaml` and optional `flujo.toml`.
- As a developer, I want generated pipelines to pass `flujo validate` (strict) and follow Flujo YAML schema so I can trust and run them safely.
- As an operator, I want guardrails so self-generated pipelines cannot import unsafe modules or run unsafe skills without confirmation.

### 3. Architecture Overview

- The Architect itself is a Flujo YAML pipeline composed of:
  - `decomposer` agent: turns a natural-language goal into structured step intents.
  - `tool_matcher` agent: maps intents to registered skills or marks as `llm_agent`.
  - `yaml_writer` agent: emits a valid Flujo YAML blueprint based on decisions.
  - `validator` callable: validates the generated YAML and returns a `ValidationReport`.
  - `conditional` + `loop`: self-repair cycle until valid or reaching `max_loops`.

- The CLI `flujo create` launches this pipeline, mediates clarifications, and writes outputs to disk.

- All execution flows through existing policies. No special logic in `ExecutorCore`.

### 4. Detailed Design

#### 4.1 CLI: `flujo create`

- Command shape:
```
flujo create --goal "<natural language goal>" \
             [--context-file <json|yaml>] \
             [--output-dir <path>] \
             [--strict] \
             [--non-interactive]
```

- Responsibilities:
  - Load and run a bundled Architect pipeline YAML (or an override via `--pipeline` in future).
  - Provide goal and optional context to initial step input/context.
  - Interactively ask for required parameters (e.g., Slack channel) unless `--non-interactive`.
  - Persist outputs to `<output-dir>/pipeline.yaml` (and optional `flujo.toml`).
  - Run internal validation; if invalid, the architect performs repair loop. Exit non-zero with `--strict` if still invalid after max loops.

- Implementation notes:
  - Use existing helpers for loading/running pipelines.
  - Do not shell out for validation; use internal API (see 4.4).
  - Respect `FLUJO_TEAM_GUIDE.md`: keep business logic in the pipeline; CLI orchestrates I/O only.

#### 4.2 Architect Pipeline (YAML)

- Conforms to current blueprint schema (top-level `version`, optional `agents`, and `steps` where each step has `kind`).
- Sketch (compilable shape):
```yaml
version: "0.1"

agents:
  decomposer:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      Decompose a natural-language goal into ordered steps {step_name, purpose, input_type, output_type}.
    output_schema:
      type: object
      properties:
        steps:
          type: array
          items:
            type: object
            properties:
              step_name: { type: string }
              purpose: { type: string }
              input_type: { type: string }
              output_type: { type: string }
            required: [step_name, purpose, input_type, output_type]
      required: [steps]

  tool_matcher:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      Given a step and available skills (id, description, input_schema), choose a tool or 'llm_agent'.
    output_schema:
      type: object
      properties:
        step_name: { type: string }
        implementation: { type: string, enum: ["tool", "llm_agent"] }
        tool_name: { type: string }
      required: [step_name, implementation]

  yaml_writer:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      Emit a valid Flujo YAML blueprint (v0). Use top-level 'version', optional 'agents', and 'steps' with 'kind'.
      Steps must use 'uses: agents.<name>' or 'agent: <import path>'. No unsupported keys.
    output_schema:
      type: object
      properties:
        yaml_text: { type: string }
      required: [yaml_text]

steps:
  - kind: step
    name: DecomposeGoal
    uses: agents.decomposer
    updates_context: true

  - kind: step
    name: DiscoverSkills
    agent:
      id: "flujo.infra.skills_catalog:load_skills_catalog"
      params: { directory: "." }
    updates_context: true

  - kind: map
    name: MapStepsToTools
    map:
      iterable_input: "decomposition.steps"
      body:
        - kind: step
          name: MatchTool
          uses: agents.tool_matcher
    updates_context: true

  - kind: step
    name: GenerateBlueprint
    uses: agents.yaml_writer
    updates_context: true

  - kind: loop
    name: ValidateAndRepair
    loop:
      max_loops: 5
      body:
        - kind: step
          name: ValidateGeneratedYAML
          agent: "flujo.domain.blueprint.loader:load_pipeline_blueprint_from_yaml"
          # Input must be 'yaml_text' from context; adapter step may be inserted if needed
        - kind: step
          name: ComputeValidationReport
          agent: "flujo.cli.helpers:validate_pipeline_file" # optional if we persisted yaml
        - kind: conditional
          name: ValidityBranch
          condition: "flujo.utils.context:predicate_is_valid_report"
          branches:
            valid:
              - kind: step
                name: Done
            invalid:
              - kind: step
                name: RepairBlueprint
                uses: agents.yaml_writer
```

Notes:
- The actual validation loop will use the internal API in 4.4 to avoid filesystem coupling. We will introduce an adapter callable that accepts in-memory YAML and returns a `ValidationReport` directly.

#### 4.3 SkillRegistry Formalization

- Extend `SkillRegistry.register` metadata to include:
  - `capabilities: list[str]` (tags for matching, e.g., ["summarize", "translate", "slack.post"]).
  - `safety_level: Literal["none", "low", "medium", "high"]`.
  - `auth_required: bool` and `auth_scope: Optional[str]`.
  - `arg_schema: dict[str, Any]` (JSON Schema for inputs), `output_schema: dict[str, Any]` (already supported).
  - `side_effects: bool` (e.g., posting to Slack).

- The `skills_catalog` loader will accept these optional fields and pass through to the registry.

- Rationale: Improves deterministic tool matching and enables safety gating in the CLI.

#### 4.4 Internal Validation API (No Subprocess)

- Add a public callable to validate YAML text in-memory:
```python
def validate_yaml_text(yaml_text: str) -> "ValidationReport":
    """Load a Pipeline from YAML text and return its ValidationReport.

    - Uses loader: load_pipeline_blueprint_from_yaml(yaml_text)
    - Returns: pipeline.validate_graph()
    - Raises: BlueprintError on YAML/schema errors (provide readable message)
    """
```

- Add a small predicate for branching in the Architect pipeline:
```python
def predicate_is_valid_report(report: "ValidationReport") -> str:
    return "valid" if report.is_valid else "invalid"
```

- Location: keep under existing modules to avoid new dependencies (e.g., `flujo.cli.helpers` or `flujo.domain.blueprint.loader` companion).

#### 4.5 Security & Governance

- Enforce `FlujoConfig.blueprint_allowed_imports` as an allowlist for imported modules/paths in generated YAML.
- CLI `flujo create` must:
  - Detect `side_effects=True` skills and require explicit user confirmation (or `--allow-side-effects` in non-interactive runs).
  - Mask secrets in logs.
  - Write outputs to a safe directory; avoid overwriting existing files unless `--force`.
- Validation loop must not execute agents; it only compiles and validates blueprints.

#### 4.6 Telemetry & Tracing

- Emit metrics for architect runs: iterations, validation error counts, convergence rate, and time-to-valid.
- Add trace spans for each architect step to aid debugging.

#### 4.7 Error Handling (Team Guide Compliance)

- No step/policy logic in `ExecutorCore`. Use existing step policies.
- Re-raise control-flow exceptions; do not convert to data failures.
- Architect loop and map steps rely on existing policies for isolation/idempotency.

### 5. Public APIs and Types (Strict Typing)

- `validate_yaml_text(yaml_text: str) -> ValidationReport` (annotated return type).
- `predicate_is_valid_report(report: ValidationReport) -> str`.
- `SkillRegistry.register(…, capabilities: list[str] | None = None, safety_level: Literal[... ] | None = None, auth_required: bool | None = None, auth_scope: str | None = None, side_effects: bool | None = None, arg_schema: dict[str, Any] | None = None, output_schema: dict[str, Any] | None = None)`.

All code must use explicit type annotations, avoid `Any` unless unavoidable, and pass mypy.

### 6. Testing Strategy (TDD-first)

Design tests to improve the system, not just make it pass. Follow the guide’s principles and keep thresholds strict.

- Unit tests:
  - YAML loader/compiler accepts `agents` and `uses: agents.<name>` and rejects unknown references.
  - Map/Loop/Conditional blueprint round-trip: YAML → Pipeline → dump → YAML → Pipeline.
  - `validate_yaml_text` returns `ValidationReport` and surfaces `BlueprintError` with line/column.
  - `SkillRegistry` extended fields registration and retrieval.
  - Tool matcher agent output schema parsing (Pydantic model via `output_schema`).

- Integration tests:
  - End-to-end `flujo create --goal "…" --non-interactive` with a golden prompt leading to a deterministic blueprint.
  - Architect repair loop converges within N iterations for a known-invalid initial YAML (inject a controlled defect via prompt fixture).
  - Generated blueprint runs through `flujo run` up to first step (no side effects) or a mocked side-effect skill.

- Security tests:
  - Allowlist prevents imports outside `blueprint_allowed_imports`.
  - CLI requires confirmation for `side_effects=True` skills; non-interactive mode fails without `--allow-side-effects`.

- Regression tests:
  - Frozen examples under `tests/golden_traces` for the architect scenario.
  - Snapshot of validation errors categorized by rule ID to catch drift.

- Performance tests (lightweight):
  - Architect pipeline completes under a reasonable bound (e.g., < 2s with mocked LLMs) using stubs.

### 7. Documentation

- New doc page: “Architect: Generate Pipelines from Natural Language”.
  - Explain YAML schema compliance, safety guardrails, and the repair loop.
  - Provide copy-paste examples and `flujo create` usage.

- Update existing docs:
  - `docs/blueprints_yaml.md`: add a subsection on generated pipelines and constraints the generator adheres to.
  - Reference validation rules page for typical failures.

### 8. Backward Compatibility & Migration

- No breaking changes to existing DSL or executor. Additive APIs only.
- The Architect pipeline ships as an internal YAML file; users can override later.
- If `skills.yaml` absent, tool matching gracefully defaults to `llm_agent` for all steps.

### 9. Risks and Mitigations

- Hallucinated YAML fields → Provide exact schema snippets and few-shot examples to `yaml_writer`. Validate every iteration.
- Unsafe tools usage → Capability/safety metadata in registry; CLI gating for side effects; import allowlist enforced.
- Non-determinism → Use small, reliable models with strict `output_schema`; record golden outputs in tests; allow `--model` override for experimentation.
- Performance → Mock LLMs in tests; keep hot-path simple; profile only if regressions appear.

### 10. Acceptance Criteria (Definition of Done)

- `flujo create --goal "…"` produces `pipeline.yaml` that passes `flujo validate --strict` for common goals (with stubs for side-effect tools).
- Self-repair loop converges within `max_loops`=5 for injected errors in YAML.
- New APIs are fully typed and covered by unit tests; CI `make all` passes.
- Security guardrails active: import allowlist enforced; side-effect tools gated.
- Documentation published; examples reproducible.

### 11. Work Breakdown & Timeline

1) Validation utilities (4.4) and tests — 0.5d

2) SkillRegistry extensions + catalog loader updates — 1d

3) Architect pipeline YAML (first draft) + stubs — 1d

4) CLI `flujo create` with interactive and non-interactive modes — 1d

5) Self-repair loop wiring + integration tests — 1d

6) Security guardrails + tests — 0.5d

7) Docs + examples — 0.5d

Buffer for polish and reviews — 1d

### 12. Appendix

#### 12.1 Example Registry Entry (skills.yaml)
```yaml
slack.post_message:
  path: "my_pkg.slack:SlackPoster"
  description: "Post a message to Slack"
  capabilities: ["slack.post", "notify"]
  side_effects: true
  auth_required: true
  arg_schema:
    type: object
    properties:
      channel: { type: string }
      message: { type: string }
    required: [channel, message]
  output_schema:
    type: object
    properties:
      ok: { type: boolean }
      ts: { type: string }
    required: [ok]
```

#### 12.2 Minimal Validation Utility (signature)
```python
from typing import Any
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml

def validate_yaml_text(yaml_text: str) -> "ValidationReport":
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    return pipeline.validate_graph()
```


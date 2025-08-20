# Architect: Generate Pipelines from Natural Language

The Architect helps you create runnable Flujo YAML blueprints from a goal. As of the latest version,
the Architect is implemented programmatically in `flujo/architect/builder.py` using a
`StateMachineStep`. A minimal one‑shot builder remains the default for compatibility with
existing tests and CI; the full conversational state machine can be enabled via an environment flag.

## Usage

```bash
# Recommended: inside a project initialized with `flujo init`
flujo create --goal "Summarize a URL and post to Slack"

# Optional flags
flujo create [--output-dir ./out] [--context-file context.yaml] \
             [--non-interactive] [--allow-side-effects] [--force] [--strict]
```

- `--context-file`: JSON/YAML map with extra context injected into the Architect.
- `--allow-side-effects`: required to proceed when the generated blueprint references skills marked with `side_effects: true`.
- `--force`: overwrite `pipeline.yaml` if it already exists (not needed for project default).
- When run inside a project, `create` prompts for a pipeline name (added to `pipeline.yaml`) and a budget per run (added to `flujo.toml`).
- `--strict`: exit non-zero if the generated blueprint is invalid.

### Conversational Architect (State Machine)

Set this flag to enable the full multi‑state Architect (gathering context, planning, approval,
parameter collection, generation, validation, optional dry run):

```bash
export FLUJO_ARCHITECT_STATE_MACHINE=1
flujo create --goal "Fetch a web page and summarize it" --output-dir ./out
```

## Safety and Governance

- The blueprint loader enforces `blueprint_allowed_imports` from `flujo.toml`.
- Side-effecting skills require confirmation or `--allow-side-effects` in non-interactive mode.
- Secrets are masked in logs by default.

## Validation and Repair Loop

The Architect validates the generated YAML and can iteratively repair it. In the programmatic
implementation, validation and repair are implemented via built-ins (`validate_yaml`,
`repair_yaml_ruamel`) and controlled by the state machine.

## Tuning Timeouts and Retries in YAML

GPT‑5 agents often need more time to reason. You can increase timeouts and adjust retries in your blueprint:

- Agent-level (affects the LLM call, enforced by the Agent wrapper):

```yaml
agents:
  architect_agent:
    model: "openai:gpt-5"
    timeout: 180        # seconds
    max_retries: 1
```

- Step-level (affects plugin/validator phases; normalized to `timeout_s`):

```yaml
steps:
  - name: DesignAndBuildBlueprint
    uses: agents.architect_agent
    config:
      timeout: 180      # alias to timeout_s for step-level operations
      max_retries: 1
```

Tip: A reference YAML pipeline remains available at `examples/architect_pipeline.yaml`. The
programmatic architect supersedes it for the CLI, but the YAML example is useful to study prompt and
agent settings (e.g., higher timeouts for GPT‑5).

## Skills Catalog

Place a `skills.yaml` next to your blueprint to register custom tools. Example entry:

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
    required: [ok]
```

## Notes

- In interactive runs, missing required `params` for registered skills are prompted.
- In non-interactive runs, provide all required parameters up front or use `--context-file`.
- When HITL is enabled (via context flag `hitl_enabled: true`), plan approval can be interactive.

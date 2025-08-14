# Architect: Generate Pipelines from Natural Language

The Architect helps you create runnable Flujo YAML blueprints from a goal.

## Usage

```bash
flujo create --goal "Summarize a URL and post to Slack" \
             --output-dir ./out \
             [--context-file context.yaml] \
             [--non-interactive] \
             [--allow-side-effects] \
             [--force] \
             [--strict]
```

- `--context-file`: JSON/YAML map with extra context injected into the Architect.
- `--allow-side-effects`: required to proceed when the generated blueprint references skills marked with `side_effects: true`.
- `--force`: overwrite `pipeline.yaml` if it already exists.
- `--strict`: exit non-zero if the generated blueprint is invalid.

## Safety and Governance

- The blueprint loader enforces `blueprint_allowed_imports` from `flujo.toml`.
- Side-effecting skills require confirmation or `--allow-side-effects` in non-interactive mode.
- Secrets are masked in logs by default.

## Validation and Repair Loop

The bundled Architect pipeline validates the generated YAML and can iteratively repair it up to a maximum number of loops.

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



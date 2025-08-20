Dynamic Prompt Template Example

Files
- `pipeline.yaml`: Defines an agent with a templated `system_prompt` loaded from a file with variables.
- `prompts/summarize_with_context.md`: Prompt template using variables `max_sentences`, `topic`, `author`, plus runtime `context` and `previous_step`.

Usage
- Validate the pipeline (no API calls):
  - `python -m flujo.cli.main validate pipeline.yaml`

- Run the pipeline (requires API keys configured via `make install` and `flujo.toml`):
  - `python -m flujo.cli.main run pipeline.yaml` (provide appropriate input/context via CLI options or your app integration)

Notes
- `topic` is resolved from `context.current_topic` at runtime.
- `author` is resolved from the previous step output (here just a placeholder; in a multi-step pipeline, pass through a dict with `author_name`).
- The agent’s `system_prompt` is rendered just-in-time per run and restored after execution.

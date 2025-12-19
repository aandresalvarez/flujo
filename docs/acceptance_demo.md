# Acceptance Demo: Conversational Loops (FSD-033)

This walkthrough demonstrates the new conversational loop features end‑to‑end.

Prerequisites
- Installed Flujo and configured provider API keys
- Project initialized (`flujo init`) with a local SQLite state (default)

## 1) Generate a conversational loop via the wizard

```bash
uv run flujo create \
  --wizard \
  --wizard-pattern loop \
  --wizard-conversation \
  --wizard-ai-turn-source last \
  --wizard-user-turn-sources hitl \
  --wizard-history-strategy truncate_tokens \
  --wizard-history-max-tokens 4096 \
  --name demo_conv \
  --output-dir .
```

This writes `pipeline.yaml` with:
- `conversation: true`
- `history_management` block
- loop `body` and conversational presets

## 2) Run the pipeline and pause at HITL

```bash
uv run flujo run --input "Initial Goal"
```

On the first run, expect a HITL pause. The state persists to the configured SQLite DB.

## 3) Resume with a response

There is no direct CLI `resume` yet; use a short Python snippet to resume with the same `run_id`:

```python
# resume_demo.py
import asyncio
from flujo.application.runner import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext

async def main():
    p = Pipeline.from_yaml_file('pipeline.yaml')
    f = Flujo(pipeline=p)
    # You can look up your run_id via `flujo lens list` or store it after the first run
    run_id = input('Enter run_id to resume: ').strip()
    # Load paused state and resume with human input
    from flujo.application.core.state.state_manager import StateManager
    from flujo.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend()
    sm = StateManager[PipelineContext](backend)
    ctx, last_output, idx, created_at, pname, pver, step_history = await sm.load_workflow_state(run_id, PipelineContext)
    from flujo.domain.models import PipelineResult
    paused = PipelineResult(step_history=step_history, final_pipeline_context=ctx)
    final = await f.resume_async(paused, human_input='Tomorrow')
    print('Final success:', final.success)

asyncio.run(main())
```

Run the resume:
```bash
uv run python resume_demo.py
```

## 4) Inspect the run with `lens`

```bash
# Show tree with durations and attributes
uv run flujo lens trace <run_id>
```

Look for these events under spans:
- `agent.prompt` for a redacted preview of the injected history
- `agent.system` / `agent.input` / `agent.response` / `agent.usage` for full agent-call lifecycle
- `loop.iteration` to correlate messages with iterations

Alternatively, you can use the run command with debug flags to see a rich Debug Trace and export a full log:

```bash
# Safe previews only
uv run flujo run --debug --trace-preview-len 1500 --input "Initial Goal"

# Full unredacted prompts/responses (unsafe; local only)
uv run flujo run --debug-prompts --trace-preview-len 4000 --input "Initial Goal"

# Export everything to a JSON file
uv run flujo run --debug --debug-export output/last_run_debug.json --input "Initial Goal"
```

## 5) Notes on cost guards
- Use `history_management` to bound cost (token/turn limits).
- For long conversations, pick `summarize` and configure a ratio; default template will remain compact.

## 6) Optional: Strict performance check
Set `FLUJO_STRICT_PERF=1` to enable < 5 ms per iteration overhead check in the benchmark:
```bash
FLUJO_STRICT_PERF=1 uv run pytest tests/benchmarks/test_conversational_overhead.py -q
```

This focuses on HistoryManager binding cost (excluding model IO).

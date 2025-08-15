## Debugging with Trace Replay

This guide shows how to deterministically replay a prior pipeline run using recorded traces and step results.

### What Replay Does

- Restores the initial input and context from the original run
- Serves recorded raw agent responses instead of calling external APIs
- Automatically injects recorded human input for HITL resumes
- Produces the same step outputs and final context unless the code changed (which is often desired to verify a fix)

### Prerequisites

- Your operations store is configured via `FLUJO_STATE_URI` (e.g., `sqlite:///./ops.db`) or default `flujo_ops.db` exists
- The prior run you want to replay is present in the store (trace + steps)
- You have access to the target pipeline definition file and variable name

### CLI Usage

Use the `lens replay` command to trigger a local replay.

- Inside a project (created via `flujo init`), it automatically uses `pipeline.yaml`:

```bash
flujo lens replay <run_id>
```

- Or point to a specific pipeline file:

```bash
flujo lens replay <run_id> --file path/to/pipeline.yaml
# or a Python file and object name
flujo lens replay <run_id> --file path/to/pipeline.py --object pipeline
```

Options:

- `--file, -f`: Path to the pipeline file (`.yaml` or Python)
- `--object, -o`: The variable name of your pipeline in a Python file (default: `pipeline`)
- `--json`: Print JSON-serialized results instead of a formatted summary

Examples:

```bash
export FLUJO_STATE_URI="sqlite:////absolute/path/to/ops.db"
flujo lens replay run_2025_08_12_15_00 -f examples/robust_flujo_pipeline.py -o pipeline
```

### Programmatic Usage

If you already have a `Flujo` runner in code:

```python
from flujo.application.runner import Flujo

# runner: Flujo = ... your existing runner with a configured pipeline
result = await runner.replay_from_trace("<run_id>")
```

### How It Works (Under the Hood)

1. Captures the raw agent responses during the original execution and persists them in the `steps` table (`raw_response` column).
2. Emits a `flujo.resumed` trace event with human input during HITL resumes.
3. During replay, a `ReplayAgent` serves responses from a map keyed by step name and attempt.
4. The runner temporarily patches `resume_async` to feed recorded human inputs.
5. The replay executes `run_async` and automatically continues resuming until completion.

### Notes

- If your code or policies changed, replay may diverge by design (this is useful to validate fixes). The `ReplayAgent` will raise if a recorded response is missing for a requested step.
- For large traces, consider archiving and retention strategies; raw responses can be big.



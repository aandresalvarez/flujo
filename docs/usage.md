# Usage

## CLI

```bash
orch solve "Write a summary of this document."
orch show-config
orch bench --prompt "hi" --rounds 3
orch --profile
```

## API

```python
from pydantic_ai_orchestrator import Orchestrator
orch = Orchestrator()
result = orch.run_sync("Write a poem.")
print(result)
```

## Environment Variables

- `ORCH_OPENAI_API_KEY` (required)
- `ORCH_LOGFIRE_API_KEY` (optional)
- `ORCH_REFLECTION_ENABLED` (default: true)
- `ORCH_MAX_ITERS`, `ORCH_K_VARIANTS`

## Scoring
- Ratio and weighted scoring supported
- Reward model stub included (extendable)

## Reflection
- Reflection agent can be toggled via `ORCH_REFLECTION_ENABLED` 
# pydantic-ai-orchestrator

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-orchestrator.svg)](https://pypi.org/project/pydantic-ai-orchestrator/)
[![codecov](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator)

Production-ready orchestration for Pydantic-based AI agents.

## Quick Start

```bash
pip install pydantic-ai-orchestrator
export ORCH_OPENAI_API_KEY=sk-...
orch solve "Write a haiku about AI."
```

## Architecture

```
[CLI] --+--> [Orchestrator] --+--> [Agent(s)]
        |                    |
        +--> [Scoring]       +--> [Reflection Agent]
        +--> [Telemetry]
```

- **Settings**: via env vars (see below)
- **Telemetry**: Logfire, OTLP
- **Extensible**: Add your own agents and scorers

## CLI Usage

- `orch solve "prompt"` – Solve a task
- `orch show-config` – Show current config (secrets masked)
- `orch bench --prompt "hi" --rounds 5` – Benchmark
- `orch --profile` – Enable Logfire span viewer

## Environment Variables

- `ORCH_OPENAI_API_KEY` (required)
- `ORCH_LOGFIRE_API_KEY` (optional)
- `ORCH_REFLECTION_ENABLED` (default: true)
- `ORCH_MAX_ITERS`, `ORCH_K_VARIANTS`

## API Usage

```python
from pydantic_ai_orchestrator import Orchestrator
orch = Orchestrator()
result = orch.run_sync("Write a poem.")
print(result)
```

## License
MIT 
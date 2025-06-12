# pydantic-ai-orchestrator

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-orchestrator.svg)](https://pypi.org/project/pydantic-ai-orchestrator/)
[![codecov](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator)

Production-ready orchestration for Pydantic-based AI agents.

## Features

* Typed settings and secrets
* Telemetry and observability
* Pluggable scoring (ratio, weighted, reward-model)
* CLI and API
* Extensible agent and reflection system

## Installation

```bash
pip install pydantic-ai-orchestrator
```

## Quick Start

Install and configure:

```bash
pip install pydantic-ai-orchestrator
cp .env.example .env  # then edit the file
export OPENAI_API_KEY=sk-...
orch solve "Write a haiku about AI."
```

Example usage in Python:

```python
from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.agents import (
    review_agent, solution_agent, validator_agent, get_reflection_agent
)

# Create orchestrator
orch = Orchestrator(
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent()
)

# Create task
task = Task(prompt="Write a short motivational haiku about debugging.")

# Execute task
best = orch.run_sync(task)

# Display result
print("Solution:\n", best.solution)
```

For more examples, see the `examples` folder.

## Architecture

```
[CLI] --+--> [Orchestrator] --+--> [Agent(s)]
        |                    |
        +--> [Scoring]       +--> [Reflection Agent]
        +--> [Telemetry]
```

* **Settings**: Environment variables
* **Telemetry**: Logfire, OTLP
* **Extensible**: Custom agents and scorers

## CLI Usage

* `orch solve "prompt"` – Solve a task
* `orch show-config` – Display config (secrets masked)
* `orch bench --prompt "hi" --rounds 5` – Benchmark (requires `numpy`; install with `pip install pydantic-ai-orchestrator[bench]`)
* `orch --profile` – Enable Logfire span viewer

## Environment Variables

* `OPENAI_API_KEY` (optional, for OpenAI models)
* `GOOGLE_API_KEY` (optional, for Gemini models)
* `ANTHROPIC_API_KEY` (optional, for Claude models)
* `LOGFIRE_API_KEY` (optional)
* `REFLECTION_ENABLED` (default: `true`)
* `MAX_ITERS`, `K_VARIANTS`

## API Usage

```python
from pydantic_ai_orchestrator import Orchestrator, Task, init_telemetry
from pydantic_ai_orchestrator.infra.agents import (
    OpenAIChat, DefaultReviewAgent, DefaultValidatorAgent, DefaultReflectionAgent
)
from pydantic_ai_orchestrator.llm.openai.chat_completion import OpenAIChatCompletion

# Initialize telemetry
init_telemetry()

# Configure agents
solution_agent = OpenAIChat(llm_provider=OpenAIChatCompletion(model="gpt-3.5-turbo"))
review_agent = DefaultReviewAgent()
validator_agent = DefaultValidatorAgent()
reflection_agent = DefaultReflectionAgent()

# Create orchestrator
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent
)

# Define task
task = Task(prompt="Write a short story about a robot who learns to paint.")

# Run task
try:
    best_candidate = orch.run_sync(task, k_variants=3)
    if best_candidate:
        print("Best Solution:\n", best_candidate.solution)
        print("\nScore:", best_candidate.score)
        print("\nFeedback:\n", best_candidate.feedback)
    else:
        print("No solution found.")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Exploring Examples

The `examples` directory includes practical demonstrations:

| File                         | Description                          |
| ---------------------------- | ------------------------------------ |
| **00\_quickstart.py**        | Minimal run with ratio scorer        |
| **01\_weighted\_scoring.py** | Custom checklist scoring             |
| **02\_custom\_agents.py**    | Custom LLM or agent logic            |
| **03\_reward\_scorer.py**    | Reward model scoring                 |
| **04\_batch\_processing.py** | Batch processing of multiple prompts |

### Running Examples

1. Install dependencies and activate your environment.
2. Set OpenAI API key:

```bash
export OPENAI_API_KEY='sk-...'
```

3. Run examples:

```bash
python examples/00_quickstart.py
```

## License

MIT

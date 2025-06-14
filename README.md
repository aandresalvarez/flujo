# pydantic-ai-orchestrator

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-orchestrator.svg)](https://pypi.org/project/pydantic-ai-orchestrator/)
[![codecov](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator)

Production-ready orchestration for Pydantic-based AI agents.

## Features

* Typed settings and secrets
* Telemetry and observability
* Pluggable scoring utilities
* CLI and API
* Extensible agent system
* **New:** Flexible pipeline DSL for custom workflows
* **Simplified orchestrator** built on the PipelineRunner

## Installation

```bash
pip install pydantic-ai-orchestrator
```

## Quick Start

Install and configure:

```bash
pip install pydantic-ai-orchestrator
cp .env.example .env  # then edit the file
# Environment variables are loaded automatically
orch solve "Write a haiku about AI."
```

Example usage in Python:

```python
from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.agents import (
    review_agent, solution_agent, validator_agent
)

# Create orchestrator
# Orchestrator runs a simple review → solve → validate pipeline
orch = Orchestrator(
    review_agent,
    solution_agent,
    validator_agent,
)

# Create task
task = Task(prompt="Write a short motivational haiku about debugging.")

# Execute task
best = orch.run_sync(task)

# Display result
print("Solution:\n", best.solution)
```

### Pipeline DSL

Use the `Step` and `PipelineRunner` classes to build custom workflows:

```python
from pydantic_ai_orchestrator import Step, PipelineRunner
from pydantic_ai_orchestrator.plugins.sql_validator import SQLSyntaxValidator
from pydantic_ai_orchestrator.testing.utils import StubAgent

solution_agent = StubAgent(["SELECT FROM"])  # invalid SQL
validator_agent = StubAgent([None])

pipeline = Step.solution(solution_agent) >> Step.validate(
    validator_agent, plugins=[SQLSyntaxValidator()]
)
runner = PipelineRunner(pipeline)
result = runner.run("SELECT FROM")
```

For more examples, see the `examples` folder.

## Architecture

```
[CLI] --+--> [Orchestrator] --+--> [PipelineRunner] --+--> [Agents]
        |
        +--> [Telemetry]
```

* **Settings**: Environment variables
* **Telemetry**: Logfire, OTLP
* **Extensible**: Custom agents and scorers

## CLI Usage

* `orch solve "prompt"` – Solve a task
* `orch show-config` – Display config (secrets masked)
* `orch bench --prompt "hi" --rounds 5` – Benchmark (requires `numpy`; install with `pip install pydantic-ai-orchestrator[bench]`)
* `orch explain pipeline.py` – Print steps defined in a pipeline file
* `orch --profile` – Enable Logfire span viewer

## Environment Variables

* `OPENAI_API_KEY` (optional, for OpenAI models)
* `GOOGLE_API_KEY` (optional, for Gemini models)
* `ANTHROPIC_API_KEY` (optional, for Claude models)
* `LOGFIRE_API_KEY` (optional)
* `REFLECTION_ENABLED` (default: `true`)
* `MAX_ITERS`, `K_VARIANTS`

Environment variables can be stored in a `.env` file. The CLI loads this file
automatically using `python-dotenv`.

## API Usage

```python
from pydantic_ai_orchestrator import Orchestrator, Task, init_telemetry
from pydantic_ai_orchestrator.infra.agents import (
    OpenAIChat, DefaultReviewAgent, DefaultValidatorAgent
)
from pydantic_ai_orchestrator.llm.openai.chat_completion import OpenAIChatCompletion

# Initialize telemetry
init_telemetry()

# Configure agents
solution_agent = OpenAIChat(llm_provider=OpenAIChatCompletion(model="gpt-3.5-turbo"))
review_agent = DefaultReviewAgent()
validator_agent = DefaultValidatorAgent()

# Create orchestrator
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent
)

# Define task
task = Task(prompt="Write a short story about a robot who learns to paint.")

# Run task
try:
    best_candidate = orch.run_sync(task)
    if best_candidate:
        print("Best Solution:\n", best_candidate.solution)
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
| **05\_pipeline_sql.py**      | Pipeline DSL with SQL validation     |

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

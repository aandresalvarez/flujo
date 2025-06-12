# pydantic-ai-orchestrator

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-orchestrator.svg)](https://pypi.org/project/pydantic-ai-orchestrator/)
[![codecov](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator/branch/main/graph/badge.svg)](https://codecov.io/gh/yourorg/pydantic-ai-orchestrator)

Production-ready orchestration for Pydantic-based AI agents.

## Features
- Typed settings and secrets
- Telemetry and observability
- Pluggable scoring (ratio, weighted, reward-model)
- CLI and API
- Extensible agent and reflection system

## Installation

```bash
pip install pydantic-ai-orchestrator
```

## Quick Start

```bash
pip install pydantic-ai-orchestrator
cp .env.example .env  # then edit the file
export ORCH_OPENAI_API_KEY=sk-...
orch solve "Write a haiku about AI."

Here's a quick example of how to use `pydantic-ai-orchestrator` in your Python code:

```python
from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.agents import review_agent, solution_agent, validator_agent, get_reflection_agent

# 1️⃣ Create a default orchestrator
orch = Orchestrator(
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent()
)

# 2️⃣ Wrap your prompt in a Task
task = Task(prompt="Write a short motivational haiku about debugging.")

# 3️⃣ Synchronous, blocking call – returns a Candidate object
best = orch.run_sync(task)

# 4️⃣ Inspect the result
print("Solution:\n", best.solution)
```

This example demonstrates creating an orchestrator, defining a task to write a haiku, running the task, and printing the generated solution. For more detailed examples, including custom agents, weighted scoring, and batch processing, please check out the `examples` folder.
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
- `orch bench --prompt "hi" --rounds 5` – Benchmark (requires `numpy`; install with `pip install pydantic-ai-orchestrator[bench]`)
- `orch --profile` – Enable Logfire span viewer

## Environment Variables

- `ORCH_OPENAI_API_KEY` (required)
- `ORCH_LOGFIRE_API_KEY` (optional)
- `ORCH_REFLECTION_ENABLED` (default: true)
- `ORCH_MAX_ITERS`, `ORCH_K_VARIANTS`

## API Usage

The `Orchestrator` class is the main entry point for using `pydantic-ai-orchestrator` programmatically.

```python
from pydantic_ai_orchestrator import Orchestrator, Task, init_telemetry
from pydantic_ai_orchestrator.infra.agents import OpenAIChat, DefaultReviewAgent, DefaultValidatorAgent, DefaultReflectionAgent
from pydantic_ai_orchestrator.llm.openai.chat_completion import OpenAIChatCompletion

# Initialize telemetry (optional, but recommended)
# Call this once at the startup of your application.
init_telemetry()

# 1. Configure your agents
# You can use the default agents or provide your own custom implementations.
# Here, we specify a particular OpenAI model for the solution agent.
solution_agent = OpenAIChat(llm_provider=OpenAIChatCompletion(model="gpt-3.5-turbo"))
review_agent = DefaultReviewAgent()
validator_agent = DefaultValidatorAgent()
reflection_agent = DefaultReflectionAgent() # Uses settings.ORCH_DEFAULT_REFLECTION_MODEL by default

# 2. Create an Orchestrator instance with your chosen agents
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
)

# 3. Define your task
task = Task(prompt="Write a short story about a robot who learns to paint.")

# 4. Run the orchestrator
# You can specify the number of variants to generate using k_variants.
# The orchestrator will run the solution agent k_variants times,
# then review, validate, and score each variant to select the best one.
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

# You can also run asynchronously using orch.run_async(task, k_variants=3)
```

Call `init_telemetry()` once at startup to configure logging and tracing for your application. For more advanced usage, such as custom scorers or different agent configurations, refer to the scripts in the `examples` folder.

## Exploring the Examples

While the "Quick Start" and "API Usage" sections provide a good starting point, the `examples` folder in this repository contains a variety of scripts that demonstrate more specific and advanced use cases of `pydantic-ai-orchestrator`. These examples are designed to be standalone and easy to run.

Here's an overview of what you can find:

| File | What it shows |
|------|---------------|
| **00_quickstart.py** | Minimal, single-task run (ratio scorer). |
| **01_weighted_scoring.py** | Passing custom checklist weights for scoring. |
| **02_custom_agents.py** | Using different LLM providers or custom agent logic. |
| **03_reward_scorer.py** | Employing a reward model for scoring. |
| **04_batch_processing.py** | Running the orchestrator on multiple prompts from a CSV file and exporting results. |

### Running the Examples

To run any of the examples:

1.  Ensure you have `pydantic-ai-orchestrator` installed and your virtual environment activated.
2.  Set your OpenAI API key as an environment variable:
    ```bash
    export ORCH_OPENAI_API_KEY='sk-...'
    ```
3.  Navigate to the root of the repository and run the desired example script using Python:
    ```bash
    python examples/00_quickstart.py
    ```
    Replace `00_quickstart.py` with the script you wish to run.

These examples are a great way to learn about the practical application of different features and configurations.

## License
MIT

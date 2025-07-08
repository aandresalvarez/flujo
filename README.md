<div align="center">
  <img src="assets/flujo.png" alt="Flujo Logo" width="180"/>
</div>

# Flujo

A powerful Python library for orchestrating AI workflows using Pydantic models.
The `flujo` package (repository hosted at
[`github.com/aandresalvarez/flujo`](https://github.com/aandresalvarez/flujo))
provides utilities to manage multi-agent pipelines with minimal setup.

## Features

- 📦 **Pydantic Native** – agents, tools, and pipeline context are all defined with Pydantic models for reliable type safety.
- 🔁 **Opinionated & Flexible** – the `Default` recipe gives you a ready‑made workflow while the DSL lets you build any pipeline.
- 🏗️ **Production Ready** – retries, telemetry, and quality controls help you ship reliable systems.
- 🗄️ **Optimized State Backend** – high-performance SQLite backend with indexing and admin queries for large-scale deployments.
- 🧠 **Intelligent Evals** – automated scoring and self‑improvement powered by LLMs.

## Quick Start

### Installation

```bash
pip install flujo
```

### Basic Usage

> **Note:** The class-based `AgenticLoop` is deprecated. Use the new `make_agentic_loop_pipeline` factory function for full transparency, composability, and future YAML/AI support.

```python
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo import make_agent_async, init_telemetry
from flujo.domain.commands import AgentCommand
from pydantic import TypeAdapter

# Enable telemetry (optional but recommended)
init_telemetry()

async def search_agent(query: str) -> str:
    """A simple tool agent that returns information."""
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language."
    return "No information found."

PLANNER_PROMPT = """
You are a research assistant. Use the `search_agent` tool to gather facts.
When you know the answer, issue a `FinishCommand` with the final result.
"""
planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    TypeAdapter(AgentCommand),
)

# Create the pipeline using the factory
pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry={"search_agent": search_agent}
)

# Run the pipeline
result = await run_agentic_loop_pipeline(pipeline, "What is Python?")
print(result)
```

### Pipeline Example

```python
from flujo import Flujo, step, PipelineResult

@step
async def to_uppercase(text: str) -> str:
    """A simple step to convert text to uppercase."""
    return text.upper()

@step
async def add_enthusiasm(text: str) -> str:
    """A second step to add emphasis."""
    return f"{text}!!!"

# Compose the pipeline using the decorator-created steps
pipeline = to_uppercase >> add_enthusiasm
runner = Flujo(pipeline)

# Run the pipeline
result: PipelineResult[str] = runner.run("hello world")

print(f"Final pipeline output: {result.step_history[-1].output}")
# Expected Output: Final pipeline output: HELLO WORLD!!!
```

## Documentation

### Getting Started

- [Installation Guide](docs/installation.md): Detailed installation instructions
- [Quickstart Guide](docs/quickstart.md): Get up and running quickly
- [Core Concepts](docs/concepts.md): Understand the fundamental concepts

### User Guides

- [Usage Guide](docs/usage.md): Learn how to use the library effectively
- [Configuration Guide](docs/configuration.md): Configure the orchestrator
- [Tools Guide](docs/tools.md): Create and use tools with agents
- [Pipeline DSL Guide](docs/pipeline_dsl.md): Build custom workflows
- [Scoring Guide](docs/scoring.md): Implement quality control
- [Telemetry Guide](docs/telemetry.md): Monitor and analyze usage

### Advanced Topics

- [Extending Guide](docs/extending.md): Create custom components
- [Use Cases](docs/use_cases.md): Real-world examples and patterns
- [API Reference](docs/api_reference.md): Detailed API documentation
- [Troubleshooting Guide](docs/troubleshooting.md): Common issues and solutions

### Development

- [Contributing Guide](CONTRIBUTING.md): How to contribute to the project
- [Development Guide](docs/dev.md): Development setup and workflow
- [Testing Guide](docs/testing_guide.md): Best practices for writing tests
- [Code of Conduct](CODE_OF_CONDUCT.md): Community guidelines
- [License](LICENSE): Dual License (AGPL-3.0 + Commercial)

## Examples

Check out the [examples directory](examples/) for more usage examples:

| Script | What it shows |
| ------ | ------------- |
| [**00_quickstart.py**](examples/00_quickstart.py) | Hello World with the AgenticLoop recipe. |
| [**01_weighted_scoring.py**](examples/01_weighted_scoring.py) | Weighted scoring to prioritize docstrings. |
| [**02_custom_agents.py**](examples/02_custom_agents.py) | Building creative agents with custom prompts. |
| [**03_reward_scorer.py**](examples/03_reward_scorer.py) | Using an LLM judge via RewardScorer. |
| [**04_batch_processing.py**](examples/04_batch_processing.py) | Running multiple workflows concurrently. |
| [**05_pipeline_sql.py**](examples/05_pipeline_sql.py) | Pipeline DSL with SQL validation plugin. |
| [**06_typed_context.py**](examples/06_typed_context.py) | Sharing state with Typed Pipeline Context. |
| [**07_loop_step.py**](examples/07_loop_step.py) | Iterative refinement using LoopStep. |
| [**08_branch_step.py**](examples/08_branch_step.py) | Dynamic routing with ConditionalStep. |

Looking for more community resources? Check out the [Awesome Flujo list](AWESOME-FLUJO.md).

## Requirements

- Python 3.11 or higher
- OpenAI API key (for OpenAI models)
- Anthropic API key (for Claude models)
- Google API key (for Gemini models)

## Installation

### Basic Installation

```bash
pip install flujo
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/flujo.git
cd flujo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Set up the Hatch environment for tooling
pip install hatch
make setup
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- [Documentation](https://flujo.readthedocs.io/)
- [Issue Tracker](https://github.com/aandresalvarez/flujo/issues)
- [Discussions](https://github.com/aandresalvarez/flujo/discussions)
- [Discord](https://discord.gg/...)

## License

This project is dual-licensed:

1. **Open Source License**: GNU Affero General Public License v3.0 (AGPL-3.0)
   - Free for open-source projects
   - Requires sharing of modifications
   - Suitable for non-commercial use

2. **Commercial License**
   - For businesses and commercial use
   - Includes support and updates
   - No requirement to share modifications
   - Contact for pricing and terms

For commercial licensing, please contact: alvaro@example.com

See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE](COMMERCIAL_LICENSE) for details.

## Acknowledgments

- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [OpenAI](https://openai.com/) for GPT models
- [Anthropic](https://www.anthropic.com/) for Claude models
- [Google](https://ai.google.dev/) for Gemini models
- All our contributors and users

## Citation

If you use this project in your research, please cite:

```bibtex
@software{flujo,
  author = {Alvaro Andres Alvarez},
  title = {Flujo},
  year = {2024},
  url = {https://github.com/aandresalvarez/flujo}
}
```

## 🚀 New: Run Custom Pipelines from the CLI

You can now run any custom pipeline directly from the command line—no boilerplate needed!

```sh
flujo run my_pipeline.py --input "Hello world" --context-model MyContext
```

- Define your pipeline in a Python file (see below for an example)
- Pass input and context data flexibly
- See [usage.md](docs/usage.md#running-custom-pipelines-from-the-cli-flujo-run) for full details

**Example pipeline file:**

```python
from flujo import step, Pipeline
from flujo.domain.models import PipelineContext
from pydantic import Field

class MyContext(PipelineContext):
    counter: int = Field(default=0)

@step
async def inc(data: str, *, context: MyContext | None = None) -> str:
    if context:
        context.counter += 1
    return data.upper()

pipeline = inc >> inc
```

**Run it:**

```sh
flujo run my_pipeline.py --input "hello" --context-model MyContext
```

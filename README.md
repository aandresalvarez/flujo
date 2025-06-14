# Pydantic AI Orchestrator

A powerful Python library for orchestrating AI workflows using Pydantic models.
The `pydantic-ai-orchestrator` package (repository hosted at
[`github.com/aandresalvarez/rloop`](https://github.com/aandresalvarez/rloop))
provides utilities to manage multi-agent pipelines with minimal setup.

## Features

- 🤖 **AI Agent Management**: Create and manage AI agents with different roles and capabilities
- 🔄 **Pipeline DSL**: Build complex AI workflows using a simple domain-specific language
- 🛠️ **Tool Integration**: Extend agent capabilities with custom tools
- 📊 **Quality Control**: Built-in validation and scoring mechanisms
- 📈 **Telemetry**: Monitor performance and usage with built-in metrics
- 🔒 **Type Safety**: Leverage Pydantic for runtime type checking and validation
- 🚀 **Async Support**: Built-in support for asynchronous operations
- 📚 **Extensive Documentation**: Comprehensive guides and examples

## Quick Start

### Installation

```bash
pip install pydantic-ai-orchestrator
```

### Basic Usage

```python
from pydantic_ai_orchestrator import (
    Orchestrator, Task,
    review_agent, solution_agent, validator_agent,
    init_telemetry,
)

# Optional: enable telemetry for your application
init_telemetry()

# Assemble an orchestrator with the library-provided agents. The class
# runs a fixed pipeline: Review -> Solution -> Validate.
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
)

# Define a task
task = Task(prompt="Write a Python function to calculate Fibonacci numbers")

# Run synchronously
best_candidate = orch.run_sync(task)

# Print the result
if best_candidate:
    print("Solution:\n", best_candidate.solution)
    if best_candidate.checklist:
        print("\nQuality Checklist:")
        for item in best_candidate.checklist.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            print(f"  - {item.description:<45} {status}")
else:
    print("No solution found.")
```

### Pipeline Example

```python
from pydantic_ai_orchestrator import (
    Step, PipelineRunner, Task,
    review_agent, solution_agent, validator_agent,
)

# Build a custom pipeline using the Step DSL. This mirrors the internal
# workflow used by :class:`Orchestrator` but is fully configurable.
custom_pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

pipeline_runner = PipelineRunner(custom_pipeline)

# Run synchronously; PipelineRunner returns a PipelineResult.
pipeline_result = pipeline_runner.run(
    "Generate a REST API using FastAPI for a to-do list application."
)

print("\nPipeline Execution History:")
for step_res in pipeline_result.step_history:
    print(f"- Step '{step_res.name}': Success={step_res.success}")

if len(pipeline_result.step_history) > 1 and pipeline_result.step_history[1].success:
    solution_output = pipeline_result.step_history[1].output
    print("\nGenerated Solution:\n", solution_output)
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

- [Contributing Guide](docs/contributing.md): How to contribute to the project
- [Development Guide](docs/dev.md): Development setup and workflow
- [Code of Conduct](CODE_OF_CONDUCT.md): Community guidelines
- [License](LICENSE): MIT License

## Examples

Check out the [examples directory](examples/) for more usage examples:

- [Basic Usage](examples/basic.py): Simple orchestrator usage
- [Pipeline Example](examples/pipeline.py): Custom pipeline creation
- [Tool Integration](examples/tools.py): Using tools with agents
- [Async Operations](examples/async.py): Asynchronous workflows
- [Quality Control](examples/quality.py): Validation and scoring
- [Telemetry](examples/telemetry.py): Monitoring and metrics

## Requirements

- Python 3.11 or higher
- OpenAI API key (for OpenAI models)
- Anthropic API key (for Claude models)
- Google API key (for Gemini models)

## Installation

### Basic Installation

```bash
pip install pydantic-ai-orchestrator
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/rloop.git
cd rloop

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## Support

- [Documentation](https://pydantic-ai-orchestrator.readthedocs.io/)
- [Issue Tracker](https://github.com/aandresalvarez/rloop/issues)
- [Discussions](https://github.com/aandresalvarez/rloop/discussions)
- [Discord](https://discord.gg/...)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [OpenAI](https://openai.com/) for GPT models
- [Anthropic](https://www.anthropic.com/) for Claude models
- [Google](https://ai.google.dev/) for Gemini models
- All our contributors and users

## Citation

If you use this project in your research, please cite:

```bibtex
@software{pydantic_ai_orchestrator,
  author = {Alvaro Andres Alvarez},
  title = {Pydantic AI Orchestrator},
  year = {2024},
  url = {https://github.com/aandresalvarez/rloop}
}
```

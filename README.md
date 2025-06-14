# Pydantic AI Orchestrator

A powerful Python library for orchestrating AI workflows using Pydantic models.

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
from pydantic_ai_orchestrator import Orchestrator

# Create an orchestrator
orchestrator = Orchestrator(
    model="openai:gpt-4",
    temperature=0.7
)

# Run a task
result = orchestrator.run(
    "Write a Python function to calculate fibonacci numbers"
)

# Print the result
print(result.solution)
```

### Pipeline Example

```python
from pydantic_ai_orchestrator import Step, PipelineRunner

# Create a pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

# Create a runner
runner = PipelineRunner(pipeline)

# Run the pipeline
result = runner.run("Generate a REST API using FastAPI")
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

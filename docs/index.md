# flujo

Production-ready orchestration for Pydantic-based AI agents with intelligent evaluation and self-improvement capabilities.

## Overview

The `flujo` is a powerful Python library that provides a structured approach to building and managing multi-agent AI workflows. Built on top of Pydantic for type safety and data validation, it offers both high-level orchestration patterns and flexible pipeline construction tools.

## Features

- **🔧 Pydantic Native** – Everything from agents to pipeline context is defined with Pydantic models for reliable type safety
- **🎯 Opinionated & Flexible** – Start with the built-in `Default` recipe for common patterns or compose custom flows using the Pipeline DSL
- **🚀 Production Ready** – Built-in retries, telemetry, scoring, and quality controls help you deploy reliable systems
- **🧠 Intelligent Evals** – Automated evaluation and self-improvement powered by LLMs
- **⚡ High Performance** – Async-first design with efficient concurrent execution
- **🔌 Extensible** – Plugin system for custom validation, scoring, and tools

## Quick Start

### Installation

```bash
pip install flujo
```

### Basic Usage

```python
from flujo.recipes import Default
from flujo import (
    Task,
    review_agent, solution_agent, validator_agent, reflection_agent,
    init_telemetry,
)

# Initialize telemetry (optional)
init_telemetry()

# Create the default recipe with built-in agents
orch = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
)

# Define and run a task
task = Task(prompt="Write a Python function to calculate Fibonacci numbers")
result = orch.run_sync(task)

if result:
    print(f"Solution: {result.solution}")
    print(f"Quality Score: {result.score}")
```

### Command Line Interface

```bash
# Solve a task quickly
flujo solve "Create a REST API for a todo app"

# Run benchmarks
flujo bench "Write a sorting algorithm" --rounds 5

# Show configuration
flujo show-config

# Generate improvement suggestions
flujo improve pipeline.py dataset.py
```

## Core Concepts

- **Default recipe**: High-level coordinator for standard multi-agent workflows
- **Pipeline DSL**: Flexible system for building custom agent workflows
- **Agents**: Specialized AI models with specific roles (review, solution, validation, reflection)
- **Tasks**: Input specifications with prompts and metadata
- **Candidates**: Generated solutions with quality assessments
- **Scoring**: Multiple strategies for evaluating solution quality
- **Telemetry**: Built-in monitoring and observability

## Architecture

The library follows a clean architecture with clear separation of concerns:

- **Application Layer**: Default recipe, Flujo engine, and high-level coordination
- **Domain Layer**: Core models, pipeline DSL, and business logic
- **Infrastructure Layer**: Agents, settings, telemetry, and external integrations
- **CLI Layer**: Command-line interface for common operations

## Next Steps

Choose your path based on your needs:

### For Beginners
1. **[Installation Guide](installation.md)** - Get set up quickly
2. **[Quickstart Guide](quickstart.md)** - Your first orchestration in 5 minutes
3. **[Tutorial](tutorial.md)** - Comprehensive guided tour

### For Developers
1. **[Core Concepts](concepts.md)** - Understand the architecture
2. **[Pipeline DSL Guide](pipeline_dsl.md)** - Build custom workflows
3. **[API Reference](api_reference.md)** - Detailed documentation

### For Advanced Users
1. **[Extending Guide](extending.md)** - Create custom components
2. **[Intelligent Evals](intelligent_evals.md)** - Automated evaluation and improvement
3. **[Telemetry Guide](telemetry.md)** - Monitor and optimize performance

## Support & Community

- **📚 Documentation**: Complete guides and API reference
- **🐛 Issues**: [GitHub Issues](https://github.com/aandresalvarez/flujo/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/aandresalvarez/flujo/discussions)
- **📦 Package**: [PyPI](https://pypi.org/project/flujo/)

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
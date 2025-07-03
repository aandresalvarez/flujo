# Flujo Documentation

Welcome to **Flujo** - a powerful Python library for orchestrating AI workflows with type safety and production-ready features.

## What is Flujo?

**Flujo** is a Python library that provides a structured approach to building and managing multi-agent AI workflows. Built on top of Pydantic for type safety and data validation, it offers both high-level orchestration patterns and flexible pipeline construction tools.

### Key Features

- **🔧 Pydantic Native** – Everything from agents to pipeline context is defined with Pydantic models for reliable type safety
- **🎯 Opinionated & Flexible** – Start with the built-in `Default` recipe for common patterns or compose custom flows using the Pipeline DSL
- **🚀 Production Ready** – Built-in retries, telemetry, scoring, and quality controls help you deploy reliable systems
- **🧠 Intelligent Evals** – Automated evaluation and self-improvement powered by LLMs
- **⚡ High Performance** – Async-first design with efficient concurrent execution
- **🔌 Extensible** – Plugin system for custom validation, scoring, and tools

### How It Works

Flujo orchestrates AI workflows through a multi-agent approach where specialized agents (Review, Solution, Validation, Reflection) work together to solve complex tasks. The system uses a shared, typed context to maintain state across pipeline steps, enabling sophisticated workflows with built-in quality control and self-improvement capabilities.

## Quick Start

### Installation

```bash
pip install flujo
```

### Your First Workflow

```python
from flujo.recipes import Default
from flujo import Task, init_telemetry

# Initialize telemetry (optional but recommended)
init_telemetry()

# Create the default recipe with built-in agents
orch = Default()

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
```

## Documentation Structure

### 🚀 Getting Started
- **[Installation Guide](installation.md)** - Set up Flujo and configure your environment
- **[Quickstart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Core Concepts](concepts.md)** - Understand the fundamental building blocks
- **[Configuration Guide](configuration.md)** - Configure API keys and settings

### 📚 Usage Guides
- **[Basic Usage](usage.md)** - Learn the essential patterns and workflows
- **[Tutorial](tutorial.md)** - Comprehensive guided tour from simple to advanced
- **[CLI Reference](cli.md)** - Complete command-line interface documentation
- **[Pipeline DSL Guide](pipeline_dsl.md)** - Build custom workflows with the DSL
- **[Tools Guide](tools.md)** - Create and integrate external tools
- **[Scoring Guide](scoring.md)** - Implement quality control and evaluation
- **[Telemetry Guide](telemetry.md)** - Monitor and analyze performance

### 🏗️ Architecture & Development
- **[Architecture Overview](architecture.md)** - Understand the system design
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Extending Guide](extending.md)** - Create custom components and plugins
- **[Testing Guide](testing_guide.md)** - Best practices for testing workflows

### 🍳 Cookbook & Examples
- **[Cost Control](cookbook/cost_control.md)** - Optimize costs while maintaining quality
- **[Human-in-the-Loop](cookbook/hitl_simple_approval.md)** - Integrate human feedback
- **[Advanced Patterns](cookbook/advanced_prompting.md)** - Sophisticated workflow patterns
- **[Real-time Applications](cookbook/realtime_chatbot.md)** - Build streaming applications

### 🔧 Development & Contributing
- **[Contributing Guide](dev.md)** - How to contribute to Flujo
- **[Development Setup](dev.md#setting-up-a-development-environment)** - Set up your development environment
- **[Documentation Guide](documentation_guide.md)** - How to write and maintain documentation
- **[Testing Guide](testing_guide.md)** - Testing strategies and best practices

### 📖 Reference
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Migration Guides](migration/)** - Upgrade guides for version changes
- **[Use Cases](use_cases.md)** - Real-world examples and patterns

## What's New

- **v2.1:** Chain entire pipelines together using the `>>` operator: `pipeline1 >> pipeline2`
- **v2.0:** Enhanced Pipeline DSL with improved type safety and performance
- **v1.9:** Intelligent evaluation system with automated self-improvement

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
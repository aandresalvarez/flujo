# The Framework for AI Systems That Learn

Go beyond static prompts. Flujo provides the structure, state, and observability to build AI agents that analyze their own performance, adapt over time, and improve with every run.

## Key Features

- Compose pipelines using a simple DSL
- Share and validate state with Pydantic models
- Integrate loops, branching, and human-in-the-loop patterns
- Agent infrastructure with factory functions
- Centralized prompt management
- **Enhanced serialization with global custom serializer registry**
- **Rich internal tracing and visualization (FSD-12)**
- **Integrated cost and token usage tracking**

## Getting Started

- [Quickstart](quickstart.md) - Get up and running in minutes
- [Usage Guide](../user_guide/usage.md) - Learn how to use the library
- [Agent Infrastructure](../advanced/agent_infrastructure.md) - Understand the agent system
- [Concepts](../user_guide/concepts.md) - Core concepts and architecture
- [Architect (Generate Blueprints)](../user_guide/architect.md) - Create YAML blueprints from natural language goals

## Advanced Features

- [Advanced Serialization](../cookbook/advanced_serialization.md) - Handle custom types and complex serialization scenarios
- [SQLite Backend](../guides/sqlite_backend_guide.md) - Production-ready persistence with observability
- [Migration Guide](../migration/v0.7.0.md) - Upgrade to the latest serialization features
- **[Rich Tracing & Debugging](../implementation-results/FSD-12_IMPLEMENTATION_RESULTS.md)** - Debug and analyze pipeline execution with hierarchical traces
- **[Cost Tracking Guide](../advanced/cost_tracking_guide.md)** - Monitor and control spending on LLM operations

Use the navigation on the left to explore the guides, examples, and API reference.

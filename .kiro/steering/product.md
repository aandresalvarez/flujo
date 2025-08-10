# Product Overview

Flujo is a Python framework for building AI-powered applications that transforms AI agents into production-grade "digital employees." It provides structured workflows, persistent state management, cost governance, and continuous improvement capabilities.

## Core Value Proposition

- **Durability**: Automatic state persistence with SQLite backend, resuming exactly where tasks left off
- **Governance**: Strict usage limits and proactive cost guards to prevent overspending
- **Improvement Loop**: `flujo improve` command analyzes failures and generates concrete suggestions
- **Safety Rails**: Human-in-the-loop capabilities and conditional branching for edge cases
- **Observability**: Real-time updates via event hooks and full run histories with `flujo lens` CLI

## Key Features

- Type-safe AI agent orchestration with Pydantic validation
- Async-first design with high performance optimizations (uvloop, orjson, blake3)
- Built-in caching and parallel execution capabilities
- Comprehensive error handling and retry logic
- Production-ready telemetry and observability
- CLI tools for running, monitoring, and improving pipelines

## Target Use Cases

- Building autonomous AI workflows that handle routine work
- Creating budget-aware AI applications with cost controls
- Developing stateful AI systems that persist across runs
- Implementing complex multi-agent workflows with human oversight

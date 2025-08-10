# Project Structure

## Root Directory Layout

```
flujo/                    # Main package directory
├── application/          # Application layer (runners, evaluators, orchestration)
├── domain/              # Domain models and business logic
├── infra/               # Infrastructure (agents, config, telemetry)
├── cli/                 # Command-line interface
├── state/               # State management and backends
├── utils/               # Utility functions and helpers
├── testing/             # Testing utilities and assertions
├── tracing/             # Distributed tracing support
├── telemetry/           # Observability and monitoring
└── recipes/             # Pre-built pipeline factories

examples/                # Usage examples and demos
tests/                   # Test suite
├── unit/                # Unit tests
├── integration/         # Integration tests
├── e2e/                 # End-to-end tests
├── benchmarks/          # Performance benchmarks
└── regression/          # Regression tests

docs/                    # Documentation
scripts/                 # Development and maintenance scripts
```

## Key Architecture Layers

### Domain Layer (`flujo/domain/`)
- **models.py**: Core domain models (PipelineResult, Step, etc.)
- **dsl/**: Domain-specific language for pipeline construction
- **agent_protocol.py**: Agent interface definitions
- **validation.py**: Domain validation logic

### Application Layer (`flujo/application/`)
- **runner.py**: Main Flujo runner class
- **core/**: Core execution engine and optimizations
- **evaluators.py**: Pipeline evaluation logic
- **parallel.py**: Parallel execution support

### Infrastructure Layer (`flujo/infra/`)
- **agents.py**: Agent factory functions
- **config.py**: Configuration management
- **backends.py**: State backend implementations
- **telemetry.py**: Observability setup

## File Naming Conventions

- **Snake case**: All Python files use snake_case
- **Descriptive names**: Files clearly indicate their purpose
- **Layer prefixes**: Some files prefixed by layer (e.g., `base_model.py`)
- **Test mirrors**: Test files mirror source structure (`test_*.py`)

## Import Patterns

- **Relative imports**: Within packages, use relative imports
- **Public API**: Main exports through `__init__.py` files
- **Type imports**: Use `from __future__ import annotations` for forward references
- **Optional dependencies**: Graceful handling of missing optional deps

## Configuration Files

- **pyproject.toml**: Python project configuration and dependencies
- **flujo.toml**: Flujo-specific configuration (models, costs, settings)
- **Makefile**: Development workflow commands
- **.pre-commit-config.yaml**: Pre-commit hooks for code quality

## Testing Structure

- **Markers**: Tests marked as `fast`, `slow`, `serial`, `benchmark`, `e2e`
- **Parallel execution**: Fast tests run in parallel, slow/serial tests run sequentially
- **Coverage**: HTML coverage reports generated in `htmlcov/`
- **Fixtures**: Shared test fixtures in `conftest.py` files

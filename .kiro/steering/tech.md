# Technology Stack

## Core Technologies

- **Python**: 3.11+ required, async-first architecture
- **Pydantic**: v2.11+ for type validation and structured outputs
- **pydantic-ai**: v0.4.7+ for AI agent orchestration
- **SQLite**: Default state backend via aiosqlite
- **Performance**: uvloop, orjson, blake3 for optimization

## Build System

- **Package Manager**: `uv` (primary), pip (fallback)
- **Build Backend**: Hatchling
- **Dependency Management**: pyproject.toml with optional extras

## Key Dependencies

- **CLI**: typer, rich for command-line interface
- **Testing**: pytest, pytest-asyncio, pytest-xdist, hypothesis
- **Quality**: ruff (formatting/linting), mypy (type checking)
- **Async**: tenacity for retry logic, aiosqlite for database
- **Observability**: OpenTelemetry, Prometheus, Logfire (optional)

## Common Commands

### Development Setup
```bash
make install          # Install with uv
make install-robust   # Install with verification
pip install -e ".[dev,bench,docs]"  # Alternative pip install
```

### Code Quality
```bash
make format          # Auto-format with ruff
make lint           # Lint code
make typecheck      # Run mypy
make all           # Run all quality checks
```

### Testing
```bash
make test           # Run all tests
make test-fast      # Run fast tests in parallel (excludes slow/benchmark)
make test-unit      # Unit tests only
make test-integration  # Integration tests only
make testcov        # Tests with coverage report
```

### CLI Usage
```bash
flujo run pipeline.py --input "data"
flujo lens show     # View run history
flujo improve       # Analyze failures and suggest improvements
```

## Architecture Patterns

- **Domain-Driven Design**: Clear separation between domain, application, and infrastructure
- **Async/Await**: All core operations are async for performance
- **Type Safety**: Extensive use of Pydantic models and mypy
- **Decorator Pattern**: `@step` decorator for pipeline components
- **Factory Pattern**: Pipeline factories for common workflows
- **Observer Pattern**: Event hooks for monitoring and telemetry
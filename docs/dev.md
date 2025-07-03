# Development & Contributing Guide

Welcome to the Flujo development community! This guide will help you set up a development environment, understand the codebase, and contribute effectively.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Quality & Testing](#code-quality--testing)
- [Documentation](#documentation)
- [Package Management](#package-management)
- [Contributing Guidelines](#contributing-guidelines)
- [Architecture for Developers](#architecture-for-developers)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic familiarity with Python development

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/flujo.git
cd flujo

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python --version                    # should print 3.11.x
```

> **Note:** If `python3.11` isn't available, install it via:
> - macOS: `brew install python@3.11`
> - Linux: Use your distribution's package manager
> - Windows: Download from python.org
> - Or use pyenv: `pyenv install 3.11.x`

## Development Environment

### Initial Setup

Flujo uses [Hatch](https://hatch.pypa.io/) to manage the development environment and command scripts. The provided `Makefile` simply calls these `hatch` scripts for convenience.

```bash
pip install hatch
make setup         # create the Hatch environment and install git hooks
```

The `make setup` command will:
- Install all required development, testing, and documentation dependencies
- Set up pre-commit hooks for code quality and secret scanning
- Ensure your environment matches the CI pipeline

> **Tip:** Always use `make setup` after pulling changes to dependencies or when setting up a new environment.

### Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Add your API keys to .env
# - OpenAI
# - Cohere
# - Vertex AI
# - Other providers as needed
```

The orchestrator automatically loads this file via **python-dotenv**.

### Available Commands

```bash
# Development workflow
make quality        # run format, lint, types and security checks
make test           # run the test suite
make test args="-k my_test"  # pass extra pytest arguments

# Documentation
make docs-serve     # start docs server at http://127.0.0.1:8000
make docs-build     # generate static site

# Package management
make package        # creates wheel and sdist in dist/
make clean-package  # removes build artifacts

# Help
make help           # see all available commands
```

## Code Quality & Testing

### Testing

Run the test suite with:

```bash
make test
```

Pass additional arguments to `pytest` using the `args` variable:

```bash
make test args="-k my_test"
make test args="--cov=flujo --cov-report=html"
```

> **Note:** Async tests are handled automatically by **pytest-asyncio**

### Code Quality Checks

#### All-in-One Quality Check
```bash
make quality                      # runs all quality checks
```

#### Individual Quality Checks
```bash
# Code Style
make lint        # check with Ruff
make format      # format code
make type-check  # static type checking with MyPy
```

### Security: Secret Detection

Our project uses `detect-secrets` to prevent API keys and other sensitive credentials from being committed. This hook runs automatically every time you make a commit.

**What to do if your commit is blocked:**

1. **If it's a real secret:**
   - **Do not bypass the hook.**
   - Remove the secret from your staged changes.
   - Place the secret in your local `.env` file (which is git-ignored) and access it via `os.getenv()` or `flujo.settings`.
   - Re-add the file and commit again.

2. **If it's a false positive (e.g., a test ID):**
   - Update the baseline to tell `detect-secrets` that this string is acceptable.
   - Run the following commands:
     ```bash
     detect-secrets scan . > .secrets.baseline
     git add .secrets.baseline
     ```
   - Commit your changes again.

### Troubleshooting: mypy and Third-Party Stubs

If you see errors from `mypy` about missing type stubs for third-party libraries, don't worry! The `pyproject.toml` is configured to ignore these using the `[[tool.mypy.overrides]]` section. If you add new dependencies that lack type stubs, add them to this list.

## Documentation

### Local Development
```bash
make docs-serve                 # start docs server at http://127.0.0.1:8000
```

### Building
```bash
make docs-build                # generate static site
```

### Writing Documentation

- Follow the [Documentation Guide](documentation_guide.md) for style guidelines
- Use Google-style docstrings for all public APIs
- Include examples in docstrings where helpful
- Update the documentation status when adding new features

## Package Management

### Building
```bash
# Build package
make package                   # creates wheel and sdist in dist/

# Clean build artifacts
make clean-package            # removes build artifacts
```

### Publishing (Maintainers Only)
```bash
# Test PyPI
make publish-test             # builds and uploads to TestPyPI

# PyPI
make publish                  # builds and uploads to PyPI
```

### GitHub Releases (Private Distribution)
```bash
# Create a new release (builds package and creates release)
make release RELEASE_NOTES="Your release notes here"

# Manage releases
make release-list            # list all releases
make release-view           # view current release details
make release-download      # download current release assets
make release-delete       # delete current release (requires confirmation)
```

### Release Process

#### Version Management
```toml
# pyproject.toml
[project]
version = "0.3.0"  # Follow semantic versioning (MAJOR.MINOR.PATCH)
```
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

#### Release Options

1. **PyPI Release (Public):**
   - Update version in `pyproject.toml`
   - `git commit -am "release: vX.Y.Z"`
   - `git tag vX.Y.Z && git push --tags`
   - GitHub Actions will handle the release

2. **GitHub Release (Private):**
   - Update version in `pyproject.toml`
   - `make release RELEASE_NOTES="Release notes"`
   - Install using: `pip install https://github.com/username/flujo/releases/download/vX.Y.Z/flujo-X.Y.Z-py3-none-any.whl`

## Contributing Guidelines

### Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly:
   ```bash
   make test
   make quality
   ```
5. **Update documentation** if needed
6. **Commit your changes** with clear, descriptive commit messages
7. **Push to your fork** and create a Pull Request

### Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(pipeline): add support for custom validators
fix(agents): resolve timeout issue with OpenAI API
docs(quickstart): improve installation instructions
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what the PR does and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs if needed
- **Breaking Changes**: Clearly mark and explain breaking changes

### Review Process

1. **Automated Checks**: All PRs must pass CI checks
2. **Code Review**: At least one maintainer must approve
3. **Documentation**: Ensure documentation is updated
4. **Testing**: Verify tests pass and coverage is maintained

## Architecture for Developers

### Project Structure

```
flujo/
├── application/          # High-level orchestration
│   ├── flujo_engine.py   # Core Flujo engine
│   ├── evaluators.py     # Evaluation systems
│   └── self_improvement.py # Self-improvement logic
├── domain/              # Core business logic
│   ├── models.py        # Core data models
│   ├── pipeline_dsl.py  # Pipeline DSL implementation
│   ├── scoring.py       # Scoring and evaluation
│   ├── commands.py      # Agent command protocol
│   └── validation.py    # Validation logic
├── infra/               # Infrastructure components
│   ├── agents.py        # Built-in agent implementations
│   ├── backends.py      # Execution backends
│   ├── settings.py      # Configuration management
│   └── telemetry.py     # Observability systems
├── recipes/             # Pre-built workflow patterns
│   ├── default.py       # Default recipe implementation
│   └── agentic_loop.py  # AgenticLoop pattern
├── cli/                 # Command-line interface
│   └── main.py          # CLI entry points
└── testing/             # Testing utilities
    └── assertions.py    # Test assertions and helpers
```

### Key Design Principles

1. **Type Safety First**: All components use Pydantic models
2. **Async-First Design**: All I/O operations are asynchronous
3. **Extensibility**: Plugin system for custom components
4. **Observability**: Built-in telemetry and tracing
5. **Production Ready**: Error handling, retries, and resource management

### Adding New Features

#### Adding a New Agent Type

1. **Define the agent** in `flujo/infra/agents.py`
2. **Add tests** in `tests/unit/test_agents.py`
3. **Update documentation** in relevant guides
4. **Add examples** if appropriate

#### Adding a New Recipe

1. **Create the recipe** in `flujo/recipes/`
2. **Add tests** in `tests/integration/`
3. **Update documentation** in the recipes section
4. **Add CLI support** if needed

#### Adding a New Tool

1. **Define the tool** following the Tool protocol
2. **Add tests** for the tool
3. **Update documentation** in the tools guide
4. **Add examples** in the cookbook

### Testing Strategy

#### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on edge cases and error conditions

#### Integration Tests
- Test component interactions
- Use real agents with mock backends
- Test end-to-end workflows

#### End-to-End Tests
- Test complete workflows
- Use real API calls (with VCR for recording)
- Test CLI functionality

### Performance Considerations

- **Async Operations**: Use async/await for all I/O
- **Caching**: Implement caching where appropriate
- **Resource Management**: Clean up resources properly
- **Monitoring**: Add telemetry for performance tracking

## Troubleshooting

### Common Issues

#### Environment Setup
```bash
# If you get import errors
pip install -e ".[dev]"

# If mypy fails
make type-check

# If tests fail
make test args="-v"
```

#### API Key Issues
- Ensure `.env` file exists and contains valid keys
- Check that keys have sufficient credits
- Verify API endpoints are accessible

#### Dependency Issues
```bash
# Clean and reinstall
make clean-package
make setup
```

### Getting Help

- **Documentation**: Check the [documentation](index.md)
- **Issues**: Search [GitHub Issues](https://github.com/aandresalvarez/flujo/issues)
- **Discussions**: Ask in [GitHub Discussions](https://github.com/aandresalvarez/flujo/discussions)
- **Code of Conduct**: Review our [Code of Conduct](CODE_OF_CONDUCT.md)

### Development Tips

1. **Use the Makefile**: Leverage the provided commands for consistency
2. **Run Tests Often**: Test your changes frequently
3. **Check Quality**: Run `make quality` before committing
4. **Document Changes**: Update docs when adding features
5. **Follow Patterns**: Study existing code for consistency

Thank you for contributing to Flujo! Your contributions help make AI workflow orchestration more accessible and powerful for everyone.

# Contributing

We welcome contributions! Please follow this guide.

## Development Setup

1.  **Clone the repo**
    ```bash
    git clone https://github.com/yourorg/pydantic-ai-orchestrator.git
    cd pydantic-ai-orchestrator
    ```

2.  **Install dependencies**
    This project uses Poetry for dependency management.
    ```bash
    pip install poetry
    poetry install
    ```

3.  **Set up environment**
    Copy the `.env.example` to `.env` and fill in your API keys. The CLI
    will automatically load variables from this file using `python-dotenv`.
    ```bash
    cp .env.example .env
    # Now edit .env
    ```

## Running Tests

We use `pytest` for testing.

```bash
poetry run pytest
```

To run with coverage:
```bash
poetry run pytest --cov=pydantic_ai_orchestrator
```

## Linting and Type Checking

We use `ruff` for linting/formatting and `mypy` for static type checking.

```bash
# Lint
poetry run ruff check .

# Format
poetry run ruff format .

# Type check
poetry run mypy .
```

## Release Flow

1.  Update `pyproject.toml` with the new version.
2.  Commit changes.
3.  Create a git tag: `git tag vX.Y.Z`
4.  Push the tag to the remote: `git push origin vX.Y.Z`
5.  The `release.yml` GitHub Actions workflow will automatically build and publish to PyPI. 
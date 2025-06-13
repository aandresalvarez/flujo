# Contributing & Local Dev Guide

Thanks for helping improve **pydantic-ai-orchestrator**! Follow the steps below to get a fully-featured development environment running on **Python 3.11+** in minutes.

---

## 1  Clone the repository

```bash
git clone https://github.com/yourorg/pydantic-ai-orchestrator.git
cd pydantic-ai-orchestrator
```

---

## 2  Create / activate a Python 3.11 virtual-env

> **Why?** Keeping deps isolated avoids system-level conflicts and matches CI.

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python --version                    # should print 3.11.x
```

*(If `python3.11` isn’t on your PATH, install it with Homebrew, pyenv, or python.org.)*

---

## 3  Install dependencies in *editable* mode

### Option A Poetry (recommended)

```bash
pip install --upgrade poetry
poetry install --with dev           # pulls in runtime + dev extras
# Every subsequent command can be run via `poetry run …`
```

### Option B Raw pip

```bash
python -m pip install -e ".[dev]"   # editable install + dev extras
# Commands below use `python -m` or `pipx run`; adapt as you prefer
```

Editable mode (`-e`) means any code change you make is picked up instantly without reinstalling.

---

## 4  Configure environment variables

```bash
cp .env.example .env
# add your OpenAI / Cohere / Vertex, etc. keys to .env
```

The orchestrator auto-loads this file via **python-dotenv**—no code changes needed.

---

## 5  Run the test suite

```bash
# Poetry users
poetry run pytest                   # quick run
poetry run pytest --cov=pydantic_ai_orchestrator  # with coverage report

# Plain venv users
python -m pytest
python -m pytest --cov=pydantic_ai_orchestrator
```

Async tests are handled automatically by **pytest-asyncio** (asyncio\_mode = auto).

---

## 6  Lint, format, and type-check

```bash
# Ruff — linter & formatter
poetry run ruff check .
poetry run ruff format .            # autofix formatting

# MyPy — static types
poetry run mypy .
```

*(Replace `poetry run …` with `python -m …` or `pipx run …` if you skipped Poetry.)*

---

## 7  Release flow (maintainers)

1. Bump `version` in **pyproject.toml**.
2. `git commit -am "release: vX.Y.Z"`
3. `git tag vX.Y.Z && git push --tags`
4. GitHub Actions (`release.yml`) will build wheels and publish to PyPI automatically.

---

### Troubleshooting tips

| Issue                                                     | Fix                                                               |
| --------------------------------------------------------- | ----------------------------------------------------------------- |
| `ERROR: Package requires a different Python`              | Activate a 3.11+ interpreter (`python --version`).                |
| `async def functions are not natively supported` in tests | Ensure `pytest-asyncio` is installed (included in `[dev]` extra). |
| Permission errors in `~/Library/Caches/pypoetry`          | `sudo chown -R $USER ~/Library/Caches/pypoetry` and rerun Poetry. |

Happy hacking! 🎉

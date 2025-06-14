# Contributing & Local Dev Guide

Thanks for helping improve **pydantic-ai-orchestrator**! Follow the steps below to get a fully-featured development environment running on **Python 3.11+** in minutes.

---

## 1  Clone the repository

```bash
git clone https://github.com/aandresalvarez/rloop.git
cd rloop
```

---

## 2  Create / activate a Python 3.11 virtual-env

> **Why?** Keeping deps isolated avoids system-level conflicts and matches CI.

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python --version                    # should print 3.11.x
```

*(If `python3.11` isn't on your PATH, install it with Homebrew, pyenv, or python.org.)*

---

## 3  Install dependencies in *editable* mode

The project supports multiple package managers. Choose your preferred one:

### Option A  Poetry (recommended)

```bash
pip install --upgrade poetry
make poetry-dev                     # installs runtime + dev + docs + bench extras
```

### Option B  UV (fastest)

```bash
pip install --upgrade uv
make uv-dev                         # fastest installation with uv
```

### Option C  Standard pip

```bash
make pip-dev                        # standard pip installation
```

All options install the package in editable mode (`-e`), meaning any code change you make is picked up instantly without reinstalling.

> **Note:** Run `make help` to see all available commands and their descriptions.

---

## 4  Configure environment variables

```bash
cp .env.example .env
# add your OpenAI / Cohere / Vertex, etc. keys to .env
```

The orchestrator auto-loads this file via **python-dotenv**—no code changes needed.

---

## 5  Run the test suite

Choose your preferred test runner:

```bash
# Poetry
make poetry-test                    # full test suite with coverage
make poetry-test-fast              # quick run without coverage
make poetry-test-unit              # unit tests only
make poetry-test-e2e              # end-to-end tests only
make poetry-test-bench            # benchmark tests only

# UV (fastest)
make uv-test                       # full test suite with coverage
make uv-test-fast                 # quick run without coverage
# ... (same options as poetry)

# Standard pip (default)
make test                         # full test suite with coverage
make test-fast                    # quick run without coverage
make test-unit                    # unit tests only
make test-e2e                     # end-to-end tests only
make test-bench                   # benchmark tests only
```

Async tests are handled automatically by **pytest-asyncio** (asyncio_mode = auto).

---

## 6  Code Quality Checks

Run all quality checks at once:

```bash
make quality                      # runs lint, format-check, type-check, and security
```

Or run individual checks:

```bash
# Linting and Formatting
make lint                        # check code style with Ruff
make format                      # format code automatically
make format-check               # check formatting without changing files (for CI)

# Type Checking
make type-check                 # run static type checking with MyPy

# Security
make security                   # check for security vulnerabilities
```

---

## 7  Documentation

```bash
make docs-serve                 # start local documentation server
make docs-build                # build documentation
```

Visit http://127.0.0.1:8000 to view the docs locally.

---

## 8  Release flow (maintainers)

1. Bump `version` in **pyproject.toml**
2. `git commit -am "release: vX.Y.Z"`
3. `git tag vX.Y.Z && git push --tags`
4. GitHub Actions (`release.yml`) will build wheels and publish to PyPI automatically

To build the package locally:

```bash
make build                     # build package using Hatch
```

---

## 9  Cleanup

When you need to clean up build artifacts or caches:

```bash
make clean                     # remove all build artifacts and caches
make clean-pyc                # remove Python cache files only
make clean-build             # remove build/dist artifacts only
make clean-test             # remove test artifacts only
make clean-docs            # remove built documentation only
make clean-cache          # remove tool caches (Ruff, MyPy) only
```

---

### Troubleshooting tips

| Issue                                                     | Fix                                                               |
| --------------------------------------------------------- | ----------------------------------------------------------------- |
| `ERROR: Package requires a different Python`              | Activate a 3.11+ interpreter (`python --version`).                |
| `async def functions are not natively supported` in tests | Ensure `pytest-asyncio` is installed (included in `[dev]` extra). |
| Permission errors in `~/Library/Caches/pypoetry`          | `sudo chown -R $USER ~/Library/Caches/pypoetry` and rerun Poetry. |
| Make command not found                                    | Install make: `brew install make` (macOS) or `apt install make` (Ubuntu) |
| Tool-specific errors                                      | Run `make clean-cache` to clear tool caches and try again         |

For more commands and options, run:

```bash
make help
```

Happy hacking! 🎉

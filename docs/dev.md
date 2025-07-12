# Contributing & Local Dev Guide

Thanks for helping improve **flujo**! This guide will help you set up a fully-featured development environment for Python 3.11+.

---

## 1. Clone and Setup

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

---

## 2. Development Environment Setup

Flujo uses [Hatch](https://hatch.pypa.io/) to manage the development
environment and command scripts. The provided `Makefile` simply calls these
`hatch` scripts for convenience.

```bash
pip install hatch
make setup         # create the Hatch environment and install git hooks
```

- `make setup` will:
  - Install all required development, testing, and documentation dependencies (including `pytest`, `pytest-asyncio`, `vcrpy`, `hypothesis`, etc.)
  - Set up pre-commit hooks for code quality and secret scanning
  - Ensure your environment matches the CI pipeline

> **Tip:** Always use `make setup` after pulling changes to dependencies or when setting up a new environment. This guarantees all tools (test, lint, type-check) will work as expected.

make quality        # run format, lint, types and security checks
make test           # run the test suite
# pass extra pytest arguments
make test args="-k my_test"

Run `make help` to see all available commands.

---

### Troubleshooting: mypy and Third-Party Stubs

If you see errors from `mypy` about missing type stubs for third-party libraries (e.g., `pydantic_ai`, `tenacity`, `logfire`), don't worry! The `pyproject.toml` is configured to ignore these using the `[[tool.mypy.overrides]]` section. If you add new dependencies that lack type stubs, add them to this list to keep type checking clean and focused on your code.

---

## 3. Environment Configuration

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

---

## 4. Testing

Run the test suite with:

```bash
make test
```

Pass additional arguments to `pytest` using the `args` variable:

```bash
make test args="-k my_test"
```

> **Note:** Async tests are handled automatically by **pytest-asyncio**

---

## 5. Code Quality

### All-in-One Quality Check
```bash
make quality                      # runs all quality checks
```

### Individual Quality Checks
```bash
# Code Style
make lint        # check with Ruff
make format      # format code
make type-check  # static type checking with MyPy
```

#### Security: Secret Detection

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
   - Commit your changes again. This will update the baseline for all contributors.

---

## 6. Documentation

```bash
# Local Development
make docs-serve                 # start docs server at http://127.0.0.1:8000

# Build
make docs-build                # generate static site
```

---

## 7. Package Management

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

> **Release Process Options:**
> 1. **PyPI Release (Public):**
>    - Update version in `pyproject.toml`
>    - `git commit -am "release: vX.Y.Z"`
>    - `git tag vX.Y.Z && git push --tags`
>    - GitHub Actions will handle the release
>
> 2. **GitHub Release (Private):**
>    - Update version in `pyproject.toml`
>    - `make release RELEASE_NOTES="Release notes"`
>    - Install using: `pip install https://github.com/username/flujo/releases/download/vX.Y.Z/flujo-X.Y.Z-py3-none-any.whl`

#### Detailed Release Process

1. **Version Management**
   ```toml
   # pyproject.toml
   [project]
   version = "0.3.0"  # Follow semantic versioning (MAJOR.MINOR.PATCH)
   ```
   - MAJOR: Breaking changes
   - MINOR: New features (backwards compatible)
   - PATCH: Bug fixes (backwards compatible)

2. **Release Notes Best Practices**
   ```markdown
   # Example Release Notes

   ## What's New
   - Added new AI agent orchestration features
   - Improved error handling in workflow execution
   - Enhanced documentation with usage examples

   ## Breaking Changes
   - Renamed `Agent.run()` to `Agent.execute()` for clarity
   - Updated configuration format in `config.yaml`

   ## Bug Fixes
   - Fixed memory leak in long-running workflows
   - Resolved race condition in parallel agent execution

   ## Dependencies
   - Updated pydantic to v2.7.0
   - Added new optional dependency: logfire>=0.3.0
   ```

3. **Release Checklist**
   - [ ] Update version in `pyproject.toml`
   - [ ] Update CHANGELOG.md
   - [ ] Run full test suite: `make test`
   - [ ] Check code quality: `make quality`
   - [ ] Build package: `make package`
   - [ ] Verify wheel contents: `unzip -l dist/flujo-*.whl`
   - [ ] Create release with notes
   - [ ] Test installation in clean environment
   - [ ] Update documentation if needed

4. **Testing the Release**
   ```bash
   # Create a clean virtual environment
   python -m venv test_env
   source test_env/bin/activate

   # Test PyPI installation
   pip install flujo==X.Y.Z

   # Test GitHub release installation
   pip install https://github.com/username/flujo/releases/download/vX.Y.Z/flujo-X.Y.Z-py3-none-any.whl
   ```

#### Troubleshooting Release Issues

| Issue | Solution |
|-------|----------|
| `Error: RELEASE_NOTES environment variable is required` | Provide release notes: `make release RELEASE_NOTES="Your notes"` |
| `Error: Release vX.Y.Z already exists` | Delete existing release: `make release-delete` or use a new version |
| `Error: No such file or directory: dist/flujo-*.whl` | Run `make package` first to build the distribution |
| `Error: Permission denied` | Ensure GitHub CLI is authenticated: `gh auth status` |
| `Error: Invalid version format` | Check version in `pyproject.toml` follows semantic versioning |
| `Error: Wheel installation fails` | Verify Python version compatibility in `pyproject.toml` |
| `Error: GitHub API rate limit exceeded` | Wait for rate limit reset or use GitHub token with higher limits |

#### Common Release Scenarios

1. **Hotfix Release**
   ```bash
   # 1. Update patch version
   # 2. Create release with focused notes
   make release RELEASE_NOTES="Hotfix: Fixed critical issue in agent execution"
   ```

2. **Feature Release**
   ```bash
   # 1. Update minor version
   # 2. Create comprehensive release notes
   make release RELEASE_NOTES="New features: Added support for custom agent types and improved workflow monitoring"
   ```

3. **Major Version Release**
   ```bash
   # 1. Update major version
   # 2. Create detailed release notes with migration guide
   make release RELEASE_NOTES="Major update: Completely redesigned agent system. See MIGRATION.md for upgrade instructions"
   ```

#### Security Considerations

1. **Private Distribution**
   - GitHub releases are private by default
   - Access control via GitHub repository permissions
   - Consider using GitHub Packages for better access management

2. **Package Signing**
   - Consider signing your releases for additional security
   - Use `twine` with GPG signing for PyPI releases
   - Document signing process for maintainers

3. **Dependency Management**
   - Regularly audit dependencies with `pip-audit`
   - Pin dependency versions in `pyproject.toml`
   - Document any security-related changes in release notes

#### Automated Release Workflows

The project supports automated releases through GitHub Actions. There are two workflows available:

1. **PyPI Release Workflow**
   ```yaml
   # .github/workflows/pypi-release.yml
   name: PyPI Release
   on:
     push:
       tags:
         - 'v*'  # Triggers on version tags

   jobs:
     release:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'
         - name: Install dependencies
           run: make pip-dev
         - name: Run tests
           run: make test
         - name: Build package
           run: make package
         - name: Publish to PyPI
           run: make publish
           env:
             TWINE_USERNAME: __token__
             TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
   ```
   - Triggered by pushing a version tag (e.g., `v0.3.0`)
   - Runs tests to ensure quality
   - Builds and publishes to PyPI automatically
   - Requires PyPI API token in repository secrets

2. **GitHub Release Workflow**
   ```yaml
   # .github/workflows/github-release.yml
   name: GitHub Release
   on:
     push:
       tags:
         - 'v*'  # Triggers on version tags

   jobs:
     release:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'
         - name: Install dependencies
           run: make pip-dev
         - name: Run tests
           run: make test
         - name: Build package
           run: make package
         - name: Create GitHub Release
           uses: softprops/action-gh-release@v1
           with:
             files: |
               dist/flujo-*.whl
               dist/flujo-*.tar.gz
             body_path: CHANGELOG.md
           env:
             GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
   ```
   - Also triggered by version tags
   - Creates GitHub releases automatically
   - Uses CHANGELOG.md for release notes
   - Uploads built packages as release assets

**How to Use Automated Releases:**

1. **Prepare for Release**
   ```bash
   # 1. Update version in pyproject.toml
   # 2. Update CHANGELOG.md
   # 3. Commit changes
   git commit -am "release: v0.3.0"

   # 4. Create and push tag
   git tag v0.3.0
   git push origin v0.3.0
   ```

2. **What Happens Automatically**
   - GitHub Actions workflow triggers
   - Tests run to verify quality
   - Package is built
   - Release is created (PyPI and/or GitHub)
   - Release notes are published
   - Assets are uploaded

3. **Required Setup**
   - PyPI API token in repository secrets (for PyPI releases)
   - GitHub token (automatically provided)
   - Proper permissions in repository settings

4. **Benefits**
   - Consistent release process
   - Automated testing before release
   - No manual upload steps
   - Release history tracking
   - Automatic changelog generation
   - Reduced human error

5. **Monitoring Releases**
   - Check Actions tab in GitHub repository
   - Review release artifacts
   - Verify PyPI/GitHub release pages
   - Monitor installation success

> **Note:** The automated workflows are configured in `.github/workflows/`. You can customize them based on your needs.

---

## 8. Maintenance

### Cleanup Commands
```bash
# Comprehensive cleanup
make clean                     # removes all artifacts and caches

# Selective cleanup
make clean-pyc                # Python cache files
make clean-build             # build/dist artifacts
make clean-test             # test artifacts
make clean-docs            # documentation
make clean-cache          # tool caches (Ruff, MyPy)
```

### Cache Management
```bash
# Clear specific tool caches
make clean-ruff             # Ruff cache
make clean-mypy            # MyPy cache
make clean-cache          # All tool caches
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ERROR: Package requires a different Python` | Ensure Python 3.11+ is active (`python --version`) |
| Async test failures | Verify `pytest-asyncio` is installed (included in `[dev]`) |
| Poetry cache permission errors | `sudo chown -R $USER ~/Library/Caches/pypoetry` |
| Make not found | Install: `brew install make` (macOS) or `apt install make` (Ubuntu) |
| Tool-specific errors | Run `make clean-cache` and retry |

For all available commands:
```bash
make help
```

Happy coding! 🚀

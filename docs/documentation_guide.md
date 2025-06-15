# Documentation Guide

This guide outlines the standards and procedures for maintaining the official documentation for `flujo`. Our goal is to provide clear, comprehensive, and up-to-date documentation for users and contributors.

## 1. Our Documentation Stack: MkDocs + `mkdocstrings`

We use a modern stack that is simple, powerful, and tightly integrated with our Python codebase.

* **[MkDocs](https://www.mkdocs.org/):** A fast, simple, and beautiful static site generator. It builds our documentation site from the Markdown files located in the `/docs` directory.
* **[`mkdocstrings`](https://mkdocstrings.github.io/):** A powerful MkDocs plugin that automatically generates professional API reference pages directly from the docstrings in our Python code. This ensures our API documentation is always in sync with the actual implementation.
* **[GitHub Pages](https://pages.github.com/):** Free, reliable hosting for our static documentation site, deployed automatically via GitHub Actions.

This stack works well because it builds on the Markdown and docstrings we already maintain.

## 2. How to Contribute to the Documentation

Documentation contributions are just as valuable as code. Here's how to help.

### Step 1: Edit or Add Markdown Files

All narrative documentation (tutorials, guides, specifications) lives in the `/docs` directory.

* To improve a guide, simply edit the corresponding `.md` file (e.g., `docs/usage.md`, `docs/tutorial.md`).
* To add a new page, create a new `.md` file in `/docs` and add it to the `nav` section of the `mkdocs.yml` file to make it appear in the site navigation.

### Step 2: Update Python Docstrings for API Reference

Our API reference is generated automatically. To improve it, you must edit the docstrings directly in the Python source code (e.g., in `flujo/domain/pipeline_dsl.py`).

We follow the **Google Python Style Guide** for docstrings. A good docstring includes:
* A one-line summary.
* A more detailed description of the class or function.
* An `Args:` section for parameters.
* A `Returns:` section for the return value.
* An optional `Raises:` section for exceptions.

```python
def my_function(param1: int, param2: str) -> bool:
    """This is a one-line summary of the function.

    This is a more detailed description that explains what the
    function does, its purpose, and any important details.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        A boolean value indicating success or failure.
    """
    # ... function code ...
```

### Step 3: Preview Your Changes Locally

Before submitting a pull request, always preview the documentation site locally to ensure your changes render correctly.

1. **Install dependencies:**
    ```bash
    # Ensure you have the 'docs' extras installed
    pip install -e ".[docs]"
    ```
2. **Serve the site:**
    ```bash
    mkdocs serve
    ```
3. **View in your browser:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to see a live-reloading preview of the documentation site.

## 3. The Automated Publishing Workflow

You do not need to manually publish the documentation. We have an automated CI/CD workflow that handles everything.

* **Trigger:** The workflow runs automatically on every `push` to the `main` branch.
* **Process:**
    1. The GitHub Actions runner checks out the latest code.
    2. It installs Python and all necessary dependencies, including `mkdocs` and `mkdocstrings`.
    3. It runs the `mkdocs build` command, which converts all Markdown files and docstrings into a static HTML site in a `./site` directory.
    4. It uses the `peaceiris/actions-gh-pages` action to automatically push the contents of the `./site` directory to the special `gh-pages` branch on our repository.
* **Result:** GitHub Pages automatically serves the content of the `gh-pages` branch as our live documentation site.

You can view the workflow definition in `.github/workflows/deploy-docs.yml`.

### The CI/CD Workflow (`deploy-docs.yml`)

For reference, here is the workflow that powers our documentation deployment:

```yaml
# in .github/workflows/deploy-docs.yml
name: Deploy Docs to GitHub Pages

on:
  push:
    branches:
      - main  # Trigger on every push to main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install documentation dependencies
        run: pip install "mkdocs-material" "mkdocstrings[python]"
      - name: Build the documentation site
        run: mkdocs build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          # Optional: Add a commit message for the gh-pages branch
          commit_message: "docs: Deploying documentation for ${{ github.sha }}"
```

This automated process ensures that our documentation is always a perfect reflection of the `main` branch.

### How to Integrate This into Your Project

1. **Save the File:** Save the content above as `docs/documentation_guide.md`.
2. **Update `mkdocs.yml`:** Add the new guide to your navigation so it's easily discoverable.

    ```yaml
    # in mkdocs.yml
    site_name: flujo
    nav:
      - Home: index.md
      - Usage: usage.md
      - Tutorial: tutorial.md
      - Use Cases: use_cases.md
      - Extending: extending.md
      - Development:
        - Contributing Guide: dev.md
        - Documentation Guide: documentation_guide.md # <-- Add this line
      - API Reference:
          - Default: 'flujo.recipes.default'
          # ... other API refs
    ```

This guide captures our best practices so they're easy for everyone to find.

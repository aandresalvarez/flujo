# Mypy Internal Error with `pydantic_ai`

## Description

When running `mypy` on a codebase that uses the `pydantic_ai` library, an internal error occurs. The error message points to a specific file within the `pydantic_ai` package (`pydantic_ai/toolsets/function.py`), suggesting an incompatibility or an issue that `mypy` cannot handle.

The error persists even when the files that directly import `pydantic_ai` are excluded from the `mypy` check via the `exclude` option in `pyproject.toml`.

## Steps to Reproduce

1.  Create a Python environment with the following packages:
    ```bash
    pip install mypy==1.3.0 pydantic-ai>=0.4.7
    ```
2.  Create a Python file (e.g., `main.py`) that imports and uses `pydantic_ai`:
    ```python
    from pydantic_ai import Agent

    # Minimal code to trigger the issue
    def get_agent() -> Agent:
        return Agent()

    if __name__ == "__main__":
        get_agent()
    ```
3.  Run `mypy` on the file:
    ```bash
    mypy main.py
    ```

## Expected Behavior

`mypy` should complete the type check without any internal errors, reporting any type errors found in the code.

## Actual Behavior

`mypy` crashes with the following internal error:

```
<path-to-site-packages>/pydantic_ai/toolsets/function.py:52: error: INTERNAL ERROR -- Please try using mypy master on GitHub:
https://mypy.readthedocs.io/en/stable/common_issues.html#using-a-development-mypy-build
If this issue continues with mypy master, please report a bug at https://github.com/python/mypy/issues
version: 1.3.0
<path-to-site-packages>/pydantic_ai/toolsets/function.py:52: : note: please use --show-traceback to print a traceback when reporting a bug
```

## Environment

*   **Python Version:** 3.11
*   **Mypy Version:** 1.3.0
*   **pydantic-ai Version:** >=0.4.7
*   **Operating System:** macOS (Darwin)

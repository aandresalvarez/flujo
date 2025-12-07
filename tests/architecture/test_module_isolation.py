"""Module isolation tests.

Verifies that a standard pipeline run in a fresh subprocess does not import the
deprecated optimization layer or the heavy psutil dependency.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Mark as slow since it spawns a subprocess
pytestmark = [pytest.mark.slow, pytest.mark.timeout(60)]


def test_standard_run_does_not_import_optimization() -> None:
    """
    Spawn a clean Python process, run a minimal pipeline, and assert that neither
    `psutil` nor `flujo.application.core.optimization` (or submodules) are loaded.
    """

    script = textwrap.dedent(
        """
        import sys
        from flujo import Pipeline, Step, Flujo

        async def simple_step(x: int) -> int:
            return x

        step = Step.from_callable(simple_step, name="noop")
        pipeline = Pipeline.from_step(step)
        runner = Flujo(pipeline)
        runner.run(42)

        forbidden = ("psutil",)
        forbidden_prefixes = ("flujo.application.core.optimization.",)

        def _is_forbidden(name: str) -> bool:
            return name in forbidden or any(name.startswith(prefix) for prefix in forbidden_prefixes)

        loaded = [name for name in sys.modules if _is_forbidden(name)]
        if loaded:
            print(f"VIOLATION: Found forbidden modules: {loaded}")
            sys.exit(1)

        print("SUCCESS")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            f"Isolation test failed (code {result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

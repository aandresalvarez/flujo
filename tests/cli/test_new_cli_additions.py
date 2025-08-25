from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import os


def _write_py_pipeline(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipe.py"
    p.write_text(content)
    return p


def test_top_level_validate_strict_default_and_json(tmp_path: Path) -> None:
    # Intentionally invalid: type mismatch between steps
    file = _write_py_pipeline(
        tmp_path,
        """
from flujo.domain.dsl import Step, Pipeline
async def a(x: int) -> int: return x
async def b(x: str) -> str: return x
pipeline = Pipeline.from_step(Step.from_callable(a, name="a")) >> Step.from_callable(b, name="b")
""",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",  # top-level alias
            str(file),
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    # Strict by default: invalid exits with EX_VALIDATION_FAILED (4)
    assert result.returncode == 4
    payload = json.loads(result.stdout or "{}")
    assert payload.get("is_valid") is False
    assert isinstance(payload.get("errors"), list) and len(payload["errors"]) >= 1


def test_run_dry_run_json_lists_steps(tmp_path: Path) -> None:
    file = _write_py_pipeline(
        tmp_path,
        """
from flujo.domain.dsl import Step, Pipeline
async def a(x: str) -> str: return x
async def b(x: str) -> str: return x
pipeline = Pipeline.from_step(Step.from_callable(a, name="a")) >> Step.from_callable(b, name="b")
""",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "run",
            str(file),
            "--dry-run",
            "--json",
            "--input",
            "",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout or "{}")
    assert payload.get("validated") is True
    assert isinstance(payload.get("steps"), list)
    assert set(payload.get("steps", [])) >= {"a", "b"}


def test_project_root_sys_path_injection_enables_imports(tmp_path: Path) -> None:
    # Arrange a project-like structure with a skills module
    skills_pkg = tmp_path / "skills"
    skills_pkg.mkdir(parents=True, exist_ok=True)
    (skills_pkg / "__init__.py").write_text("")
    (skills_pkg / "helpers.py").write_text(
        """
async def echo(x: str) -> str:
    return x
"""
    )
    file = _write_py_pipeline(
        tmp_path,
        """
from flujo.domain.dsl import Step, Pipeline
from skills.helpers import echo
pipeline = Pipeline.from_step(Step.from_callable(echo, name="s"))
""",
    )

    # Without --project, imports should fail with a helpful message
    bad = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(file), "--input", "hi"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    out = bad.stdout + bad.stderr
    assert (
        "Import error: module 'skills' not found" in out
        or "Import error: module 'skills.helpers' not found" in out
    )

    # With --project, sys.path injection should allow the import
    ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "--project",
            str(tmp_path),
            "run",
            str(file),
            "--input",
            "hi",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0
    # Should produce JSON; we won't assert schema, just that it parses
    _ = json.loads(ok.stdout or "{}")


def test_verbose_and_trace_show_traceback(tmp_path: Path) -> None:
    # Force an import error to trigger traceback display
    file = _write_py_pipeline(
        tmp_path,
        """
from flujo.domain.dsl import Step, Pipeline
from skills.missing import nope  # noqa: F401
async def a(x: str) -> str: return x
pipeline = Pipeline.from_step(Step.from_callable(a, name="a"))
""",
    )

    # --verbose should show 'Traceback:'
    res_v = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "-v", "run", str(file), "--input", "x"],
        capture_output=True,
        text=True,
    )
    assert res_v.returncode != 0
    assert "Traceback:" in (res_v.stdout + res_v.stderr)

    # --trace should also show 'Traceback:'
    res_t = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "--trace", "run", str(file), "--input", "x"],
        capture_output=True,
        text=True,
    )
    assert res_t.returncode != 0
    assert "Traceback:" in (res_t.stdout + res_t.stderr)


def test_validate_uses_env_project_root_default_pipeline_yaml(tmp_path: Path) -> None:
    # Create minimal project with pipeline.yaml and skills module
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "__init__.py").write_text("")
    (tmp_path / "skills" / "helpers.py").write_text(
        """
async def echo(x: str) -> str:
    return x
"""
    )
    (tmp_path / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - kind: step
    name: s
    agent: "skills.helpers:echo"
"""
    )

    env = os.environ.copy()
    env["FLUJO_PROJECT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", "--format=json"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout or "{}")
    assert payload.get("is_valid") is True


def test_run_uses_env_project_root_default_pipeline_yaml(tmp_path: Path) -> None:
    # Create minimal project with pipeline.yaml and skills module
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "__init__.py").write_text("")
    (tmp_path / "skills" / "helpers.py").write_text(
        """
async def echo(x: str) -> str:
    return x
"""
    )
    (tmp_path / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - kind: step
    name: s
    agent: "skills.helpers:echo"
"""
    )

    env = os.environ.copy()
    env["FLUJO_PROJECT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", "--json", "--input", "hi"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    # Output should be JSON; ensure it parses
    _ = json.loads(result.stdout or "{}")

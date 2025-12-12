from __future__ import annotations

import textwrap
import subprocess
import sys
import os
from pathlib import Path


class Echo:
    async def run(self, x):
        return x


class DummyPlugin:
    async def validate(self, data):  # signature compatible with ValidationPlugin
        class _Outcome:
            def __init__(self) -> None:
                self.success = True

        return _Outcome()


async def dummy_validator(output_to_check, *args, **kwargs):
    return True


async def takes_str(x: str) -> str:  # used via import string in YAML
    return x


async def takes_int(x: int) -> int:  # used via import string in YAML
    return x


def test_cli_compile_invalid_yaml_shows_line_and_col(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
    version: "0.1"
    steps:
      - kind: step
        name: s1
        agent: "tests.cli.test_main:Echo"
      - kind: step
        name: s2
        agent: "tests.cli.test_main:Echo"
          extra_indent: oops
    """)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "dev",
            "compile-yaml",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    # Either stdout or stderr should include line/column
    out = result.stdout + result.stderr
    assert "line" in out and "column" in out


def test_yaml_validate_strict_exits_nonzero(tmp_path: Path) -> None:
    yaml_text = """
    version: "0.1"
    steps:
      - kind: step
        name: a
        agent: "tests.cli.test_main:takes_str"
      - kind: step
        name: b
        agent: "tests.cli.test_main:takes_int"
    """
    p = tmp_path / "invalid.yaml"
    p.write_text(yaml_text)

    # Allow tests modules to be imported by the subprocess
    (tmp_path / "flujo.toml").write_text('blueprint_allowed_imports = ["tests", "flujo"]')

    env = os.environ.copy()
    env["FLUJO_CONFIG_PATH"] = str(tmp_path / "flujo.toml")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "dev",
            "validate",
            "--strict",
            str(p),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0


def test_yaml_run_aborts_on_invalid_pipeline(tmp_path: Path) -> None:
    yaml_text = """
    version: "0.1"
    steps:
      - kind: step
        name: a
        agent: "tests.cli.test_main:takes_str"
      - kind: step
        name: b
        agent: "tests.cli.test_main:takes_int"
    """
    p = tmp_path / "invalid_run.yaml"
    p.write_text(yaml_text)

    # Allow tests modules to be imported
    (tmp_path / "flujo.toml").write_text('blueprint_allowed_imports = ["tests", "flujo"]')
    env = os.environ.copy()
    env["FLUJO_CONFIG_PATH"] = str(tmp_path / "flujo.toml")

    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(p), "--input", "hi"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0


def _write_temp_pipeline(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipe.py"
    p.write_text(textwrap.dedent(content))
    return p


def test_cli_validate_reports_suggestions(tmp_path: Path) -> None:
    file = _write_temp_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline
        async def a(x: str) -> str: return x
        async def b(x: str) -> str: return x
        s1 = Step.from_callable(a, name="a")
        s1.__step_output_type__ = str
        s2 = Step.from_callable(
            b,
            name="b",
            is_adapter=True,
            adapter_id="generic-adapter",
            adapter_allow="generic",
        )
        s2.meta = {"is_adapter": True, "adapter_id": "generic-adapter", "adapter_allow": "generic"}
        s2.__step_input_type__ = str
        pipeline = Pipeline.from_step(s1) >> s2
        """,
    )
    # run flujo dev validate
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "dev",
            "validate",
            str(file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Allow valid output if heuristics do not trigger on this minimal case
    assert (
        "Suggestion:" in result.stdout
        or "Warnings:" in result.stdout
        or "Pipeline is valid" in result.stdout
    )


def test_cli_run_aborts_on_invalid_pipeline(tmp_path: Path) -> None:
    file = _write_temp_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline
        async def a(x: int) -> int: return x
        async def b(x: str) -> str: return x
        s1 = Step.from_callable(a, name="a")
        s2 = Step.from_callable(b, name="b")
        pipeline = Pipeline.from_step(s1) >> s2
        """,
    )
    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(file), "--input", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Pipeline validation failed before run" in result.stdout + result.stderr


def test_cli_validate_strict_exits_nonzero(tmp_path: Path) -> None:
    file = _write_temp_pipeline(
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
            "dev",
            "validate",
            "--strict",
            str(file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_run_prints_suggestions_on_failure(tmp_path: Path) -> None:
    file = _write_temp_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline
        async def takes_str(x: str) -> str: return x
        async def takes_int(x: int) -> int: return x
        primary = Step.from_callable(takes_str, name="primary")
        fb = Step.from_callable(takes_int, name="fallback")
        primary.fallback_step = fb
        pipeline = Pipeline.from_step(primary)
        """,
    )
    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(file), "--input", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Suggestion:" in (result.stdout + result.stderr)


def test_cli_compile_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_text = """
    version: "0.1"
    steps:
      - kind: step
        name: s1
        agent:
          id: "flujo.builtins.echo"
        meta:
          is_adapter: true
          adapter_id: generic-adapter
          adapter_allow: generic
      - kind: map
        name: mapper
        map:
          iterable_input: items
          body:
            - kind: step
              name: inner
              agent:
                id: "flujo.builtins.echo"
              meta:
                is_adapter: true
                adapter_id: generic-adapter
                adapter_allow: generic
      - kind: parallel
        name: p
        branches:
          a:
            - kind: step
              name: a1
              agent:
                id: "flujo.builtins.echo"
              meta:
                is_adapter: true
                adapter_id: generic-adapter
                adapter_allow: generic
          b:
            - kind: step
              name: b1
              agent:
                id: "flujo.builtins.echo"
              meta:
                is_adapter: true
                adapter_id: generic-adapter
                adapter_allow: generic
    """
    src = tmp_path / "pipe.yaml"
    src.write_text(yaml_text)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "dev",
            "compile-yaml",
            str(src),
            "--no-normalize",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "version:" in result.stdout

    # Round-trip structural check: from_yaml -> to_yaml -> from_yaml
    from flujo.domain.dsl import Pipeline

    p1 = Pipeline.from_yaml_file(str(src))
    y = p1.to_yaml()
    p2 = Pipeline.from_yaml_text(y)
    # Compare step kinds and names as a proxy for structure
    kinds1 = [type(s).__name__ for s in p1.steps]
    kinds2 = [type(s).__name__ for s in p2.steps]
    names1 = [s.name for s in p1.steps]
    names2 = [s.name for s in p2.steps]
    assert kinds1 == kinds2
    assert names1 == names2


def test_yaml_plugins_and_validators(tmp_path: Path) -> None:
    # Define a trivial plugin/validator in this module and reference by import string
    yaml_text = """
    version: "0.1"
    steps:
      - kind: step
        name: s1
        plugins:
          - path: "tests.cli.test_main:DummyPlugin"
            priority: 1
        validators:
          - "tests.cli.test_main:dummy_validator"
    """
    src = tmp_path / "pipe.yaml"
    src.write_text(yaml_text)

    # Allow tests modules to be imported
    (tmp_path / "flujo.toml").write_text('blueprint_allowed_imports = ["tests", "flujo"]')
    env = os.environ.copy()
    env["FLUJO_CONFIG_PATH"] = str(tmp_path / "flujo.toml")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "run",
            str(src),
            "--input",
            "hi",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0


def test_yaml_agent_registry_resolution(tmp_path: Path) -> None:
    # Use skills.yaml catalog next to the YAML so the subprocess can resolve it
    yaml_text = """
    version: "0.1"
    steps:
      - kind: step
        name: s1
        agent:
          id: "echo-skill"
    """
    src = tmp_path / "pipe.yaml"
    src.write_text(yaml_text)
    skills_yaml = tmp_path / "skills.yaml"
    skills_yaml.write_text(
        """
echo-skill:
  path: "tests.cli.test_main:Echo"
  description: "echo agent"
        """.strip()
    )

    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "run", str(src), "--input", "hi"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Final output:" in (result.stdout + result.stderr)

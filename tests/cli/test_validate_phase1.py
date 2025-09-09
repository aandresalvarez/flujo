from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_validate_v_sm1_state_machine_warnings_json(tmp_path: Path) -> None:
    # StateMachine with no path from start_state to end state -> V-SM1 warning
    yaml_text = """
version: "0.1"
steps:
  - kind: StateMachine
    name: sm
    start_state: s1
    end_states: [done]
    states:
      s1:
        - kind: step
          name: a
      s2:
        - kind: step
          name: b
    transitions:
      - from: s1
        on: success
        to: s2
"""
    y = _write_file(tmp_path, "pipe.yaml", yaml_text)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(y),
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0  # warnings only should not fail strict by default
    payload = json.loads(res.stdout or "{}")
    warnings = payload.get("warnings") or []
    assert any(w.get("rule_id") == "V-SM1" for w in warnings)


def test_validate_v_s1_schema_warnings_json(tmp_path: Path) -> None:
    # Agent output_schema with array type but missing 'items' -> V-S1 warning
    yaml_text = """
version: "0.1"
agents:
  bad:
    model: "openai:gpt-4o"
    system_prompt: "x"
    output_schema:
      type: array
steps:
  - kind: step
    name: s1
    uses: agents.bad
"""
    y = _write_file(tmp_path, "pipe.yaml", yaml_text)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(y),
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    payload = json.loads(res.stdout or "{}")
    warnings = payload.get("warnings") or []
    assert any(w.get("rule_id") == "V-S1" for w in warnings)


def test_validate_explain_flag_in_json(tmp_path: Path) -> None:
    # Create a pipeline with a type mismatch to trigger V-A2 error and check explain is present
    py = _write_file(
        tmp_path,
        "pipe.py",
        """
from flujo.domain.dsl import Step, Pipeline
async def a(x: int) -> int: return x
async def b(x: str) -> str: return x
pipeline = Pipeline.from_step(Step.from_callable(a, name="a")) >> Step.from_callable(b, name="b")
""",
    )
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(py),
            "--format=json",
            "--explain",
        ],
        capture_output=True,
        text=True,
    )
    # Strict by default; V-A2 should fail with EX_VALIDATION_FAILED (4)
    assert res.returncode == 4
    payload = json.loads(res.stdout or "{}")
    errors = payload.get("errors") or []
    assert any(e.get("rule_id") == "V-A2" for e in errors)
    # explain should be present for findings when --explain is used
    assert any("explain" in e and e["explain"] for e in errors)

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from flujo.cli.main import _validate_impl


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(content)
    return p


def test_baseline_hides_existing_warnings(tmp_path: Path, capsys) -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: First
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
            updates_context: true
          - name: Second
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"
        """
    )
    ypath = _write_yaml(tmp_path, yaml_text)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "errors": [],
                "warnings": [{"rule_id": "V-T1", "step_name": "Second"}],
            }
        )
    )

    _validate_impl(
        str(ypath),
        strict=False,
        output_format="json",
        include_imports=False,
        fail_on_warn=False,
        rules=None,
        explain=False,
        baseline=str(baseline),
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    # Post-baseline view should have no warnings, but a baseline summary is present
    assert payload["warnings"] == []
    assert payload.get("baseline", {}).get("applied") is True


def test_update_baseline_writes_current_report(tmp_path: Path, capsys) -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: A
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
            updates_context: true
          - name: B
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"
        """
    )
    ypath = _write_yaml(tmp_path, yaml_text)
    baseline = tmp_path / "bl.json"

    # Run with --update-baseline (no prior baseline)
    _validate_impl(
        str(ypath),
        strict=False,
        output_format="json",
        include_imports=False,
        fail_on_warn=False,
        rules=None,
        explain=False,
        baseline=str(baseline),
        update_baseline=True,
    )
    # Baseline file should exist and be valid JSON with keys
    data = json.loads(baseline.read_text())
    assert set(data.keys()) == {"errors", "warnings"}

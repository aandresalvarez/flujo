from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from flujo.cli.main import _validate_impl


def test_rules_file_glob_suppresses_template_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    ypath = tmp_path / "p.yaml"
    ypath.write_text(yaml_text)

    # Rules JSON turning off all template rules via glob V-T*
    rules = {"V-T*": "off"}
    rpath = tmp_path / "rules.json"
    rpath.write_text(json.dumps(rules))

    # Run validate_impl to produce JSON; ensure no warnings present
    _validate_impl(
        str(ypath),
        strict=False,
        output_format="json",
        include_imports=False,
        fail_on_warn=False,
        rules=str(rpath),
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["warnings"] == [], payload

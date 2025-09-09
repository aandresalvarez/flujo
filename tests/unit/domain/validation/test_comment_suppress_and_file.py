from __future__ import annotations

import textwrap
from pathlib import Path
import json

from flujo.cli.helpers import validate_pipeline_file


def test_comment_based_suppression_and_file_field(tmp_path: Path) -> None:
    """Inline comment '# flujo: ignore' should suppress matching rules; warnings should include file when present."""
    yml = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: First
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
            updates_context: true
          - name: Second  # flujo: ignore V-T1
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"
          - name: Third
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step | foo }}"  # unknown filter → V-T3
        """
    )
    path = tmp_path / "p.yaml"
    path.write_text(yml)
    report = validate_pipeline_file(str(path), include_imports=False)

    # Ensure V-T1 is suppressed on 'Second'
    assert all(not (w.step_name == "Second" and w.rule_id == "V-T1") for w in report.warnings)

    # Ensure at least one remaining warning has file field set to our path
    assert any(getattr(w, "file", None) == str(path) for w in report.warnings), [
        json.dumps(w.model_dump()) for w in report.warnings
    ]

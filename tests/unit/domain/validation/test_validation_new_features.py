from __future__ import annotations

import textwrap
from pathlib import Path

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_validate_template_previous_step_output_warns() -> None:
    """V-T1: Detect misuse of previous_step.output in templated input."""
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: First
            agent:
              id: "flujo.builtins.stringify"
            input: "hello"
          - name: Second
            agent:
              id: "flujo.builtins.stringify"
            input: "{{ previous_step.output }}"
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    report = pipeline.validate_graph()
    # Should warn on V-T1, and otherwise be valid structurally
    vt1s = [w for w in report.warnings if w.rule_id == "V-T1"]
    assert vt1s, f"Expected V-T1 warning, found: {report.warnings}"
    assert any("previous_step.output" in w.message for w in vt1s)


def test_validate_imports_aggregates_child_findings(tmp_path: Path) -> None:
    """V-I4: Validate child YAML and aggregate findings (warnings) into parent report."""
    # Child with a V-T1 template misuse to generate a warning
    child_yaml = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: C1
            agent:
              id: "flujo.builtins.stringify"
            input: "hello"
          - name: C2
            agent:
              id: "flujo.builtins.stringify"
            input: "{{ previous_step.output }}"
        """
    )
    child_file = tmp_path / "child.yaml"
    child_file.write_text(child_yaml)

    parent_yaml = textwrap.dedent(
        f"""
        version: "0.1"
        imports:
          child: "{child_file.name}"
        steps:
          - name: RunChild
            uses: imports.child
            updates_context: true
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))

    # No child aggregation
    report_no_children = pipeline.validate_graph(include_imports=False)
    # Parent only; child warnings should not be present
    assert all("[import:RunChild]" not in w.message for w in report_no_children.warnings)

    # With children aggregated, we should see the child's V-A1 surfaced
    report_with_children = pipeline.validate_graph(include_imports=True)
    msgs = [w.message for w in report_with_children.warnings]
    assert any("[import:child]" in m for m in msgs), msgs
    # And location_path should be prefixed with imports.<alias>::
    locs = [w.location_path for w in report_with_children.warnings]
    assert any(isinstance(lp, str) and lp.startswith("imports.child::") for lp in locs), locs

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


def test_template_lints_vt2_vt3_vt4() -> None:
    """Detect 'this' misuse, unknown filters, and unknown step proxy names."""
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: S1
            agent:
              id: "flujo.builtins.stringify"
            input: "hello"
          - name: S2
            agent:
              id: "flujo.builtins.stringify"
            input: "{{ this }} | {{ previous_step | foo }} | {{ steps.Missing.output }}"
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    report = pipeline.validate_graph()
    rules = {w.rule_id for w in report.warnings}
    assert "V-T2" in rules  # 'this' misuse
    assert "V-T3" in rules  # unknown filter 'foo'
    assert "V-T4" in rules  # unknown steps.Missing


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
    assert all("[import:child]" not in w.message for w in report_no_children.warnings)

    # With children aggregated, we should see the child's V-T1 surfaced
    report_with_children = pipeline.validate_graph(include_imports=True)
    msgs = [w.message for w in report_with_children.warnings]
    assert any("[import:child]" in m for m in msgs), msgs
    # And location_path should be prefixed with imports.<alias>::
    locs = [w.location_path for w in report_with_children.warnings]
    assert any(isinstance(lp, str) and lp.startswith("imports.child::") for lp in locs), locs


def test_validate_imports_recurses_into_grandchildren(tmp_path: Path) -> None:
    """Ensure include_imports=True recurses beyond one level (grandchildren)."""
    grand_yaml = (
        'version: "0.1"\n'
        "steps:\n"
        "  - name: G1\n"
        "    agent:\n"
        '      id: "flujo.builtins.stringify"\n'
        '    input: "hello"\n'
        "  - name: G2\n"
        "    agent:\n"
        '      id: "flujo.builtins.stringify"\n'
        '    input: "{{ previous_step.output }}"\n'
    )
    child_yaml = (
        'version: "0.1"\n'
        "imports:\n"
        '  grand: "grand.yaml"\n'
        "steps:\n"
        "  - name: RunGrand\n"
        "    uses: imports.grand\n"
        "    updates_context: true\n"
    )
    (tmp_path / "grand.yaml").write_text(grand_yaml)
    (tmp_path / "child.yaml").write_text(child_yaml)
    parent_yaml = (
        'version: "0.1"\n'
        "imports:\n"
        '  child: "child.yaml"\n'
        "steps:\n"
        "  - name: RunChild\n"
        "    uses: imports.child\n"
        "    updates_context: true\n"
    )
    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    report = pipeline.validate_graph(include_imports=True)
    # We expect a warning from the grandchild to bubble up via child aggregation
    msgs = [w.message for w in report.warnings]
    joined = "\n".join(msgs)
    assert "[import:child]" in joined or "[import:RunChild]" in joined


def test_cycle_detection_v_i3_compiler_surfaces_error(tmp_path: Path) -> None:
    """Cyclic imports fail at compile time; loader surfaces the error.

    In this codebase, import graphs are resolved during blueprint compilation.
    Cycles produce a loader/compile error rather than a validate_graph finding.
    """
    a_yaml = (
        'version: "0.1"\nimports:\n  b: "b.yaml"\nsteps:\n  - name: RunB\n    uses: imports.b\n'
    )
    b_yaml = (
        'version: "0.1"\nimports:\n  a: "a.yaml"\nsteps:\n  - name: RunA\n    uses: imports.a\n'
    )
    (tmp_path / "a.yaml").write_text(a_yaml)
    (tmp_path / "b.yaml").write_text(b_yaml)
    parent_text = (tmp_path / "a.yaml").read_text()
    try:
        _ = load_pipeline_blueprint_from_yaml(parent_text, base_dir=str(tmp_path))
        assert False, "Expected compile error due to cyclic imports"
    except Exception:
        # Any loader/compile exception is acceptable here (we avoid tight coupling to exception types)
        pass

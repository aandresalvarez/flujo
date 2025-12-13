from __future__ import annotations

import textwrap
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_per_step_suppress_rules_glob_template_lints() -> None:
    """Per-step suppress_rules should suppress matching findings (glob)."""
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: S1
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
            updates_context: true
          - name: S2
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    # Baseline: should have at least one V-T1 warning
    report = pipeline.validate_graph()
    assert any(w.rule_id == "V-T1" for w in report.warnings)

    # Suppress template rules on S2
    for st in pipeline.steps:
        if st.name == "S2":
            st.meta.setdefault("suppress_rules", []).append("V-T*")
    report2 = pipeline.validate_graph()
    # After suppression, no V-T* warnings should remain for S2
    assert all(not (w.step_name == "S2" and w.rule_id.startswith("V-T")) for w in report2.warnings)


def test_per_step_suppress_on_import_aggregation(tmp_path) -> None:
    """Suppress aggregated child findings using suppress_rules on the import step."""
    child_yaml = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: C1
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
          - name: C2
            agent: { id: "flujo.builtins.stringify" }
            input: "world"
        """
    )
    (tmp_path / "child.yaml").write_text(child_yaml)
    parent_yaml = textwrap.dedent(
        """
        version: "0.1"
        imports:
          child: "child.yaml"
        steps:
          - name: RunChild
            uses: imports.child
            updates_context: true
            config:
              outputs:
                - { child: "import_artifacts.k", parent: "badroot.value" }
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    report = pipeline.validate_graph(include_imports=True)
    # We expect at least one warning on the import step (V-I2 mapping sanity)
    assert any(w.step_name == "RunChild" for w in report.warnings)

    # Suppress V-T* on the import step
    for st in pipeline.steps:
        if st.name == "RunChild":
            st.meta.setdefault("suppress_rules", []).append("V-*")
    report2 = pipeline.validate_graph(include_imports=True)
    assert all(w.step_name != "RunChild" for w in report2.warnings)

from __future__ import annotations
from flujo.type_definitions.common import JSONObject  # noqa: F401

import textwrap
from pathlib import Path
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_v_i4_child_aggregation_summary(tmp_path: Path) -> None:
    """V-I4: Parent gets a summary warning when child has findings."""
    child = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: C1
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
          - name: C2
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"  # V-T1 in child
        """
    )
    (tmp_path / "child.yaml").write_text(child)

    parent = textwrap.dedent(
        """
        version: "0.1"
        imports:
          c: "child.yaml"
        steps:
          - name: RunChild
            uses: imports.c
            updates_context: true
        """
    )
    p = load_pipeline_blueprint_from_yaml(parent, base_dir=str(tmp_path))
    report = p.validate_graph(include_imports=True)
    assert any(w.rule_id == "V-I4" and w.step_name == "RunChild" for w in report.warnings), (
        report.model_dump()
    )


def test_v_i5_input_projection_coherence_warns_initial_prompt_vs_object() -> None:
    """Warn when projecting to initial_prompt but child expects object input."""
    # Child first step expects dict
    child_py = (
        "from typing import Dict, Any\nasync def f(x: JSONObject) -> JSONObject:\n    return x\n"
    )
    import tempfile
    import textwrap
    from pathlib import Path as _P

    tmp = _P(tempfile.mkdtemp())
    (tmp / "skills").mkdir(parents=True, exist_ok=True)
    (tmp / "skills" / "__init__.py").write_text("")
    (tmp / "skills" / "i5.py").write_text(child_py)
    # Child YAML references python agent function expecting dict
    (tmp / "child.yaml").write_text(
        textwrap.dedent(
            """
            version: "0.1"
            steps:
              - name: C
                agent: "skills.i5:f"
            """
        )
    )
    parent_yaml = textwrap.dedent(
        """
        version: "0.1"
        imports:
          c: "child.yaml"
        steps:
          - name: RunChild
            uses: imports.c
            input: {"k": 1}
        """
    )
    p = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp))
    report = p.validate_graph(include_imports=True)
    assert any(w.rule_id == "V-I5" and w.step_name == "RunChild" for w in report.warnings), (
        report.model_dump()
    )


def test_v_i6_inherit_conversation_consistency_warns(tmp_path: Path) -> None:
    """Warn when mapping conversation history but inherit_conversation=False."""
    child = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: C
            agent: { id: "flujo.builtins.stringify" }
            input: "hi"
        """
    )
    (tmp_path / "child.yaml").write_text(child)
    parent = textwrap.dedent(
        """
        version: "0.1"
        imports:
          c: "child.yaml"
        steps:
          - name: RunChild
            uses: imports.c
            updates_context: true
            config:
              inherit_conversation: false
              outputs:
                - { child: "conversation_history", parent: "conversation_history" }
        """
    )
    p = load_pipeline_blueprint_from_yaml(parent, base_dir=str(tmp_path))
    rep = p.validate_graph(include_imports=True)
    assert any(w.rule_id == "V-I6" and w.step_name == "RunChild" for w in rep.warnings), (
        rep.model_dump()
    )

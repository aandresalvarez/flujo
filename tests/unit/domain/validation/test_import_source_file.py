from __future__ import annotations

import textwrap
from pathlib import Path

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_imported_child_steps_have_file_in_yaml_loc(tmp_path: Path) -> None:
    child_yaml = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: C1
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
        """
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(child_yaml)

    parent_yaml = textwrap.dedent(
        f"""
        version: "0.1"
        imports:
          child: "{child_path.name}"
        steps:
          - name: RunChild
            uses: imports.child
            updates_context: true
        """
    )
    parent = load_pipeline_blueprint_from_yaml(
        parent_yaml, base_dir=str(tmp_path), source_file=str(tmp_path / "parent.yaml")
    )
    imp = parent.steps[0]
    # Access child pipeline from import step
    child_pipe = getattr(imp, "pipeline", None)
    assert child_pipe is not None
    c1 = child_pipe.steps[0]
    yloc = getattr(c1, "meta", {}).get("_yaml_loc")
    assert isinstance(yloc, dict)
    assert yloc.get("file") == str(child_path)

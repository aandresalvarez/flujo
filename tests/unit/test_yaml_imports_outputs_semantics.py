from __future__ import annotations

from pathlib import Path


def _write_child(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - name: Child
        """.strip()
    )
    return d


def test_yaml_imports_outputs_dict_is_converted_to_list(tmp_path: Path) -> None:
    _write_child(tmp_path, "child_a")
    parent = tmp_path / "parent_a"
    parent.mkdir()
    (parent / "pipeline.yaml").write_text(
        """
version: "0.1"
imports:
  c: "../child_a/pipeline.yaml"
steps:
  - name: UseC
    uses: imports.c
    updates_context: true
    config:
      outputs: { "import_artifacts.foo": "import_artifacts.bar" }
""".strip()
    )

    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.domain.dsl.import_step import ImportStep, OutputMapping

    text = (parent / "pipeline.yaml").read_text()
    pipe = load_pipeline_blueprint_from_yaml(text, base_dir=str(parent))
    step0 = pipe.steps[0]
    assert isinstance(step0, ImportStep)
    assert step0.outputs is not None
    assert len(step0.outputs) == 1
    assert isinstance(step0.outputs[0], OutputMapping)
    assert step0.outputs[0].child == "import_artifacts.foo"
    assert step0.outputs[0].parent == "import_artifacts.bar"


def test_yaml_imports_outputs_unset_is_none(tmp_path: Path) -> None:
    _write_child(tmp_path, "child_b")
    parent = tmp_path / "parent_b"
    parent.mkdir()
    (parent / "pipeline.yaml").write_text(
        """
version: "0.1"
imports:
  c: "../child_b/pipeline.yaml"
steps:
  - name: UseC
    uses: imports.c
    updates_context: true
""".strip()
    )

    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.domain.dsl.import_step import ImportStep

    text = (parent / "pipeline.yaml").read_text()
    pipe = load_pipeline_blueprint_from_yaml(text, base_dir=str(parent))
    step0 = pipe.steps[0]
    assert isinstance(step0, ImportStep)
    assert step0.outputs is None


def test_yaml_imports_outputs_empty_list_is_empty(tmp_path: Path) -> None:
    _write_child(tmp_path, "child_c")
    parent = tmp_path / "parent_c"
    parent.mkdir()
    (parent / "pipeline.yaml").write_text(
        """
version: "0.1"
imports:
  c: "../child_c/pipeline.yaml"
steps:
  - name: UseC
    uses: imports.c
    updates_context: true
    config:
      outputs: []
""".strip()
    )

    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.domain.dsl.import_step import ImportStep

    text = (parent / "pipeline.yaml").read_text()
    pipe = load_pipeline_blueprint_from_yaml(text, base_dir=str(parent))
    step0 = pipe.steps[0]
    assert isinstance(step0, ImportStep)
    assert step0.outputs == []

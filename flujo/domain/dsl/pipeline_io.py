from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover - for type checking without runtime cycles
    from .pipeline import Pipeline


PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")


def load_from_yaml(yaml_source: str, *, is_path: bool = True) -> "Pipeline[object, object]":
    """Load a Pipeline from YAML (file path or raw text)."""
    from ..blueprint import load_pipeline_blueprint_from_yaml

    if is_path:
        with open(yaml_source, "r") as f:
            yaml_text = f.read()
    else:
        yaml_text = yaml_source
    return load_pipeline_blueprint_from_yaml(yaml_text)


def load_from_yaml_text(yaml_text: str) -> "Pipeline[object, object]":
    """Load a Pipeline from YAML text."""
    from ..blueprint import load_pipeline_blueprint_from_yaml

    return load_pipeline_blueprint_from_yaml(yaml_text)


def load_from_yaml_file(path: str) -> "Pipeline[object, object]":
    """Load a Pipeline from a YAML file path."""
    return load_from_yaml(path, is_path=True)


def dump_to_yaml(pipeline: "Pipeline[PipeInT, PipeOutT]") -> str:
    """Serialize a Pipeline to YAML string."""
    from ..blueprint import dump_pipeline_blueprint_to_yaml

    return dump_pipeline_blueprint_to_yaml(pipeline)


def dump_to_yaml_file(pipeline: "Pipeline[PipeInT, PipeOutT]", path: str) -> None:
    """Serialize a Pipeline to YAML file."""
    text = dump_to_yaml(pipeline)
    with open(path, "w") as f:
        f.write(text)

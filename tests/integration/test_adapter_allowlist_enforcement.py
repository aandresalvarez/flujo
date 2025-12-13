from __future__ import annotations

import textwrap

import pytest

from flujo.domain.dsl import Pipeline
from flujo.domain.blueprint.loader_models import BlueprintError


def test_adapter_missing_token_in_yaml_is_rejected() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: flujo.builtins.stringify
      - name: adapt
        agent: flujo.builtins.stringify
        meta:
          is_adapter: true
          adapter_id: generic-adapter
    """
    with pytest.raises(BlueprintError, match="adapter_id and adapter_allow"):
        Pipeline.from_yaml_text(textwrap.dedent(yaml_text))


def test_adapter_with_allow_token_validates() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: flujo.builtins.stringify
      - name: adapt
        agent: flujo.builtins.stringify
        meta:
          is_adapter: true
          adapter_id: generic-adapter
          adapter_allow: generic
    """
    pipeline = Pipeline.from_yaml_text(textwrap.dedent(yaml_text))
    pipeline.steps[0].__step_output_type__ = str
    pipeline.steps[1].__step_input_type__ = str

    report = pipeline.validate_graph()
    assert not any(f.rule_id == "V-ADAPT-ALLOW" for f in report.errors)

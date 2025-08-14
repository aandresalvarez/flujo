from __future__ import annotations

import pytest

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml, BlueprintError


def test_invalid_uses_format_rejected() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: step
    name: bad
    uses: agents.1not_valid
"""
    with pytest.raises(BlueprintError) as ei:
        load_pipeline_blueprint_from_yaml(yaml_text)
    msg = str(ei.value)
    assert "uses" in msg
    assert "valid identifier" in msg or "valid Python import path" in msg


def test_unknown_agent_reference_rejected() -> None:
    yaml_text = """
version: "0.1"
agents:
  known:
    model: "openai:gpt-4o-mini"
    system_prompt: "test"
    output_schema:
      type: object
      properties:
        x:
          type: string
steps:
  - kind: step
    name: s1
    uses: agents.unknown
"""
    with pytest.raises(BlueprintError) as ei:
        load_pipeline_blueprint_from_yaml(yaml_text)
    assert "Unknown declarative agent referenced" in str(ei.value)


def test_agent_model_requires_dict_output_schema() -> None:
    # output_schema must be a mapping/dict; strings should be rejected
    yaml_text = """
version: "0.1"
agents:
  a:
    model: "openai:gpt-4o-mini"
    system_prompt: "test"
    output_schema: "string"
steps:
  - kind: step
    name: s1
"""
    with pytest.raises(BlueprintError) as ei:
        load_pipeline_blueprint_from_yaml(yaml_text)
    assert "output_schema" in str(ei.value)

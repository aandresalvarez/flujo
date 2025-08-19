import yaml
from pydantic import BaseModel

from flujo.domain.blueprint.loader import (
    dump_pipeline_blueprint_to_yaml,
    load_pipeline_blueprint_from_yaml,
)
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import HumanInTheLoopStep


class _Minimal(BaseModel):
    value: str


def test_dump_hitl_without_schema_omits_input_schema():
    hitl = HumanInTheLoopStep(name="Ask", message_for_user="Say hi", input_schema=None)
    y = dump_pipeline_blueprint_to_yaml(Pipeline(steps=[hitl]))
    data = yaml.safe_load(y)
    node = data["steps"][0]
    assert node["kind"] == "hitl"
    assert node.get("input_schema") is None
    assert node.get("message") == "Say hi"


def test_roundtrip_hitl_dump_then_load_preserves_kind_and_message():
    hitl = HumanInTheLoopStep(name="Confirm", message_for_user="Approve?", input_schema=_Minimal)
    p1 = Pipeline(steps=[hitl])
    dumped = dump_pipeline_blueprint_to_yaml(p1)
    p2 = load_pipeline_blueprint_from_yaml(dumped)
    assert isinstance(p2, Pipeline)
    assert len(p2.steps) == 1
    st = p2.steps[0]
    assert isinstance(st, HumanInTheLoopStep)
    assert getattr(st, "message_for_user") == "Approve?"
    assert getattr(st, "input_schema") is not None


def test_dump_after_load_contains_hitl_block():
    y1 = """
version: "0.1"
name: "hitl-rt"
steps:
  - kind: hitl
    name: Ask
    message: "Go on"
"""
    p = load_pipeline_blueprint_from_yaml(y1)
    y2 = dump_pipeline_blueprint_to_yaml(p)
    # Ensure 'kind: hitl' survives and message is present
    data = yaml.safe_load(y2)
    node = data["steps"][0]
    assert node["kind"] == "hitl"
    assert node.get("message") == "Go on"

import yaml
from typing import Optional
from pydantic import BaseModel

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.blueprint.loader import (
    dump_pipeline_blueprint_to_yaml,
    load_pipeline_blueprint_from_yaml,
)


class _ApprovalInput(BaseModel):
    confirmation: str
    reasoning: Optional[str] = None


def test_dump_pipeline_with_hitl_outputs_kind_and_schema():
    step = HumanInTheLoopStep(
        name="ConfirmPlan",
        message_for_user="Review the plan. Approve to proceed.",
        input_schema=_ApprovalInput,
    )
    p = Pipeline(steps=[step])

    yaml_text = dump_pipeline_blueprint_to_yaml(p)
    data = yaml.safe_load(yaml_text)

    assert data["version"] == "0.1"
    assert isinstance(data["steps"], list) and len(data["steps"]) == 1
    node = data["steps"][0]
    assert node["kind"] == "hitl"
    assert node["name"] == "ConfirmPlan"
    assert node["message"].startswith("Review the plan")
    schema = node.get("input_schema")
    assert isinstance(schema, dict)
    assert "properties" in schema and "confirmation" in schema["properties"]


def test_load_pipeline_yaml_with_hitl_compiles_to_human_step():
    yaml_text = """
version: "0.1"
name: "hitl-demo"
steps:
  - kind: hitl
    name: GetUserApproval
    message: "Approve the plan? (yes/no)"
    input_schema:
      type: object
      properties:
        confirmation:
          type: string
          enum: ["yes", "no"]
        reasoning:
          type: string
      required: [confirmation]
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
    st = pipeline.steps[0]
    assert isinstance(st, HumanInTheLoopStep)
    assert getattr(st, "message_for_user", None) == "Approve the plan? (yes/no)"
    # Should have a pydantic model class as input_schema with model_json_schema available
    schema = getattr(st, "input_schema", None)
    assert schema is not None and hasattr(schema, "model_json_schema")
    js = schema.model_json_schema()
    assert "properties" in js and "confirmation" in js["properties"]

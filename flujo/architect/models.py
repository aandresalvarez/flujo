from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from flujo.type_definitions.common import JSONObject


class PlannedStep(BaseModel):
    step_name: str = Field(..., description="Short name for the step, e.g., 'FetchWebpage'.")
    purpose: str = Field(..., description="One-sentence explanation of what this step achieves.")


class ExecutionPlan(BaseModel):
    plan_summary: str = Field(..., description="High-level summary of the multi-step plan.")
    steps: List[PlannedStep] = Field(
        ..., description="Ordered list of steps to accomplish the goal."
    )


class ToolSelection(BaseModel):
    step_name: str
    chosen_agent_id: str = Field(
        ..., description="The ID of the best skill, e.g., 'flujo.builtins.web_search'."
    )
    agent_params: JSONObject = Field(
        default_factory=dict,
        description="Parameters to pass to the chosen agent.",
    )


class GeneratedYaml(BaseModel):
    generated_yaml: str = Field(
        ..., description="The complete, valid Flujo pipeline.yaml as a single string."
    )

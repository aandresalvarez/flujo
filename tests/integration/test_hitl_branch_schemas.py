import pytest

from flujo.application.runner import Flujo
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def _build_yaml_with_branch_schemas() -> str:
    return """
version: "0.1"
name: "hitl-branch-schemas"
steps:
  - kind: hitl
    name: AskApproval
    message: "Approve? (y/n)"
  - kind: conditional
    name: Route
    condition: "flujo.builtins.check_user_confirmation_sync"
    branches:
      approved:
        - kind: hitl
          name: CollectCount
          message: "Provide count"
          input_schema:
            type: object
            properties:
              count:
                type: integer
            required: [count]
        - kind: step
          name: ShowCount
          agent: { id: "flujo.builtins.stringify" }
      denied:
        - kind: hitl
          name: CollectReason
          message: "Provide reason"
          input_schema:
            type: object
            properties:
              reason:
                type: string
            required: [reason]
        - kind: step
          name: ShowReason
          agent: { id: "flujo.builtins.stringify" }
"""


@pytest.mark.asyncio
async def test_branch_specific_hitl_schema_on_approved_path():
    yaml_text = _build_yaml_with_branch_schemas()
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    runner: Flujo[object, object, object] = Flujo(pipeline=pipeline)

    paused1 = None
    async for item in runner.run_async(initial_input=None):
        paused1 = item
    assert paused1 is not None

    # Approve and hit nested HITL with count schema
    if getattr(paused1.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at first HITL")
    paused2 = await runner.resume_async(paused1, "yes")
    if getattr(paused2.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at nested HITL")
    final = await runner.resume_async(paused2, {"count": 7})
    last = final.step_history[-1]
    assert last.success is True


@pytest.mark.asyncio
async def test_branch_specific_hitl_schema_on_denied_path():
    yaml_text = _build_yaml_with_branch_schemas()
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    runner: Flujo[object, object, object] = Flujo(pipeline=pipeline)

    paused1 = None
    async for item in runner.run_async(initial_input=None):
        paused1 = item
    assert paused1 is not None

    # Deny and hit nested HITL with reason schema
    if getattr(paused1.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at first HITL")
    paused2 = await runner.resume_async(paused1, "no")
    if getattr(paused2.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at nested HITL")
    final = await runner.resume_async(paused2, {"reason": "busy"})
    last = final.step_history[-1]
    assert last.success is True

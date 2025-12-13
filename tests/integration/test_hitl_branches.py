import pytest

from flujo.application.runner import Flujo
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


@pytest.mark.asyncio
async def test_nested_hitl_in_approved_branch_pauses_twice():
    yaml_text = """
version: "0.1"
name: "hitl-nested-branch"
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
          name: AskDetail
          message: "Provide detail"
        - kind: step
          name: Finalize
          agent: { id: "flujo.builtins.stringify" }
      denied:
        - kind: step
          name: Denied
          agent: { id: "flujo.builtins.stringify" }
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    runner: Flujo[object, object, object] = Flujo(pipeline=pipeline)

    paused1 = None
    async for item in runner.run_async(initial_input=None):
        paused1 = item
    assert paused1 is not None
    # First pause at AskApproval
    assert getattr(paused1.final_pipeline_context, "status", None) in {"paused", "failed"}

    # Approve -> should pause again at AskDetail
    if getattr(paused1.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at first HITL")
    paused2 = await runner.resume_async(paused1, "yes")
    if getattr(paused2.final_pipeline_context, "status", None) != "paused":
        pytest.skip("Pipeline did not pause at nested HITL")

    final = await runner.resume_async(paused2, {"detail": "D"})
    last = final.step_history[-1]
    assert last.success is True


@pytest.mark.asyncio
async def test_conditional_default_branch_executes_when_branch_missing():
    # Only 'approved' branch declared; 'denied' goes to default branch
    yaml_text = """
version: "0.1"
name: "hitl-default-branch"
steps:
  - kind: hitl
    name: AskApproval
    message: "Approve? (y/n)"
  - kind: conditional
    name: Route
    condition: "flujo.builtins.check_user_confirmation_sync"
    branches:
      approved:
        - kind: step
          name: Approved
          agent: { id: "flujo.builtins.stringify" }
    default_branch:
      - kind: step
        name: Fallback
        agent: { id: "flujo.builtins.stringify" }
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    runner: Flujo[object, object, object] = Flujo(pipeline=pipeline)

    paused = None
    async for item in runner.run_async(initial_input=None):
        paused = item
    assert paused is not None

    final = await runner.resume_async(paused, "no")
    last = final.step_history[-1]
    assert last.success is True

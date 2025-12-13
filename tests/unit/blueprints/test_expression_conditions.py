from __future__ import annotations

from typing import Any

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def _make_ctx(steps: dict[str, Any] | None = None) -> Any:
    class Ctx:
        def __init__(self) -> None:
            self.step_outputs = steps or {}

    return Ctx()


def test_conditional_condition_expression_returns_branch_key() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: conditional
    name: route
    condition_expression: "previous_step.kind"
    branches:
      text:
        - kind: step
          name: A
      code:
        - kind: step
          name: B
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    step = pipeline.steps[0]
    # Simulate previous step output
    prev = {"kind": "code"}
    # Call condition callable directly
    branch = step.condition_callable(prev, _make_ctx())
    assert branch == "code"


def test_loop_exit_expression_truthiness() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: l
    loop:
      body:
        - kind: step
          name: s
      max_loops: 3
      exit_expression: "previous_step.finished == True"
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    step = pipeline.steps[0]
    # Before finished
    out1 = {"finished": False}
    assert step.exit_condition_callable(out1, _make_ctx()) is False
    # After finished
    out2 = {"finished": True}
    assert step.exit_condition_callable(out2, _make_ctx()) is True

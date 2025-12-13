from __future__ import annotations

from typing import Any

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.dsl.conditional import ConditionalStep


def _make_ctx(steps: dict[str, Any] | None = None) -> Any:
    class Ctx:
        def __init__(self) -> None:
            self.step_outputs = steps or {}

    return Ctx()


def test_conditional_yaml_accepts_bool_branch_keys_and_coerces_to_strings() -> None:
    """YAML with unquoted true/false keys should load and be coerced to strings.

    Implements FSD-026: allow boolean results/keys for ConditionalStep branches.
    """
    yaml_text = """
version: "0.1"
steps:
  - kind: conditional
    name: check_bool
    condition_expression: "previous_step"  # identity: passes input as condition value
    branches:
      true:
        - kind: step
          name: true_branch_step
      false:
        - kind: step
          name: false_branch_step
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    step = pipeline.steps[0]
    assert isinstance(step, ConditionalStep)
    # Keys are coerced to string form
    assert set(step.branches.keys()) == {"true", "false"}
    # Expression carried into metadata for observability
    assert step.meta.get("condition_expression") == "previous_step"

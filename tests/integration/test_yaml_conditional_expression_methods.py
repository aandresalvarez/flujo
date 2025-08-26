from __future__ import annotations

import pytest

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.models import PipelineContext


async def _run_pipeline(pipeline, input_data: str):
    # Minimal in-process runner for YAML → pipeline integration
    from flujo.application.core.executor_core import ExecutorCore

    core = ExecutorCore()
    ctx = PipelineContext(initial_prompt=input_data)
    result = await core.run_pipeline(pipeline, input_data, ctx)
    return result


def test_loader_compiles_conditional_with_string_methods() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: step
    name: seed
  - kind: conditional
    name: route_by_prefix
    condition_expression: "previous_step.lower().startswith('go') or previous_step.upper().endswith('!')"
    branches:
      true:
        - kind: step
          name: true_branch
      false:
        - kind: step
          name: false_branch
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    # Quick static checks
    assert len(pipeline.steps) == 2
    cond = pipeline.steps[1]
    assert getattr(cond, "name", "") == "route_by_prefix"
    assert cond.meta.get("condition_expression") is not None


@pytest.mark.asyncio
async def test_conditional_expression_string_methods_runtime() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: conditional
    name: route_by_prefix
    condition_expression: "previous_step.lower().startswith('go') or previous_step.upper().endswith('!')"
    branches:
      true:
        - kind: step
          name: ok
      false:
        - kind: step
          name: no_step
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    # Use standard runner helpers for consistency with the suite
    from tests.conftest import create_test_flujo
    from flujo.testing.utils import gather_result

    runner = create_test_flujo(pipeline)
    result1 = await gather_result(runner, "go")
    # Conditional step is final; assert branch metadata reflects True
    assert result1.step_history[-1].name == "route_by_prefix"
    assert result1.step_history[-1].metadata_.get("executed_branch_key") is True

    # Hello! → true (endswith '!') → ok
    result2 = await gather_result(runner, "Hello!")
    assert result2.step_history[-1].name == "route_by_prefix"
    assert result2.step_history[-1].metadata_.get("executed_branch_key") is True

    # xyz → false → no
    result3 = await gather_result(runner, "xyz")
    assert result3.step_history[-1].name == "route_by_prefix"
    assert result3.step_history[-1].metadata_.get("executed_branch_key") is False

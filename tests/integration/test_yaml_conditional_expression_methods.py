from __future__ import annotations

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


def test_conditional_expression_string_methods_runtime_event_loop(event_loop) -> None:
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
          name: no
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

    async def _run(msg: str):
        from flujo.application.core.executor_core import ExecutorCore
        from flujo.domain.models import PipelineContext

        core = ExecutorCore()
        ctx = PipelineContext(initial_prompt=msg)
        res = await core.run_pipeline(pipeline, msg, ctx)
        # Done if no exception
        return res

    # go → true → ok
    result1 = event_loop.run_until_complete(_run("go"))
    assert result1.step_history[-1].name == "ok"

    # Hello! → true (endswith '!') → ok
    result2 = event_loop.run_until_complete(_run("Hello!"))
    assert result2.step_history[-1].name == "ok"

    # xyz → false → no
    result3 = event_loop.run_until_complete(_run("xyz"))
    assert result3.step_history[-1].name == "no"

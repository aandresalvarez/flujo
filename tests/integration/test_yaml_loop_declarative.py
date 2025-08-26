from __future__ import annotations

from typing import Any


from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.models import PipelineContext


async def add_history_and_mark(
    data: Any, context: PipelineContext
) -> str:  # used by YAML via import
    sp = context.scratchpad
    sp.setdefault("history", [])
    sp["history"].append("Body")
    sp["counter"] = int(sp.get("counter", 0)) + 1
    sp["saw_context_input"] = isinstance(data, PipelineContext)
    return "ok"


def test_loop_init_runs_once_and_propagation_context_is_applied() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: declarative_loop_exec
    loop:
      init:
        - set: "context.scratchpad.counter"
          value: 0
        - append: { target: "context.scratchpad.history", value: "Init" }
      body:
        - kind: step
          name: mutate
          uses: "tests.integration.test_yaml_loop_declarative:add_history_and_mark"
          updates_context: true
      max_loops: 2
      propagation:
        next_input: context
      exit_expression: "context.scratchpad.counter >= 2"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

    core = ExecutorCore()
    ctx = PipelineContext(initial_prompt="seed")
    # Run the pipeline via the core (policy-driven)
    import asyncio

    result = asyncio.run(
        core._execute_pipeline_via_policies(
            pipeline,
            "input",  # initial data
            ctx,
            None,
            None,
            None,
            None,
        )
    )

    final_ctx = result.final_pipeline_context
    assert final_ctx is not None
    # Init applied once; at least one body append should be present in history
    assert final_ctx.scratchpad.get("history")[0] == "Init"
    assert "Body" in final_ctx.scratchpad.get("history")
    # Counter incremented twice by the body step
    assert final_ctx.scratchpad.get("counter") == 2
    # Propagation: second iteration received context as input and marked it
    assert final_ctx.scratchpad.get("saw_context_input") is True

from __future__ import annotations


from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.models import PipelineContext


class ItemsContext(PipelineContext):
    items: list[int] = []


async def double(x: int) -> int:  # used via import path in YAML
    return int(x) * 2


def test_map_init_and_finalize_end_to_end() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: map
    name: map_finalize
    map:
      iterable_input: items
      body:
        - kind: step
          name: double
          uses: "tests.integration.test_map_declarative:double"
      init:
        - set: "context.import_artifacts.note"
          value: "mapping"
      finalize:
        output:
          results_str: "{{ previous_step }}"
"""

    p = load_pipeline_blueprint_from_yaml(yaml_text)
    core = ExecutorCore()
    ctx = ItemsContext(initial_prompt="seed", items=[1, 2])

    import asyncio

    res = asyncio.run(core._execute_pipeline_via_policies(p, None, ctx, None, None, None))

    out = res.step_history[0].output if res.step_history else None
    assert isinstance(out, dict)
    # Finalize mapper should see aggregated list [2, 4]
    assert "results_str" in out
    s = str(out.get("results_str"))
    assert "2" in s and "4" in s

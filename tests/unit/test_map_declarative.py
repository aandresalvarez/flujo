from __future__ import annotations

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.dsl.loop import MapStep


def test_loader_compiles_map_init_and_finalize_meta() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: map
    name: map_demo
    map:
      iterable_input: items
      body:
        - kind: step
          name: echo
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      init:
        - set: "context.scratchpad.note"
          value: "started"
      finalize:
        output:
          results: "{previous_step}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    st = pipeline.steps[0]
    assert isinstance(st, MapStep)
    assert isinstance(getattr(st, "meta", {}), dict)
    assert "compiled_init_ops" in st.meta
    assert callable(st.meta["compiled_init_ops"])  # type: ignore[index]
    assert "map_finalize_mapper" in st.meta
    assert callable(st.meta["map_finalize_mapper"])  # type: ignore[index]

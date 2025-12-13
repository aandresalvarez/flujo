from __future__ import annotations

from typing import Any

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def _ctx() -> Any:
    class Ctx:
        def __init__(self) -> None:
            self.data_store = {"steps": {}, "history": []}
            self.metrics = {}

    return Ctx()


def test_loop_state_append_set_merge() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: L
    loop:
      body:
        - kind: step
          name: s
      max_loops: 2
      state:
        append:
          - target: "context.data_store.history"
            value: "OUT: {{ previous_step }}"
        set:
          - target: "context.summary"
            value: "{{ previous_step }}"
        merge:
          - target: "context.metrics"
            value: '{"count": 1}'
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    step = pipeline.steps[0]
    it_mapper = step.iteration_input_mapper
    assert callable(it_mapper)
    c = _ctx()
    out = it_mapper("foo", c, 1)
    # iteration mapper returns the output by default
    assert out == "foo"
    # append executed
    assert c.data_store["history"] == ["OUT: foo"]
    # set executed
    assert getattr(c, "summary", None) == "foo"
    # merge executed
    assert c.metrics == {"count": 1}

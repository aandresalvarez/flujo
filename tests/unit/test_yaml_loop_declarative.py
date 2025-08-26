from __future__ import annotations


from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.dsl.loop import LoopStep


def test_loader_compiles_loop_init_and_propagation_and_output() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: declarative_loop
    loop:
      body:
        - kind: step
          name: noop
      max_loops: 1
      init:
        - set: "context.scratchpad.flag"
          value: "on"
      propagation:
        next_input: context
      output:
        summary: "{context.scratchpad.flag}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert len(pipeline.steps) == 1
    loop_step = pipeline.steps[0]
    assert isinstance(loop_step, LoopStep)

    # Init ops are attached on meta for the policy to invoke
    assert isinstance(getattr(loop_step, "meta", {}), dict)
    assert "compiled_init_ops" in loop_step.meta
    assert callable(loop_step.meta["compiled_init_ops"])  # type: ignore[index]

    # Propagation produces an iteration_input_mapper
    assert callable(loop_step.iteration_input_mapper)

    # Output mapping compiles to loop_output_mapper
    assert callable(loop_step.loop_output_mapper)


def test_loader_compiles_output_template() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: loop_with_template
    loop:
      body:
        - kind: step
          name: noop
      exit_expression: "True"
      output_template: "Run: {context.run_id}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    loop_step = pipeline.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert callable(loop_step.loop_output_mapper)


def test_propagation_auto_chooses_context_when_updates_context_true() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: auto_prop_loop
    loop:
      body:
        - kind: step
          name: mutate
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
          updates_context: true
      exit_expression: "True"
      propagation: auto
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    loop_step = pipeline.steps[0]
    assert isinstance(loop_step, LoopStep)

    # The compiled iteration_input_mapper should return the context when propagation is auto
    from flujo.domain.models import PipelineContext

    ctx = PipelineContext(initial_prompt="x")
    out = loop_step.iteration_input_mapper("prev", ctx, 1)  # type: ignore[arg-type]
    assert out is ctx

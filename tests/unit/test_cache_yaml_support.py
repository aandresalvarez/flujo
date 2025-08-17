from flujo.domain.blueprint.loader import (
    load_pipeline_blueprint_from_yaml,
    dump_pipeline_blueprint_to_yaml,
)
from flujo.domain.dsl.pipeline import Pipeline


async def _echo(x: object) -> object:
    return x


def test_load_cache_step_wraps_inner_step():
    y = """
version: "0.1"
name: "cache-demo"
steps:
  - kind: cache
    name: CachedEcho
    wrapped_step:
      kind: step
      name: Echo
      agent: { id: "flujo.builtins.stringify" }
"""
    # Build the inner step programmatically (the loader's cache support is about structure)
    # Here we ensure the structure compiles without raising
    p = load_pipeline_blueprint_from_yaml(y)
    assert isinstance(p, Pipeline)
    assert len(p.steps) == 1


def test_dump_cache_step_emits_cache_kind():
    # Build from YAML to ensure end-to-end parity, then dump
    y = """
version: "0.1"
name: "cache-demo"
steps:
  - kind: cache
    name: CachedEcho
    wrapped_step:
      kind: step
      name: Echo
      agent: { id: "flujo.builtins.stringify" }
"""
    p = load_pipeline_blueprint_from_yaml(y)
    yaml_text = dump_pipeline_blueprint_to_yaml(p)
    assert "kind: cache" in yaml_text
    assert "wrapped_step" in yaml_text

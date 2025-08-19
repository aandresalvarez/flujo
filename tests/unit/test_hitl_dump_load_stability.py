import yaml

from flujo.domain.blueprint.loader import (
    dump_pipeline_blueprint_to_yaml,
    load_pipeline_blueprint_from_yaml,
)
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.dsl.pipeline import Pipeline


def test_hitl_dump_load_dump_is_stable_on_core_fields():
    # Initial pipeline with HITL
    p1 = Pipeline(steps=[HumanInTheLoopStep(name="Ask", message_for_user="Go?")])
    y1 = dump_pipeline_blueprint_to_yaml(p1)

    # Load then dump again
    p2 = load_pipeline_blueprint_from_yaml(y1)
    y2 = dump_pipeline_blueprint_to_yaml(p2)

    d1 = yaml.safe_load(y1)
    d2 = yaml.safe_load(y2)

    assert d1["version"] == d2["version"] == "0.1"
    assert isinstance(d1["steps"], list) and isinstance(d2["steps"], list)
    n1, n2 = d1["steps"][0], d2["steps"][0]
    # Core shape remains: kind/name/message keys
    assert n1.get("kind") == n2.get("kind") == "hitl"
    assert n1.get("name") == n2.get("name") == "Ask"
    assert n1.get("message") == n2.get("message") == "Go?"

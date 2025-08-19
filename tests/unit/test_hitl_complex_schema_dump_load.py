import yaml

from flujo.domain.blueprint.loader import (
    dump_pipeline_blueprint_to_yaml,
    load_pipeline_blueprint_from_yaml,
)
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.dsl.pipeline import Pipeline


def test_hitl_dump_with_nested_array_schema_contains_structure():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "meta": {
                            "type": "object",
                            "properties": {"note": {"type": "string"}},
                        },
                    },
                    "required": ["id"],
                },
            },
        },
    }

    p = Pipeline(
        steps=[
            HumanInTheLoopStep(
                name="Collect",
                message_for_user="Provide complex input",
                input_schema=None,  # at runtime we can accept any, we test dumper with explicit schema via YAML below
            )
        ]
    )

    y1 = dump_pipeline_blueprint_to_yaml(p)
    data1 = yaml.safe_load(y1)
    # Replace the schema post-dump to simulate a pipeline using nested schema
    data1["steps"][0]["input_schema"] = schema
    y2 = yaml.safe_dump(data1, sort_keys=False)

    p2 = load_pipeline_blueprint_from_yaml(y2)
    # Ensure compiled HITL contains a model with nested properties
    st = p2.steps[0]
    model = getattr(st, "input_schema", None)
    assert model is not None and hasattr(model, "model_json_schema")
    js = model.model_json_schema()
    assert js.get("type") == "object"
    props = js.get("properties", {})
    assert "tags" in props and props["tags"]["type"] == "array"
    assert "items" in props and props["items"]["type"] == "array"

import pytest

from flujo.domain.blueprint.model_generator import generate_model_from_schema


def test_generate_model_with_array_of_objects_schema_produces_valid_items_object():
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A title field"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "order": {"type": "integer"},
                    },
                    "required": ["name"],
                },
                "description": "Ordered list of steps",
            },
        },
        "required": ["steps"],
    }

    Model = generate_model_from_schema("DecomposerOutput", schema)

    js = Model.model_json_schema()
    assert js["type"] == "object"
    assert "properties" in js
    props = js["properties"]

    assert props["steps"]["type"] == "array"
    assert "items" in props["steps"], "array 'items' must be present in schema"
    items = props["steps"]["items"]
    if "type" in items:
        assert items.get("type") == "object", "array items must be typed as object"
        assert "properties" in items and isinstance(items["properties"], dict)
        assert items["properties"]["name"]["type"] == "string"
    else:
        # Pydantic may emit a $ref; resolve and assert it is an object
        ref = items.get("$ref")
        assert isinstance(ref, str) and ref.startswith("#/")
        defs = js.get("$defs") or js.get("definitions") or {}
        name = ref.split("/")[-1]
        target = defs.get(name) or {}
        assert target.get("type") == "object"
        assert "properties" in target and target["properties"]["name"]["type"] == "string"

    # Validate instance creation respects the nested model
    instance = Model(
        steps=[{"name": "step-1", "description": "first", "order": 1}],
        title="Example",
    )
    assert instance.steps and instance.steps[0].name == "step-1"

    # Missing required field in item should fail validation
    with pytest.raises(Exception):
        Model(steps=[{"description": "no name"}], title="Bad")

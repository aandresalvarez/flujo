from flujo.domain.blueprint.model_generator import generate_model_from_schema


def test_generate_model_with_array_of_strings_property_accepts_instances():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags",
            }
        },
        "required": ["tags"],
    }

    Model = generate_model_from_schema("ObjectWithTags", schema)
    js = Model.model_json_schema()

    # Schema reflects an array property; items may be untyped (Any)
    assert js["type"] == "object"
    assert js["properties"]["tags"]["type"] == "array"
    assert isinstance(js["properties"]["tags"].get("items"), dict)

    # Creating an instance with string items works
    m = Model(tags=["a", "b", "c"])
    assert m.tags == ["a", "b", "c"]


def test_generate_model_with_root_level_array_wraps_in_value_field():
    schema = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Root-level array schema",
    }

    Model = generate_model_from_schema("RootArray", schema)
    js = Model.model_json_schema()

    # Root arrays are wrapped in an object with a 'value' field
    assert js["type"] == "object"
    assert "value" in js["properties"]
    value = js["properties"]["value"]
    assert value["type"] == "array"
    assert isinstance(value.get("items"), dict)

    # Instance creation via 'value' field
    inst = Model(value=["x", "y"])
    assert inst.value == ["x", "y"]

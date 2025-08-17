from flujo.domain.blueprint.model_generator import generate_model_from_schema


def test_array_of_string_enum_items_produces_literal_items():
    schema = {
        "type": "object",
        "properties": {
            "choices": {
                "type": "array",
                "items": {"type": "string", "enum": ["a", "b"]},
            }
        },
    }
    Model = generate_model_from_schema("ChoiceList", schema)
    js = Model.model_json_schema()
    props = js.get("properties", {})
    assert props.get("choices", {}).get("type") == "array"
    items = props.get("choices", {}).get("items", {})
    # Pydantic may emit a $ref; resolve if needed
    if "$ref" in items:
        name = str(items["$ref"]).split("/")[-1]
        defs = js.get("$defs") or js.get("definitions") or {}
        lit = defs.get(name, {})
        assert lit.get("enum") == ["a", "b"]
    else:
        assert items.get("enum") == ["a", "b"]

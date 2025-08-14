from __future__ import annotations


from flujo.domain.blueprint.model_generator import generate_model_from_schema


def test_generate_model_from_object_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["a", "b"]},
            "score": {"type": "number", "description": "confidence score"},
            "flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["category"],
    }
    Model = generate_model_from_schema("CategorizerOutput", schema)
    assert hasattr(Model, "model_validate")
    obj = Model.model_validate({"category": "a", "score": 0.9, "flags": ["x"]})
    assert obj.category == "a"
    assert obj.score == 0.9
    assert obj.flags == ["x"]


def test_generate_model_from_primitive_schema() -> None:
    schema = {"type": "string", "description": "simple text"}
    Model = generate_model_from_schema("Simple", schema)
    inst = Model.model_validate({"value": "hello"})
    assert inst.value == "hello"

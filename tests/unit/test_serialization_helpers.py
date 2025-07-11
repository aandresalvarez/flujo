import warnings
from pydantic import BaseModel, field_serializer
from flujo.utils.serialization import create_field_serializer, serializable_field


class CustomType:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


def test_create_field_serializer_helper():
    class MyModel(BaseModel):
        custom: CustomType
        model_config = {"arbitrary_types_allowed": True}

        @field_serializer("custom", when_used="json")
        def serialize_custom(self, value):
            return create_field_serializer("custom", lambda x: x.to_dict())(value)

    obj = MyModel(custom=CustomType(123))
    result = obj.model_dump(mode="json")
    assert result == {"custom": {"value": 123}}


def test_serializable_field_deprecation():
    # Should warn and attach _serializer_func
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @serializable_field(lambda x: x.to_dict())
        def dummy_field():
            pass

        assert hasattr(dummy_field, "_serializer_func")
        assert any("deprecated" in str(warning.message).lower() for warning in w)


def test_manual_field_serializer():
    class MyModel(BaseModel):
        custom: CustomType
        model_config = {"arbitrary_types_allowed": True}

        @field_serializer("custom", when_used="json")
        def serialize_custom(self, value):
            return value.to_dict()

    obj = MyModel(custom=CustomType(456))
    result = obj.model_dump(mode="json")
    assert result == {"custom": {"value": 456}}

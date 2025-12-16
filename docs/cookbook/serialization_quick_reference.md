# Serialization Quick Reference

A quick reference guide for Flujo's enhanced serialization features.

## Global Registry

### Register a Custom Serializer

```python
from flujo.utils import register_custom_serializer

# Register once, use everywhere
register_custom_serializer(MyType, lambda x: x.to_dict())
```

### Common Patterns

```python
# Database connections
register_custom_serializer(DatabaseConnection, lambda conn: {
    "host": conn.host, "port": conn.port
})

# Complex numbers
register_custom_serializer(complex, lambda c: f"{c.real}+{c.imag}j")

# Enums
register_custom_serializer(Priority, lambda p: p.name.lower())

# Datetime
register_custom_serializer(datetime, lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S"))
```

## Model Configuration

### Enable Arbitrary Types

```python
from flujo.models import BaseModel

class MyModel(BaseModel):
    custom_field: MyCustomType
    model_config = {"arbitrary_types_allowed": True}
```

### Field-Specific Serialization

```python
from pydantic import field_serializer

class MyModel(BaseModel):
    custom_field: MyCustomType

    @field_serializer('custom_field', when_used='json')
    def serialize_custom_field(self, value: MyCustomType) -> dict:
        return value.to_dict()
```

## Serialization Modes

### Use Global Registry

```python
# This uses the global registry
serialized = model.model_dump(mode="json")

# This does NOT use the global registry
serialized = model.model_dump()
```

### JSON Export

```python
# Uses global registry
json_string = model.model_dump_json()
```

## Utility Functions

### Safe Serialization

```python
# For Pydantic models, call model_dump(mode="json") on the instance
result = complex_object.model_dump(mode="json")
```

### Create Type-Specific Serializer

```python
from flujo.utils import create_serializer_for_type

def serialize_my_type(obj: MyType) -> dict:
    return {"id": obj.id, "name": obj.name}

MyTypeSerializer = create_serializer_for_type(MyType, serialize_my_type)
```

### Field Decorator (Deprecated)

```python
# DEPRECATED - Use global registry or manual field_serializer instead
from flujo.utils import serializable_field

class MyModel(BaseModel):
    @serializable_field(lambda x: x.to_dict())
    complex_object: ComplexType
```

**Recommended alternatives:**

1. **Global Registry:**
```python
from flujo.utils import register_custom_serializer

register_custom_serializer(ComplexType, lambda x: x.to_dict())

class MyModel(BaseModel):
    complex_object: ComplexType  # Uses global serializer
```

2. **Manual field_serializer:**
```python
from pydantic import field_serializer

class MyModel(BaseModel):
    complex_object: ComplexType

    @field_serializer('complex_object', when_used='json')
    def serialize_complex_object(self, value: ComplexType) -> dict:
        return value.to_dict()
```

## Automatic Type Handling

The enhanced `BaseModel` automatically handles:

- ✅ **datetime** → ISO format strings
- ✅ **Enum** → enum value
- ✅ **complex** → `{"real": x, "imag": y}`
- ✅ **set/frozenset** → list
- ✅ **bytes/memoryview** → UTF-8 string
- ✅ **callable** → module.qualname
- ✅ **custom objects** → dict or string representation

## Integration Points

### State Backends

```python
# Custom types automatically serialized
backend = FileBackend("/tmp/state")
context = PipelineContext(custom_data=MyCustomType("value"))
# Serialization handled automatically
```

### Cache Keys

```python
# Custom types in cache keys automatically serialized
step = CacheStep.cached(some_step)
# Global registry used for cache key generation
```

### Pipeline Context

```python
# Custom types in context automatically handled
class CustomContext(PipelineContext):
    custom_data: MyCustomType
    model_config = {"arbitrary_types_allowed": True}
```

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| `"Unable to generate pydantic-core schema"` | Add `model_config = {"arbitrary_types_allowed": True}` |
| `"Object of type X is not JSON serializable"` | Register custom serializer: `register_custom_serializer(X, lambda x: x.to_dict())` |
| Global registry not working | Use `model_dump(mode="json")` instead of `model_dump()` |

### Migration Checklist

- [ ] Identify custom types in your models
- [ ] Register global serializers for custom types
- [ ] Add `arbitrary_types_allowed = True` where needed
- [ ] Test serialization with `model_dump(mode="json")`
- [ ] Update state backends if using custom serialization
- [ ] Verify cache keys work with custom types

## Best Practices

### ✅ Do

- Use global registry for application-wide custom types
- Use `model_dump(mode="json")` when you need custom serialization
- Register serializers early in your application startup
- Test serialization with complex nested objects

### ❌ Don't

- Don't use `model_dump()` when you need custom serialization
- Don't repeat serialization logic across multiple models
- Don't forget to handle `None` values in custom serializers
- Don't use complex serialization in performance-critical paths

## Examples

### Complete Example

```python
from flujo.utils import register_custom_serializer
from flujo.models import BaseModel
from datetime import datetime, timezone

# Register custom serializers
register_custom_serializer(datetime, lambda dt: dt.strftime("%Y-%m-%d"))
register_custom_serializer(complex, lambda c: f"{c.real}+{c.imag}j")

class MyModel(BaseModel):
    timestamp: datetime
    result: complex
    custom_data: Any
    model_config = {"arbitrary_types_allowed": True}

# Create and serialize
model = MyModel(
    timestamp=datetime.now(timezone.utc),
    result=3.14 + 2.71j,
    custom_data={"key": "value"}
)

# Use JSON mode for custom serialization
serialized = model.model_dump(mode="json")
# Result: {"timestamp": "2024-01-15", "result": "3.14+2.71j", "custom_data": {"key": "value"}}
```

For detailed documentation, see the [Advanced Serialization Guide](advanced_serialization.md).

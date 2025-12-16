# Advanced Serialization Guide

Flujo now uses Pydantic v2's native serialization with intelligent fallback handling for backward compatibility. This guide explains how to handle custom types and advanced serialization scenarios.

## Overview

The `BaseModel` in Flujo automatically handles serialization of common types:

- **Pydantic models**: Native serialization via `model_dump()`
- **datetime objects**: ISO format strings
- **Enum values**: The enum's value
- **Complex numbers**: `{"real": x, "imag": y}` format
- **Sets and frozensets**: Converted to lists
- **Bytes and memoryview**: UTF-8 decoded strings
- **Callable objects**: Module-qualified names (e.g., `"mymodule.my_function"`)
- **Custom objects**: Dictionary representation or string representation

## Automatic Fallback Serialization

Most custom types will work automatically without any configuration:

```python
from flujo.models import BaseModel
from datetime import datetime, timezone
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"

class MyContext(BaseModel):
    timestamp: datetime
    status: Status
    custom_data: Any  # Will be automatically serialized

# This works automatically
context = MyContext(
    timestamp=datetime.now(timezone.utc),
    status=Status.PENDING,
    custom_data={"complex": "data"}
)
```

## Global Custom Serializer Registry

Flujo provides a powerful global registry for custom serializers that applies to all models throughout your application. This is the most robust solution for handling custom types.

### Basic Usage

Register a custom serializer for any type globally:

```python
from flujo.utils import register_custom_serializer
from datetime import datetime

def serialize_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# Register the serializer globally
register_custom_serializer(datetime, serialize_datetime)
```

Now, **any** Flujo model containing a `datetime` will use your custom format when serialized with `model_dump(mode="json")` or `model_dump_json()`.

### Advanced Examples

#### Custom Object Serialization

```python
from flujo.utils import register_custom_serializer
from flujo.models import BaseModel

class DatabaseConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

def serialize_db_connection(conn: DatabaseConnection) -> dict:
    return {
        "type": "database_connection",
        "host": conn.host,
        "port": conn.port
    }

# Register the serializer
register_custom_serializer(DatabaseConnection, serialize_db_connection)

class MyModel(BaseModel):
    db: DatabaseConnection
    model_config = {"arbitrary_types_allowed": True}

# Now this will serialize correctly
model = MyModel(db=DatabaseConnection("localhost", 5432))
serialized = model.model_dump(mode="json")
# Result: {"db": {"type": "database_connection", "host": "localhost", "port": 5432}}
```

#### Complex Number Custom Format

```python
from flujo.utils import register_custom_serializer

def serialize_complex(c: complex) -> str:
    return f"{c.real:.2f} + {c.imag:.2f}i"

register_custom_serializer(complex, serialize_complex)

class CalculationResult(BaseModel):
    result: complex
    model_config = {"arbitrary_types_allowed": True}

calc = CalculationResult(result=3.14159 + 2.71828j)
serialized = calc.model_dump(mode="json")
# Result: {"result": "3.14 + 2.72i"}
```

#### Custom Enum Serialization

```python
from enum import Enum
from flujo.utils import register_custom_serializer

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

def serialize_priority(p: Priority) -> str:
    return p.name.lower()

register_custom_serializer(Priority, serialize_priority)

class Task(BaseModel):
    priority: Priority
    model_config = {"arbitrary_types_allowed": True}

task = Task(priority=Priority.HIGH)
serialized = task.model_dump(mode="json")
# Result: {"priority": "high"}
```

### Registry Management

The global registry is thread-safe and persists throughout your application's lifetime. You can register multiple serializers for different types:

```python
from flujo.utils import register_custom_serializer
from decimal import Decimal
from pathlib import Path

# Register multiple serializers
register_custom_serializer(Decimal, lambda d: str(d))
register_custom_serializer(Path, lambda p: str(p))
register_custom_serializer(set, list)  # Convert sets to lists
```

### When to Use Global Registry

Use the global registry when:

1. **You have custom types used throughout your application**
2. **You want consistent serialization across all models**
3. **You're working with third-party types that need custom serialization**
4. **You want to avoid repeating `@field_serializer` decorators**

## Custom Serialization for Specific Fields

For fields that need custom serialization, use the `@field_serializer` decorator:

```python
from pydantic import field_serializer
from flujo.models import BaseModel

class MyModel(BaseModel):
    custom_field: MyCustomType

    @field_serializer('custom_field', when_used='json')
    def serialize_custom_field(self, value: MyCustomType) -> dict:
        return {
            "data": value.data,
            "metadata": value.metadata
        }
```

## Using the Convenience Decorator (Deprecated)

Flujo previously provided a convenience decorator for field serialization, but this approach has fundamental design issues with Pydantic v2 and is now deprecated:

```python
# DEPRECATED - Don't use this approach
from flujo.utils import serializable_field
from flujo.models import BaseModel

class MyModel(BaseModel):
    @serializable_field(lambda x: x.to_dict())
    complex_object: ComplexType
```

**Recommended alternatives:**

1. **Global Registry (Recommended):**
```python
from flujo.utils import register_custom_serializer

# Register once, use everywhere
register_custom_serializer(ComplexType, lambda x: x.to_dict())

class MyModel(BaseModel):
    complex_object: ComplexType  # Will use global serializer automatically
```

2. **Manual field_serializer:**
```python
from pydantic import field_serializer
from flujo.models import BaseModel

class MyModel(BaseModel):
    complex_object: ComplexType

    @field_serializer('complex_object', when_used='json')
    def serialize_complex_object(self, value: ComplexType) -> dict:
        return value.to_dict()
```

## Creating Custom Serializers

For reusable serialization logic, create custom serializer functions:

```python
from flujo.utils import create_serializer_for_type

# Create a serializer for a specific type
def serialize_my_type(obj: MyType) -> dict:
    return {"id": obj.id, "name": obj.name}

MyTypeSerializer = create_serializer_for_type(MyType, serialize_my_type)

# For Pydantic models, call model_dump(mode="json") on the instance
serialized = complex_object.model_dump(mode="json")
```

## Handling Non-Serializable Types

If you encounter types that cannot be serialized automatically, you have several options:

### Option 1: Use Global Registry (Recommended)

```python
from flujo.utils import register_custom_serializer

def serialize_my_type(obj: MyCustomType) -> str:
    return obj.get_serializable_representation()

register_custom_serializer(MyCustomType, serialize_my_type)
```

### Option 2: Use `@field_serializer`

```python
from pydantic import field_serializer
from flujo.models import BaseModel

class MyModel(BaseModel):
    non_serializable_field: MyCustomType

    @field_serializer('non_serializable_field', when_used='json')
    def serialize_custom_field(self, value: MyCustomType) -> str:
        return value.get_serializable_representation()
```

### Option 3: Convert to Serializable Types

```python
from flujo.models import BaseModel

class MyModel(BaseModel):
    # Store as string instead of custom object
    custom_data: str

    def set_custom_data(self, obj: MyCustomType):
        self.custom_data = obj.to_json()

    def get_custom_data(self) -> MyCustomType:
        return MyCustomType.from_json(self.custom_data)
```

### Option 4: Use `Any` Type with Automatic Fallback

```python
from flujo.models import BaseModel

class MyModel(BaseModel):
    # The BaseModel will automatically handle serialization
    custom_data: Any
```

## Migration from Previous Serialization

If you were using the old custom serialization approach, here's how to migrate:

### Before (Old Approach)
```python
class MyModel(BaseModel):
    custom_field: MyCustomType

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data['custom_field'] = self.custom_field.to_dict()
        return data
```

### After (New Approach)
```python
from pydantic import field_serializer
from flujo.models import BaseModel

class MyModel(BaseModel):
    custom_field: MyCustomType

    @field_serializer('custom_field', when_used='json')
    def serialize_custom_field(self, value: MyCustomType) -> dict:
        return value.to_dict()
```

### Or Use Global Registry (Even Better)
```python
from flujo.utils import register_custom_serializer

# Register once, use everywhere
register_custom_serializer(MyCustomType, lambda x: x.to_dict())

# No need to modify individual models
class MyModel(BaseModel):
    custom_field: MyCustomType  # Will use global serializer automatically
```

## Best Practices

1. **Use Pydantic's native types when possible**: `str`, `int`, `float`, `bool`, `dict`, `list`, etc.
2. **For custom objects, use the global registry**: This provides the most consistent and maintainable approach.
3. **Use `@field_serializer` for model-specific serialization**: When you need different serialization for the same type in different models.
4. **Use `Any` type sparingly**: While it works, it's less type-safe.
5. **Test serialization**: Always test that your models can be serialized and deserialized correctly.
6. **Consider performance**: Complex serialization logic can impact performance in high-throughput scenarios.

## Troubleshooting

### Common Issues

**TypeError: Object of type X is not JSON serializable**

This usually means you have a field that cannot be automatically serialized. Use the global registry or `@field_serializer`:

```python
# Global registry approach
register_custom_serializer(ProblematicType, lambda x: str(x))

# Or field-specific approach
@field_serializer('problematic_field', when_used='json')
def serialize_problematic_field(self, value):
    return str(value)
```

**Serialization is inconsistent**

Ensure your serialization logic is deterministic and handles all edge cases:

```python
def serialize_my_type(value):
    if value is None:
        return None
    return value.to_dict()  # Ensure this method exists and is consistent
```

**Global registry not working**

Make sure you're using `model_dump(mode="json")` or `model_dump_json()`:

```python
# This will use the global registry
serialized = model.model_dump(mode="json")

# This will NOT use the global registry (uses native Pydantic serialization)
serialized = model.model_dump()
```

### Debugging Serialization

Use the `model_dump(mode="json")` utility to test serialization:

```python
from flujo.utils import model_dump(mode="json")

# Test if an object can be serialized
try:
    result = model_dump(mode="json")(my_object)
    print(f"Serialized: {result}")
except Exception as e:
    print(f"Serialization failed: {e}")
```

## Advanced Usage

### Conditional Serialization

You can make serialization conditional based on context:

```python
@field_serializer('field', when_used='json')
def serialize_field(self, value):
    if self.some_condition:
        return value.serialize_for_json()
    else:
        return value.serialize_for_storage()
```

### Chaining Serializers

You can chain multiple serializers for complex types:

```python
def serialize_nested_object(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return str(obj)

register_custom_serializer(ComplexType, serialize_nested_object)
```

### Serialization with Context

For serializers that need access to the model instance:

```python
@field_serializer('field', when_used='json')
def serialize_field(self, value):
    # Access to self (the model instance)
    return value.serialize_with_context(self.context)
```

## Performance Considerations

- **Global registry lookups are fast**: The registry uses a simple dictionary lookup.
- **Avoid complex serialization in hot paths**: For high-throughput scenarios, consider pre-serializing data.
- **Use `model_dump(mode="json")` sparingly**: It's slower than `model_dump()` but necessary for custom serialization.

## Integration with Flujo Features

The global serializer registry works seamlessly with all Flujo features:

- **State backends**: Custom types are automatically serialized when saving state
- **Cache keys**: Custom types in cache keys are properly serialized
- **Context updates**: Custom types in pipeline context are handled correctly
- **JSON export**: All custom serializers are used when exporting to JSON

This approach provides maximum flexibility while maintaining backward compatibility and leveraging Pydantic v2's performance benefits.

## Safe Deserialization

Flujo offers a matching `safe_deserialize` helper to reconstruct objects previously serialized with `model_dump(mode="json")`. Use it together with the custom deserializer registry when reading JSON back into Python objects.

### Registering Custom Deserializers

```python
from flujo.utils import register_custom_serializer, register_custom_deserializer, safe_deserialize

class Widget:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, data: dict) -> "Widget":
        return cls(data["name"])

register_custom_serializer(Widget, lambda w: w.to_dict())
register_custom_deserializer(Widget, Widget.from_dict)

widget = Widget("demo")
serialized = widget.to_dict()  # or a Pydantic model could call .model_dump(mode="json")
restored = safe_deserialize(serialized, Widget)
assert isinstance(restored, Widget)
```

Flujo state backends and CLI commands call `safe_deserialize` to automatically rebuild custom types when loading data. Register your deserializers to enable full round-trip behavior.

## Robust Serialization for Logging

For debugging or prompt-generation scenarios where serialization errors should not halt execution, use the internal never-raise helpers:

```python
import json
from flujo.utils.serialization import _robust_serialize_internal, _serialize_to_json_internal

# Option 1: get a JSON-ready object that never raises
safe_obj = _robust_serialize_internal(complex_object)
json_str = json.dumps(safe_obj, ensure_ascii=False)

# Option 2: get a JSON string directly (never raises)
json_str = _serialize_to_json_internal(complex_object)
```

These helpers fall back to safe string placeholders for unknown types, ensuring logging and prompt assembly continue even with unexpected data.

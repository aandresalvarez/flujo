# How to Create a Custom StateBackend

Sometimes you need to store workflow state in a system not supported out of the box. This guide shows how to implement your own backend by walking through a simplified Redis example.

## The StateBackend Contract

Any backend must implement three asynchronous methods:

```python
class StateBackend(ABC):
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None: ...
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]: ...
    async def delete_state(self, run_id: str) -> None: ...
```

`state` is the serialized `WorkflowState` dictionary. Backends are responsible for storing and retrieving this object, handling any serialization and ensuring atomic writes.

## Tutorial: Redis Backend

```python
import json
import redis.asyncio as redis
from flujo.state.backends.base import StateBackend, _serialize_for_json
from flujo.utils.serialization import safe_deserialize

class RedisBackend(StateBackend):
    def __init__(self, url: str) -> None:
        self._url = url
        self._client: redis.Redis | None = None

    async def _conn(self) -> redis.Redis:
        if self._client is None:
            self._client = await redis.from_url(self._url)
        return self._client

    async def save_state(self, run_id: str, state: dict) -> None:
        r = await self._conn()
        # Use _serialize_for_json for JSON-safe serialization
        serialized_state = _serialize_for_json(state)
        await r.set(run_id, json.dumps(serialized_state))

    async def load_state(self, run_id: str) -> dict | None:
        r = await self._conn()
        data = await r.get(run_id)
        return safe_deserialize(json.loads(data)) if data else None

    async def delete_state(self, run_id: str) -> None:
        r = await self._conn()
        await r.delete(run_id)
```

### Pydantic Model Serialization

For Pydantic models, use `model_dump(mode="json")` for native JSON-safe output:

```python
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
    value: int

# Serialize Pydantic models directly
model = MyModel(name="test", value=42)
json_data = model.model_dump(mode="json")
```

### Enhanced Serialization for Custom Types

For mixed payloads or custom types, register serializers in the global registry:

```python
from flujo.utils import register_custom_serializer, register_custom_deserializer

# Register custom serializers for your types
def serialize_my_type(obj: MyCustomType) -> dict:
    return {"id": obj.id, "name": obj.name}

register_custom_serializer(MyCustomType, serialize_my_type)
register_custom_deserializer(MyCustomType, lambda d: MyCustomType(**d))

# Now your custom types are automatically serialized in state backends
```

### Custom Serialization for Specific Types

If you need custom serialization for specific types in your backend:

```python
from flujo.state.backends.base import _serialize_for_json

class CustomBackend(StateBackend):
    async def save_state(self, run_id: str, state: dict) -> None:
        # Use _serialize_for_json for robust handling of custom types
        serialized = _serialize_for_json(state)
        # Your storage logic here...
```

## Best Practices

1. **Use `model_dump(mode="json")` for Pydantic models**: Native, fast serialization
2. **Use `_serialize_for_json` for mixed payloads**: Handles primitives and custom types
3. **Register global serializers/deserializers**: Keep your type conversions centralized
4. **Handle errors gracefully**: The serialization utilities include error handling and fallbacks
5. **Test with complex objects**: Ensure your backend works with nested Pydantic models and custom types

## Serialization Summary

Flujo provides two recommended serialization approaches:

| Use Case | Recommended Approach |
|----------|---------------------|
| Pydantic models | `model.model_dump(mode="json")` |
| Mixed payloads / primitives | `_serialize_for_json(data)` |

The `_serialize_for_json` helper is available from `flujo.state.backends.base` and handles:
- **Pydantic models**: Via `model_dump(mode="json")`
- **Primitives**: Passed through unchanged
- **Custom types**: Via the global serializer registry
- **Edge cases**: Datetime, enums, complex numbers, bytes, etc.

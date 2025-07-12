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
from flujo.state.backends.base import StateBackend
from flujo.utils.serialization import safe_serialize, safe_deserialize

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
        # Use enhanced serialization for custom types
        serialized_state = safe_serialize(state)
        await r.set(run_id, json.dumps(serialized_state))

    async def load_state(self, run_id: str) -> dict | None:
        r = await self._conn()
        data = await r.get(run_id)
        return safe_deserialize(json.loads(data)) if data else None

    async def delete_state(self, run_id: str) -> None:
        r = await self._conn()
        await r.delete(run_id)
```

### Enhanced Serialization

The enhanced serialization approach automatically handles custom types through the global registry:

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
from flujo.utils import safe_serialize

class CustomBackend(StateBackend):
    async def save_state(self, run_id: str, state: dict) -> None:
        # Use safe_serialize for robust handling of custom types
        serialized = safe_serialize(state)
        # Your storage logic here...
```

## Best Practices

1. **Use `safe_serialize` and `safe_deserialize`**: Together they handle custom types round-trip
2. **Register global serializers/deserializers**: Keep your type conversions centralized
3. **Handle errors gracefully**: The enhanced serialization includes error handling and fallbacks
4. **Test with complex objects**: Ensure your backend works with nested Pydantic models and custom types

## Migration from orjson

If you were previously using orjson, the enhanced serialization approach provides better compatibility:

```python
# Before (with orjson)
import orjson

def pydantic_default(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError

serialized = orjson.dumps(state, default=pydantic_default)

# After (with enhanced serialization)
from flujo.utils import safe_serialize

serialized = safe_serialize(state)
json_string = json.dumps(serialized)
```

The enhanced approach provides:
- **Better error handling**: Graceful fallbacks for unsupported types
- **Global registry**: Consistent serialization across your application
- **Automatic Pydantic support**: No need for custom default handlers
- **Backward compatibility**: Works with existing code

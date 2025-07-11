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
import orjson
import redis.asyncio as redis
from flujo.state.backends.base import StateBackend

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
        await r.set(run_id, orjson.dumps(state))

    async def load_state(self, run_id: str) -> dict | None:
        r = await self._conn()
        data = await r.get(run_id)
        return orjson.loads(data) if data else None

    async def delete_state(self, run_id: str) -> None:
        r = await self._conn()
        await r.delete(run_id)
```

### Serializer Customization

`StateBackend` constructors accept a ``serializer_default`` callable used with
``orjson.dumps``. This lets you handle additional types:

```python
def my_serializer(obj):
    if isinstance(obj, MyModel):
        return obj.model_dump()
    raise TypeError

backend = RedisBackend("redis://localhost:6379/0", serializer_default=my_serializer)
```

Use your backend when creating a runner:

```python
backend = RedisBackend("redis://localhost:6379/0")
runner = Flujo(
    registry=registry,
    pipeline_name="my_pipeline",
    pipeline_version="1.0.0",
    state_backend=backend,
)
```

This pattern lets you integrate Flujo with any durable storage system.

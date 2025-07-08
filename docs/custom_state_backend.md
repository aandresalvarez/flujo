# How to Create a Custom StateBackend

Sometimes you need to store workflow state in a system that `flujo` does not provide out of the box. The `StateBackend` interface lets you plug in any storage layer.

## The StateBackend Contract
Each backend implements three async methods:

```python
class StateBackend(ABC):
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None: ...
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]: ...
    async def delete_state(self, run_id: str) -> None: ...
```
`state` is a dictionary created from `WorkflowState.model_dump()`.
Make sure writes are atomic to avoid corrupted data.

## Step-by-Step Tutorial: Redis Example
Below is a simplified backend using `redis-py`.
```python
import orjson
import redis.asyncio as redis
from flujo.state.backends.base import StateBackend

class RedisBackend(StateBackend):
    def __init__(self, url: str) -> None:
        self.client = redis.from_url(url)

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        await self.client.set(run_id, orjson.dumps(state))

    async def load_state(self, run_id: str) -> dict | None:
        data = await self.client.get(run_id)
        return orjson.loads(data) if data else None

    async def delete_state(self, run_id: str) -> None:
        await self.client.delete(run_id)
```
Use it with `Flujo` just like the built-in backends:
```python
backend = RedisBackend("redis://localhost:6379/0")
runner = Flujo(registry=my_registry, pipeline_name="demo", state_backend=backend)
```

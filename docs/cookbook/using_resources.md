# Cookbook: Sharing Resources Like a Database Connection

## The Problem

Your pipeline steps need to interact with a shared, long-lived resource like a database connection pool, a secrets manager client, or a reusable API client. Creating a new connection for every single step is inefficient and hard to manage.

## The Solution

The `Flujo` runner accepts a `resources` object that it will pass to every agent and plugin in the pipeline. You define the shape of this object by inheriting from `AppResources`. If your resource implements `__enter__/__aenter__`, Flujo will enter and exit it **per step attempt** (retries and parallel branches included) so you can tie transactions or ephemeral handles to one attempt.

```python
from unittest.mock import MagicMock
from flujo import Flujo, Step, AppResources

# 1. Define the structure of your shared resources
class MyWebAppResources(AppResources):
    db_conn: MagicMock
    secrets_client: MagicMock

# 2. Create an agent that declares a 'resources' dependency
class UserLookupAgent:
    async def run(self, user_id: int, *, resources: MyWebAppResources) -> str:
        # The engine will inject the resources object here
        print(f"AGENT: Looking up user {user_id}...")
        user_name = resources.db_conn.get_user_by_id(user_id)
        return user_name

# 3. Instantiate your resources and the runner
# In a real app, you would create real DB connections here.
# For this example, we'll use mocks.
shared_resources = MyWebAppResources(
    db_conn=MagicMock(),
    secrets_client=MagicMock()
)
shared_resources.db_conn.get_user_by_id.return_value = "Alice"

pipeline = Step("lookup_user", UserLookupAgent())
runner = Flujo(pipeline, resources=shared_resources)

# 4. Run the pipeline
result = runner.run(123)

# 5. Verify the resource was used
shared_resources.db_conn.get_user_by_id.assert_called_once_with(123)
print(f"\n✅ Agent successfully used the database connection to find: {result.step_history[0].output}")
```

### How It Works

1.  We create our own `MyWebAppResources` class that inherits from `flujo.AppResources`. This provides a clear, type-safe contract for our shared objects.
2.  The `UserLookupAgent`'s `run` method declares a **keyword-only argument** named `resources` and type-hints it with our custom class. This is the signal to the `Flujo` engine to inject the object.
3.  We instantiate our resources *once* and pass the `shared_resources` object to the `Flujo` constructor.
4.  When the engine executes the `lookup_user` step, it inspects the agent's `run` method signature, sees the `resources` parameter, and passes the shared object to it.
5.  The agent can then use the methods on the injected object (e.g., `resources.db_conn.get_user_by_id`).

This dependency injection pattern keeps your agents clean and decoupled from how resources are created, making them much easier to test and maintain.

### Transactional example

For database-backed workloads, make your resources a context manager so each step attempt gets its own transaction/session:

```python
class DBResources(AppResources):
    pool: Any
    session: Any | None = None

    async def __aenter__(self) -> "DBResources":
        self.session = await self.pool.acquire()
        await self.session.begin()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type:
                await self.session.rollback()
            else:
                await self.session.commit()
        finally:
            await self.pool.release(self.session)
            self.session = None
```

Because Flujo enters the context per attempt, a failed retry or paused HITL step will roll back before the next attempt runs. Ensure your context manager is re-entrant (or hands out per-attempt handles) if you plan to run steps in parallel.

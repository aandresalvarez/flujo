# Context & Resources

Understand how data flows through pipelines and how to manage external resources.

## Overview

- **Context**: A `PipelineContext` instance shared across steps. Steps declare a `context` kwarg to read/update it. Retries use isolated copies and merge on success.
- **Resources**: An optional `AppResources` container passed to agents/plugins via a `resources` kwarg. If the object implements `__enter__/__aenter__`, Flujo enters/exits it **per step attempt** (including retries and parallel branches) so you can scope transactions or temporary handles to one attempt. Non-context-manager resources are injected as-is.
- **State Providers**: External state managers for `ContextReference` fields. Use these to hydrate large datasets (databases, knowledge graphs) on-demand instead of serializing them in the context.
- **Parallelism**: Make resource context managers re-entrant or return per-attempt handles; parallel branches may enter the same object concurrently.
- **Injection**: Type-hint `resources: MyResources` (subclass of `AppResources`) on keyword-only params to receive it automatically.

---

## State Providers and ContextReference

For production workloads, you often need to reference large external datasets (knowledge graphs, conversation history, vector stores) without serializing them into the context object. **ContextReference** provides lightweight pointers to external state, while **StateProvider** handles the actual data loading and persistence.

### Why Use ContextReference?

| Approach | Context Size | Serialization | Best For |
|----------|-------------|---------------|----------|
| Store data in context | Large (grows with data) | Full serialization each step | Small datasets |
| **ContextReference** | Small (just pointers) | Only pointers serialized | Large/external datasets |

### Defining a StateProvider

A `StateProvider` implements a small async lifecycle:

```python
from flujo.domain.interfaces import StateProvider
from typing import Any

class KnowledgeGraphProvider(StateProvider):
    """Manages a knowledge graph stored in a database."""
    
    def __init__(self, db_connection):
        self._db = db_connection
    
    async def load(self, key: str) -> Any:
        """Fetch data from external storage."""
        return await self._db.query(f"SELECT * FROM graphs WHERE key = ?", key)
    
    async def save(self, key: str, data: Any) -> None:
        """Commit data to external storage."""
        await self._db.execute(
            "INSERT OR REPLACE INTO graphs (key, data) VALUES (?, ?)",
            key, data
        )

    async def delete(self, key: str) -> bool:
        """Optional: delete a record; return True if removed."""
        return await self._db.execute("DELETE FROM graphs WHERE key = ?", key) > 0

    async def exists(self, key: str) -> bool:
        """Optional: check if a record exists."""
        row = await self._db.query("SELECT 1 FROM graphs WHERE key = ? LIMIT 1", key)
        return row is not None

    async def close(self) -> None:
        """Optional: release connections/caches."""
        await self._db.close()
```

### Using ContextReference in Your Context

Define a `ContextReference` field in your custom context:

```python
from flujo.domain.models import PipelineContext, ContextReference
from typing import List
from pydantic import Field

class ResearchContext(PipelineContext):
    """Context with a reference to an external knowledge graph."""
    
    # This is a lightweight pointer, not the actual data
    knowledge_graph: ContextReference[List[dict]] = Field(
        default_factory=lambda: ContextReference(
            provider_id="kg_provider",  # Maps to state_providers key
            key="research_graph"         # Key passed to provider.load()
        )
    )
```

### Registering State Providers with Flujo

Pass your providers when creating the `Flujo` runner:

```python
from flujo import Flujo, Step

# Create your provider
kg_provider = KnowledgeGraphProvider(my_db_connection)

# Register with Flujo
runner = Flujo(
    pipeline=my_pipeline,
    context_model=ResearchContext,
    state_providers={
        "kg_provider": kg_provider  # Key matches provider_id in ContextReference
    }
)
```

### Accessing Hydrated Data in Steps

When your step runs, Flujo automatically hydrates the `ContextReference`:

```python
from flujo import step

@step
async def analyze_research(query: str, *, context: ResearchContext) -> str:
    # Data is automatically loaded from the provider
    graph_data = context.knowledge_graph.get()
    
    # Use the data
    relevant_nodes = [n for n in graph_data if query in n.get("topic", "")]
    
    return f"Found {len(relevant_nodes)} relevant nodes"

@step
async def update_graph(new_data: dict, *, context: ResearchContext) -> str:
    # Get current data
    graph = context.knowledge_graph.get()
    
    # Modify
    graph.append(new_data)
    
    # Set back (will be persisted after step completes)
    context.knowledge_graph.set(graph)
    
    return "Graph updated"
```

### Complete Example

```python
import asyncio
from typing import List
from flujo import Flujo, Step, step
from flujo.domain.models import PipelineContext, ContextReference
from flujo.domain.interfaces import StateProvider
from pydantic import Field

# 1. Define your StateProvider
class InMemoryKGProvider(StateProvider):
    def __init__(self):
        self._storage = {}
    
    async def load(self, key: str) -> List[dict]:
        return self._storage.get(key, [])
    
    async def save(self, key: str, data: List[dict]) -> None:
        self._storage[key] = data

# 2. Define context with ContextReference
class MyContext(PipelineContext):
    facts: ContextReference[List[dict]] = Field(
        default_factory=lambda: ContextReference(
            provider_id="facts_provider",
            key="user_facts"
        )
    )

# 3. Create steps that use the reference
@step
async def add_fact(fact: str, *, context: MyContext) -> str:
    facts = context.facts.get()
    facts.append({"content": fact, "verified": False})
    context.facts.set(facts)
    return f"Added fact. Total: {len(facts)}"

@step
async def count_facts(data: str, *, context: MyContext) -> str:
    facts = context.facts.get()
    return f"You have {len(facts)} facts stored."

# 4. Build pipeline and run
async def main():
    provider = InMemoryKGProvider()
    # Pre-populate some data
    provider._storage["user_facts"] = [{"content": "Initial fact", "verified": True}]
    
    pipeline = add_fact >> count_facts
    
    runner = Flujo(
        pipeline=pipeline,
        context_model=MyContext,
        state_providers={"facts_provider": provider}
    )
    
    async with runner:
        result = await runner.run_async("New discovery about AI")
    
    print(result.output)  # "You have 2 facts stored."
    print(provider._storage["user_facts"])  # Shows both facts

asyncio.run(main())
```

### Best Practices

1. **Use for Large Data**: Reserve `ContextReference` for data too large to serialize efficiently
2. **Immutable Keys**: Keep `provider_id` and `key` stable across pipeline versions
3. **Error Handling**: Implement graceful degradation if provider fails to load
4. **Testing**: Mock providers in tests to avoid external dependencies

```python
# In tests, use a mock provider
class MockProvider(StateProvider):
    async def load(self, key: str) -> list:
        return [{"test": "data"}]
    
    async def save(self, key: str, data) -> None:
        pass

runner = Flujo(
    pipeline=my_pipeline,
    context_model=MyContext,
    state_providers={"kg_provider": MockProvider()}
)
```

### When NOT to Use ContextReference

- **Small datasets**: If your data fits comfortably in context, use regular fields
- **Frequently changing keys**: If `key` changes every step, overhead may exceed benefit
- **No persistence needed**: If data is computed fresh each run, use regular context

## RAG Recipe with Flujo Memory

This recipe shows how to enable Retrieval-Augmented Generation using Flujo’s vector memory.

### Configure persistence
- For local dev (durable): set `state_uri = "sqlite:///.flujo/state.db"` in `flujo.toml`.
- For production: set `state_uri = "postgres://user:pass@host:5432/dbname"` (pgvector required; see migrations).
- Enable memory indexing and choose an embedding model:
```toml
[settings]
memory_indexing_enabled = true
memory_embedding_model = "openai:text-embedding-3-small"
```

### Write a pipeline step that stores outputs
- Ensure steps declare outputs so they can be indexed:
```python
Step(name="summarize", agent=..., output_keys=["summary"])
```
- MemoryManager indexes successful step outputs automatically when enabled.

### Retrieve memories in later steps
```python
from flujo.domain.models import PipelineContext

async def recall(data, context: PipelineContext):
    results = await context.retrieve(query_text="latest summary", limit=3)
    return {"memories": [r.record.payload for r in results]}
```

### Sandbox / embeddings notes
- Memory indexing is optional; if embeddings fail to initialize, it falls back to no-op.
- SQLite uses a naive cosine search; Postgres uses pgvector (`CREATE EXTENSION vector;`).

### CLI tips
- Ensure migrations are applied (`flujo migrate`) so `memories` (pgvector) tables exist.
- Use `FLUJO_MEMORY_INDEXING_ENABLED=1` and `FLUJO_MEMORY_EMBEDDING_MODEL=...` in CI or env overrides.

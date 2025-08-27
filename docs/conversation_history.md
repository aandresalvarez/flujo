# Conversation History in PipelineContext

Flujo adds a first‑class, typed field on `PipelineContext` to support conversational loops:

- Field: `conversation_history: list[ConversationTurn]`
- Types: `ConversationTurn` with `role: ConversationRole (user|assistant)` and `content: str`

Key properties
- Persistence: Serialized with the rest of the context and persisted to state backends (e.g., SQLite). Round‑trips via `StateManager` and `StateSerializer` without custom adapters.
- Scope: Populated and used when a loop is configured with `conversation: true` (see FSD-033). Outside that mode, the field remains an empty list by default and adds no overhead.
- Safety: The history is bounded and optionally summarized by `HistoryManager` during prompt injection to control costs and context length.

Notes
- The slot is intentionally simple (text only) in v1 to maximize portability across providers. Rich content can be added in a backward‑compatible evolution.
- For HITL workflows, human responses can be mirrored as `user` turns by the loop policy when enabled.


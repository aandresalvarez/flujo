#!/usr/bin/env python3
"""
State Providers and ContextReference Demo

This example demonstrates how to use StateProviders and ContextReference
for managing large external state (databases, knowledge graphs, etc.)
without serializing the entire dataset into the pipeline context.

Key Concepts:
1. StateProvider - Protocol for loading/saving external state
2. ContextReference - Lightweight pointer to external data
3. state_providers - Flujo parameter to register providers

Run: uv run python examples/state_providers_demo.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List
from uuid import uuid4

from pydantic import Field

from flujo import Flujo, step
from flujo.domain.models import PipelineContext, ContextReference, BaseModel
from flujo.domain.interfaces import StateProvider


# =============================================================================
# DOMAIN MODELS
# =============================================================================


class ResearchNode(BaseModel):
    """A node in our research knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    topic: str
    assertion: str
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STATE PROVIDERS
# =============================================================================


class InMemoryKnowledgeGraphProvider(StateProvider):
    """
    In-memory StateProvider for demonstration.

    In production, this would connect to a real database like:
    - PostgreSQL with pgvector
    - Neo4j for graph queries
    - Redis for caching
    - SQLite for local persistence
    """

    def __init__(self) -> None:
        self._storage: dict[str, List[ResearchNode]] = {}
        self.load_count = 0
        self.save_count = 0

    async def load(self, key: str) -> List[ResearchNode]:
        """Load data from storage."""
        self.load_count += 1
        print(f"  📥 [Provider] Loading '{key}' (load #{self.load_count})")
        return self._storage.get(key, [])

    async def save(self, key: str, data: List[ResearchNode]) -> None:
        """Save data to storage."""
        self.save_count += 1
        print(f"  📤 [Provider] Saving '{key}' with {len(data)} nodes (save #{self.save_count})")
        self._storage[key] = data

    def seed_data(self, key: str, nodes: List[ResearchNode]) -> None:
        """Helper to pre-populate data for demo."""
        self._storage[key] = nodes


class ConversationHistoryProvider(StateProvider):
    """Provider for managing conversation history."""

    def __init__(self) -> None:
        self._history: dict[str, List[dict[str, str]]] = {}

    async def load(self, key: str) -> List[dict[str, str]]:
        print(f"  📥 [History] Loading conversation '{key}'")
        return self._history.get(key, [])

    async def save(self, key: str, data: List[dict[str, str]]) -> None:
        print(f"  📤 [History] Saving {len(data)} messages to '{key}'")
        self._history[key] = data


# =============================================================================
# CONTEXT MODEL
# =============================================================================


class ResearchContext(PipelineContext):
    """
    Pipeline context with ContextReference fields.

    The ContextReference fields are lightweight pointers that get
    hydrated from StateProviders when the step runs, and persisted
    back after the step completes.
    """

    # Reference to research knowledge graph
    knowledge_graph: ContextReference[List[ResearchNode]] = Field(
        default_factory=lambda: ContextReference(provider_id="kg_provider", key="research_nodes")
    )

    # Reference to conversation history
    conversation: ContextReference[List[dict[str, str]]] = Field(
        default_factory=lambda: ContextReference(provider_id="history_provider", key="session_001")
    )


# =============================================================================
# PIPELINE STEPS
# =============================================================================


@step
async def search_knowledge(query: str, *, context: ResearchContext) -> str:
    """Search the knowledge graph for relevant nodes."""
    print(f"\n🔍 Searching for: '{query}'")

    # Access the hydrated knowledge graph
    nodes = context.knowledge_graph.get()

    # Simple search (in production, use vector similarity)
    matches = [
        n for n in nodes if query.lower() in n.topic.lower() or query.lower() in n.assertion.lower()
    ]

    if matches:
        results = "\n".join(
            f"  - [{n.topic}] {n.assertion} (confidence: {n.confidence:.0%})" for n in matches
        )
        return f"Found {len(matches)} relevant nodes:\n{results}"
    return f"No nodes found for '{query}'"


@step
async def add_knowledge(data: str, *, context: ResearchContext) -> str:
    """Add a new node to the knowledge graph."""
    print(f"\n➕ Adding knowledge: '{data[:50]}...'")

    # Parse the input (simple format: "topic: assertion")
    if ":" in data:
        topic, assertion = data.split(":", 1)
    else:
        topic, assertion = "General", data

    # Get current graph
    nodes = context.knowledge_graph.get()

    # Create new node
    new_node = ResearchNode(
        topic=topic.strip(),
        assertion=assertion.strip(),
        confidence=0.7,  # Default confidence for new assertions
    )

    # Add to graph
    nodes.append(new_node)

    # Update the reference (will be persisted after step)
    context.knowledge_graph.set(nodes)

    return f"Added node: [{new_node.topic}] {new_node.assertion}"


@step
async def log_interaction(data: str, *, context: ResearchContext) -> str:
    """Log the interaction to conversation history."""
    print("\n📝 Logging interaction")

    # Get current history
    history = context.conversation.get()

    # Add this interaction
    history.append(
        {"role": "user", "content": data, "timestamp": datetime.now(timezone.utc).isoformat()}
    )

    # Update reference
    context.conversation.set(history)

    return data  # Pass through


@step
async def summarize_session(data: str, *, context: ResearchContext) -> str:
    """Summarize the current session state."""
    print("\n📊 Generating session summary")

    nodes = context.knowledge_graph.get()
    history = context.conversation.get()

    return (
        f"Session Summary:\n"
        f"  - Knowledge nodes: {len(nodes)}\n"
        f"  - Conversation turns: {len(history)}\n"
        f"  - Last action: {data}"
    )


# =============================================================================
# DEMO RUNNERS
# =============================================================================


async def demo_basic_usage() -> None:
    """Demonstrate basic StateProvider usage."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic StateProvider Usage")
    print("=" * 60)

    # Create providers
    kg_provider = InMemoryKnowledgeGraphProvider()

    # Seed with initial data
    kg_provider.seed_data(
        "research_nodes",
        [
            ResearchNode(
                topic="AI", assertion="Neural networks can learn patterns", confidence=0.95
            ),
            ResearchNode(
                topic="AI", assertion="Transformers excel at sequence modeling", confidence=0.9
            ),
            ResearchNode(
                topic="Biology", assertion="DNA contains genetic instructions", confidence=0.99
            ),
        ],
    )

    # Create pipeline
    pipeline = search_knowledge

    # Create runner with state_providers
    runner = Flujo(
        pipeline=pipeline,
        context_model=ResearchContext,
        state_providers={
            "kg_provider": kg_provider,
            "history_provider": ConversationHistoryProvider(),
        },
    )

    # Run search
    async with runner:
        result = await runner.run_async("AI")

    print(f"\n✅ Result:\n{result.output}")
    print(f"\n📈 Provider stats: {kg_provider.load_count} loads, {kg_provider.save_count} saves")


async def demo_multi_step_pipeline() -> None:
    """Demonstrate a multi-step pipeline with state providers."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Step Pipeline with State Persistence")
    print("=" * 60)

    # Create providers
    kg_provider = InMemoryKnowledgeGraphProvider()
    history_provider = ConversationHistoryProvider()

    # Seed with initial data
    kg_provider.seed_data(
        "research_nodes",
        [
            ResearchNode(
                topic="Machine Learning",
                assertion="Gradient descent optimizes loss",
                confidence=0.92,
            ),
        ],
    )

    # Create multi-step pipeline
    pipeline = log_interaction >> add_knowledge >> search_knowledge >> summarize_session

    runner = Flujo(
        pipeline=pipeline,
        context_model=ResearchContext,
        state_providers={"kg_provider": kg_provider, "history_provider": history_provider},
    )

    # Run pipeline
    async with runner:
        result = await runner.run_async(
            "Machine Learning: Attention mechanisms improve context understanding"
        )

    print(f"\n✅ Final Output:\n{result.output}")

    # Verify persistence
    print("\n📈 Provider stats:")
    print(f"  - KG: {kg_provider.load_count} loads, {kg_provider.save_count} saves")
    print(f"  - Storage has {len(kg_provider._storage.get('research_nodes', []))} nodes")


async def demo_multiple_runs() -> None:
    """Demonstrate state persistence across multiple runs."""
    print("\n" + "=" * 60)
    print("DEMO 3: State Persistence Across Multiple Runs")
    print("=" * 60)

    # Create providers (shared across runs)
    kg_provider = InMemoryKnowledgeGraphProvider()
    history_provider = ConversationHistoryProvider()

    # Pipeline for adding knowledge
    add_pipeline = add_knowledge >> summarize_session

    # Create runner
    runner = Flujo(
        pipeline=add_pipeline,
        context_model=ResearchContext,
        state_providers={"kg_provider": kg_provider, "history_provider": history_provider},
    )

    # Run multiple times - state persists!
    async with runner:
        print("\n--- Run 1 ---")
        result1 = await runner.run_async("Physics: E=mc² relates energy and mass")
        print(f"Output: {result1.output}")

        print("\n--- Run 2 ---")
        result2 = await runner.run_async("Chemistry: Water is H2O")
        print(f"Output: {result2.output}")

        print("\n--- Run 3 ---")
        result3 = await runner.run_async("Biology: Cells are the basic unit of life")
        print(f"Output: {result3.output}")

    # Show final state
    print(
        f"\n📊 Final Knowledge Graph has {len(kg_provider._storage.get('research_nodes', []))} nodes:"
    )
    for node in kg_provider._storage.get("research_nodes", []):
        print(f"  - [{node.topic}] {node.assertion}")


async def demo_custom_provider() -> None:
    """Demonstrate creating a custom StateProvider."""
    print("\n" + "=" * 60)
    print("DEMO 4: Custom StateProvider (Simulated Database)")
    print("=" * 60)

    class DatabaseProvider(StateProvider):
        """
        Simulated database provider with transaction logging.

        In production, this would use aiosqlite, asyncpg, etc.
        """

        def __init__(self, name: str) -> None:
            self.name = name
            self._data: dict[str, Any] = {}
            self._transaction_log: List[str] = []

        async def load(self, key: str) -> Any:
            self._transaction_log.append(f"SELECT * FROM {self.name} WHERE key='{key}'")
            await asyncio.sleep(0.01)  # Simulate DB latency
            return self._data.get(key, [])

        async def save(self, key: str, data: Any) -> None:
            self._transaction_log.append(
                f"INSERT INTO {self.name} (key, data) VALUES ('{key}', ...)"
            )
            await asyncio.sleep(0.01)  # Simulate DB latency
            self._data[key] = data

        def show_transaction_log(self) -> None:
            print(f"\n📋 Transaction log for '{self.name}':")
            for i, tx in enumerate(self._transaction_log, 1):
                print(f"  {i}. {tx}")

    # Create custom provider
    db_provider = DatabaseProvider("knowledge_db")

    # Seed data
    db_provider._data["research_nodes"] = [
        ResearchNode(topic="Demo", assertion="StateProviders are powerful", confidence=1.0)
    ]

    pipeline = search_knowledge >> add_knowledge >> summarize_session

    runner = Flujo(
        pipeline=pipeline,
        context_model=ResearchContext,
        state_providers={
            "kg_provider": db_provider,
            "history_provider": ConversationHistoryProvider(),
        },
    )

    async with runner:
        result = await runner.run_async("Demo: This is a test assertion")

    print(f"\n✅ Result:\n{result.output}")
    db_provider.show_transaction_log()


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    """Run all demos."""
    print("🚀 State Providers and ContextReference Demo")
    print("=" * 60)

    await demo_basic_usage()
    await demo_multi_step_pipeline()
    await demo_multiple_runs()
    await demo_custom_provider()

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

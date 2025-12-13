from __future__ import annotations

from typing import Any

from pydantic import Field

from flujo.agents import make_agent_async
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.memory import ScoredMemory
from flujo.domain.models import PipelineContext


class RAGContext(PipelineContext):
    run_id: str | None = None
    summary: str | None = None
    query: str | None = None
    retrieved: list[Any] = Field(default_factory=list)

    async def retrieve(self, query_text: str, limit: int = 3) -> list[ScoredMemory]:
        # Delegate to base context retrieval; keeps example type-safe.
        return await super().retrieve(query_text=query_text, limit=limit)


async def summarize(data: dict[str, Any], context: RAGContext) -> dict[str, Any]:
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Summarize the text concisely.",
    )
    text = data.get("text", "")
    out = await agent.run({"text": text})
    context.summary = out  # optional in-context storage
    return {"summary": out}


async def recall(_data: dict[str, Any], context: RAGContext) -> dict[str, Any]:
    results = await context.retrieve(query_text=context.query or "latest", limit=3)
    context.retrieved = [r.record.payload for r in results]
    return {"retrieved": context.retrieved}


async def answer(data: dict[str, Any], context: RAGContext) -> dict[str, Any]:
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Answer using provided retrieved context; say if insufficient.",
    )
    resp = await agent.run({"question": data.get("question"), "context": context.retrieved})
    return {"answer": resp}


pipeline = Pipeline(
    steps=[
        Step(name="summarize", agent=summarize, output_keys=["summary"]),
        Step(name="recall", agent=recall, input_keys=["summary"]),
        Step(name="answer", agent=answer, input_keys=["question", "retrieved"]),
    ]
)

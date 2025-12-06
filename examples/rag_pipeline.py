from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from flujo.agents import make_agent_async
from flujo.domain.dsl import Pipeline, Step


class RAGContext(BaseModel):
    run_id: str | None = None
    scratchpad: dict[str, Any] = {}
    summary: str | None = None
    query: str | None = None
    retrieved: list[Any] = []


async def summarize(data: dict[str, Any], context: RAGContext) -> dict[str, Any]:
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="Summarize the text concisely.",
    )
    text = data.get("text", "")
    out = await agent.run({"text": text})
    context.summary = out  # optional in-context storage
    return {"summary": out}


async def recall(data: dict[str, Any], context: RAGContext) -> dict[str, Any]:
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


pipeline = Pipeline.from_steps(
    [
        Step(name="summarize", agent=summarize, output_keys=["scratchpad.summary"]),
        Step(name="recall", agent=recall, input_keys=["scratchpad.summary"]),
        Step(name="answer", agent=answer, input_keys=["retrieved"]),
    ]
)


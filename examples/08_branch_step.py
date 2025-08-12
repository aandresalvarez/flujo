"""
Demonstrates using a `ConditionalStep` to route tasks to different sub-pipelines.

This example simulates a system that first classifies a user's query and then
routes it to a specialized agent based on whether the query is about code
or a general question. For more details, see docs/pipeline_branching.md.
"""

import asyncio
from typing import Any, Literal
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.domain.agent_protocol import AsyncAgentProtocol


# Agent wrapper classes for our routing workflow
class ClassifyQueryAgent(AsyncAgentProtocol[str, Literal["code", "qa"]]):
    """Classifies the user's query to determine the correct route."""

    async def run(self, query: str, **kwargs: Any) -> Literal["code", "qa"]:
        print(f"🧐 Classifying query: '{query}'")
        if "function" in query.lower() or "python" in query.lower():
            print("   -> Classified as: 'code'")
            return "code"
        print("   -> Classified as: 'qa'")
        return "qa"


class CodeGenerationAgent(AsyncAgentProtocol[str, str]):
    """A specialized agent for writing code."""

    async def run(self, query: str, **kwargs: Any) -> str:
        print("   -> 🐍 Routing to Code Generation Agent.")
        return f'def solution():\n  """Solves: {query}"""\n  pass'


class GeneralQAAgent(AsyncAgentProtocol[str, str]):
    """A specialized agent for answering general questions."""

    async def run(self, query: str, **kwargs: Any) -> str:
        print("   -> ❓ Routing to General QA Agent.")
        return f"Here is a detailed answer to your question about '{query}'."


# 1. Define the different pipelines for each branch. These are our routes.
code_pipeline = Pipeline.from_step(
    Step.model_validate({"name": "GenerateCode", "agent": CodeGenerationAgent()})
)
qa_pipeline = Pipeline.from_step(
    Step.model_validate({"name": "AnswerQuestion", "agent": GeneralQAAgent()})
)


# 2. Define the `ConditionalStep`. This is our router.
def choose_branch(classification_result: str, ctx: BaseModel | None) -> str:
    return classification_result


branch_step = Step.branch_on(
    name="QueryRouter",
    # The `condition_callable` receives the output of the previous step.
    # Its return value ("code" or "qa") is used as the key to select a branch.
    condition_callable=choose_branch,
    branches={
        "code": code_pipeline,
        "qa": qa_pipeline,
    },
)

# 3. Assemble the full pipeline: first classify, then route.
full_pipeline = (
    Step.model_validate({"name": "ClassifyQuery", "agent": ClassifyQueryAgent()}) >> branch_step
)

runner = Flujo(full_pipeline)


# 4. Run the pipeline with different inputs to see the routing in action.
async def run_and_print(prompt: str):
    print("-" * 60)
    print(f"🚀 Running router pipeline for prompt: '{prompt}'\n")
    result = None
    async for item in runner.run_async(prompt):
        result = item
    final_output = result.step_history[-1].output
    print(f"\n✅ Final Output:\n{final_output}")
    print("-" * 60)


async def main():
    await run_and_print("Write a python function for fibonacci.")
    await run_and_print("What is the capital of France?")


if __name__ == "__main__":
    asyncio.run(main())

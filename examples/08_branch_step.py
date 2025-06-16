"""
Demonstrates using a `ConditionalStep` to route tasks to different sub-pipelines.

This example simulates a system that first classifies a user's query and then
routes it to a specialized agent based on whether the query is about code
or a general question. For more details, see docs/pipeline_branching.md.
"""
import asyncio
from typing import Any, Literal, cast

from flujo import Flujo, Step, Pipeline


# Agents for our routing workflow
async def classify_query_agent(query: str) -> Literal["code", "qa"]:
    """Classifies the user's query to determine the correct route."""
    print(f"🧐 Classifying query: '{query}'")
    if "function" in query.lower() or "python" in query.lower():
        print("   -> Classified as: 'code'")
        return "code"
    print("   -> Classified as: 'qa'")
    return "qa"


async def code_generation_agent(query: str) -> str:
    """A specialized agent for writing code."""
    print("   -> 🐍 Routing to Code Generation Agent.")
    return f'def solution():\n  """Solves: {query}"""\n  pass'


async def general_qa_agent(query: str) -> str:
    """A specialized agent for answering general questions."""
    print("   -> ❓ Routing to General QA Agent.")
    return f"Here is a detailed answer to your question about '{query}'."


# 1. Define the different pipelines for each branch. These are our routes.
code_pipeline = Pipeline.from_step(Step("GenerateCode", cast(Any, code_generation_agent)))
qa_pipeline = Pipeline.from_step(Step("AnswerQuestion", cast(Any, general_qa_agent)))

# 2. Define the `ConditionalStep`. This is our router.
branch_step = Step.branch_on(
    name="QueryRouter",
    # The `condition_callable` receives the output of the previous step.
    # Its return value ("code" or "qa") is used as the key to select a branch.
    condition_callable=lambda classification_result, ctx: classification_result,
    branches={
        "code": code_pipeline,
        "qa": qa_pipeline,
    },
)

# 3. Assemble the full pipeline: first classify, then route.
full_pipeline = Step("ClassifyQuery", cast(Any, classify_query_agent)) >> branch_step

runner = Flujo(full_pipeline)

# 4. Run the pipeline with different inputs to see the routing in action.
async def run_and_print(prompt: str):
    print("-" * 60)
    print(f"🚀 Running router pipeline for prompt: '{prompt}'\n")
    result = await runner.run_async(prompt)
    final_output = result.step_history[-1].output
    print(f"\n✅ Final Output:\n{final_output}")
    print("-" * 60)


async def main():
    await run_and_print("Write a python function for fibonacci.")
    await run_and_print("What is the capital of France?")


if __name__ == "__main__":
    asyncio.run(main())

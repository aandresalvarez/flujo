"""
Demonstrates building a custom pipeline with the Flujo engine and DSL (`Step`).
This example uses a built-in `Plugin` to validate the syntax of generated SQL.
Plugins are a powerful way to add custom, non-LLM logic to your workflows.
For more details, see docs/extending.md.
"""
import asyncio
from typing import Any, cast

from flujo import Flujo, Step
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

# To deterministically demonstrate the plugin, we'll use a `StubAgent` that
# always produces invalid SQL. In a real app, this would be a powerful LLM agent.
sql_agent = StubAgent(["SELEC * FRM users WHERE id = 1;"])  # Intentionally incorrect SQL

# A simple agent for the validation step. The real work is done by the plugin.
validation_agent = StubAgent([None])

# 1. Create a solution step with our SQL-generating agent.
solution_step = Step("GenerateSQL", agent=cast(Any, sql_agent))

# 2. Create a validation step and attach the `SQLSyntaxValidator` plugin.
#    The plugin runs *after* the agent and checks its output. If the plugin
#    finds an issue, it will mark the step as failed and provide feedback.
validation_step = Step(
    "ValidateSQL",
    agent=cast(Any, validation_agent),
    plugins=[SQLSyntaxValidator()],
)

# 3. Compose the steps into a pipeline using the '>>' operator.
sql_pipeline = solution_step >> validation_step

# 4. Create a Flujo runner for our custom pipeline.
runner = Flujo(sql_pipeline)


async def main() -> None:
    print("🧠 Running a custom SQL generation and validation pipeline...")
    result = None
    async for item in runner.run_async("Generate a query to select all users."):
        result = item

    # 5. Inspect the results from the pipeline's `step_history`.
    solution_result = result.step_history[0]
    validation_result = result.step_history[1]

    print(f"\n🔍 SQL Generation Step ('{solution_result.name}'):")
    print(f"  - Success: {solution_result.success}")
    print(f"  - Output: '{solution_result.output}'")

    print(f"\n🔍 SQL Validation Step ('{validation_result.name}'):")
    print(f"  - Success: {validation_result.success}")
    if not validation_result.success:
        print(f"  - Feedback from Plugin: {validation_result.feedback}")

    if not validation_result.success:
        print("\n\n🎉 The SQLSyntaxValidator plugin correctly identified the invalid SQL!")


if __name__ == "__main__":
    asyncio.run(main())

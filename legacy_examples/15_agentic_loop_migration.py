"""Refactoring a LoopStep workflow to AgenticLoop.as_step().

This script reworks the text refinement example from ``07_loop_step.py``
using ``AgenticLoop``. The planner decides whether to run the ``editor``
agent again or finish based on the latest result.
"""

import asyncio
from typing import Any, Dict

from pydantic import BaseModel

from flujo import Flujo, Pipeline, Step
from flujo.recipes import AgenticLoop
from flujo.domain.commands import FinishCommand, RunAgentCommand, AgentCommand
from flujo.models import PipelineContext


class TextEdit(BaseModel):
    text: str
    is_good_enough: bool = False


async def editor_agent(text: str) -> TextEdit:
    """Simple editing agent that lengthens short text."""
    if len(text) < 30:
        text += " It is known for its simple syntax."
    return TextEdit(text=text, is_good_enough=len(text) >= 30)


async def planner_agent(data: Dict[str, Any]) -> AgentCommand:
    """Planner that decides whether to keep editing or finish."""
    last: TextEdit | None = data.get("last_command_result")
    goal: str = data.get("goal", "")
    if last is None or not last.is_good_enough:
        return RunAgentCommand(agent_name="editor", input_data=goal if last is None else last.text)
    return FinishCommand(final_answer=last.text)


loop_step = AgenticLoop(
    planner_agent=planner_agent,
    agent_registry={"editor": editor_agent},
).as_step(name="RefineText")

pipeline = Pipeline.from_step(loop_step)
runner = Flujo(pipeline, context_model=PipelineContext)


async def main() -> None:
    print("🚀 Running AgenticLoop refinement...\n")
    result = None
    async for item in runner.run_async("Python is a language."):
        result = item
    assert result is not None
    final_ctx = result.final_pipeline_context
    assert final_ctx is not None
    print("\n✅ Final answer:", final_ctx.command_log[-1].execution_result)


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

from typing import Any, Literal, Union

from flujo.domain.base_model import BaseModel
from pydantic import Field


class RunAgentCommand(BaseModel):
    """Instructs the loop to run a registered sub-agent."""

    type: Literal["run_agent"] = "run_agent"
    agent_name: str = Field(..., description="The name of the agent to run from the registry.")
    input_data: Any = Field(..., description="The input data to pass to the sub-agent.")


class AskHumanCommand(BaseModel):
    """Pause execution and ask a human for input."""

    type: Literal["ask_human"] = "ask_human"
    question: str = Field(..., description="The question to present to the human user.")


class FinishCommand(BaseModel):
    """Finish the loop with a final answer."""

    type: Literal["finish"] = "finish"
    final_answer: Any = Field(..., description="The final result or summary of the task.")


AgentCommand = Union[RunAgentCommand, AskHumanCommand, FinishCommand]

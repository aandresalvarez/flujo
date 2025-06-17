from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class RunAgentCommand(BaseModel):
    """Instructs the loop to run a registered sub-agent."""

    type: Literal["run_agent"] = "run_agent"
    agent_name: str = Field(..., description="The name of the agent to run from the registry.")
    input_data: Any = Field(..., description="The input data to pass to the sub-agent.")


class RunPythonCodeCommand(BaseModel):
    """Execute a snippet of Python code. Requires a secure sandbox."""

    type: Literal["run_python"] = "run_python"
    code: str = Field(..., description="The Python code to execute.")
    # Result is expected in variable 'result'


class AskHumanCommand(BaseModel):
    """Pause execution and ask a human for input."""

    type: Literal["ask_human"] = "ask_human"
    question: str = Field(..., description="The question to present to the human user.")


class FinishCommand(BaseModel):
    """Finish the loop with a final answer."""

    type: Literal["finish"] = "finish"
    final_answer: Any = Field(..., description="The final result or summary of the task.")


AgentCommand = Union[RunAgentCommand, RunPythonCodeCommand, AskHumanCommand, FinishCommand]


class ExecutedCommandLog(BaseModel):
    """Structured log entry for a command executed in the loop."""

    turn: int
    generated_command: AgentCommand
    execution_result: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}

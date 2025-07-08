"""Pipeline factory functions for creating standard workflow patterns.

This module provides factory functions that return standard Pipeline objects,
making workflows transparent, inspectable, and composable. These factories
replace the class-based recipe approach to enable better serialization,
visualization, and AI-driven modification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast, TYPE_CHECKING

from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import Step
from ..domain.models import PipelineContext, Task, Candidate, Checklist
from ..domain.scoring import ratio_score
from ..domain.commands import AgentCommand, FinishCommand, ExecutedCommandLog
from ..application.runner import Flujo
from ..testing.utils import gather_result

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..infra.agents import AsyncAgentProtocol


def make_default_pipeline(
    review_agent: "AsyncAgentProtocol[Any, Any]",
    solution_agent: "AsyncAgentProtocol[Any, Any]",
    validator_agent: "AsyncAgentProtocol[Any, Any]",
    reflection_agent: Optional["AsyncAgentProtocol[Any, Any]"] = None,
    max_retries: int = 3,
) -> Pipeline[str, Checklist]:
    """Create a default Review → Solution → Validate pipeline.

    Args:
        review_agent: Agent that creates a checklist of requirements
        solution_agent: Agent that generates a solution
        validator_agent: Agent that validates the solution against requirements
        reflection_agent: Optional agent for reflection/improvement
        max_retries: Maximum retries for each step

    Returns:
        A Pipeline object that can be inspected, composed, and executed
    """

    async def review_step(data: str, *, context: PipelineContext) -> str:
        """Review the task and create a checklist."""
        result = await _invoke(review_agent, data, context=context)
        checklist = cast(Checklist, getattr(result, "output", result))
        context.scratchpad["checklist"] = checklist
        return data

    async def solution_step(data: str, *, context: PipelineContext) -> str:
        """Generate a solution based on the task."""
        result = await _invoke(solution_agent, data, context=context)
        solution = cast(str, getattr(result, "output", result))
        context.scratchpad["solution"] = solution
        return solution

    async def validate_step(_data: Any, *, context: PipelineContext) -> Checklist:
        """Validate the solution against the checklist."""
        payload = {
            "solution": context.scratchpad.get("solution", ""),
            "checklist": context.scratchpad.get("checklist", Checklist(items=[])),
        }
        result = await _invoke(validator_agent, payload, context=context)
        return cast(Checklist, getattr(result, "output", result))

    # Create steps with configuration
    review_step_s = Step.from_callable(review_step, max_retries=max_retries)
    solution_step_s = Step.from_callable(solution_step, max_retries=max_retries)
    validate_step_s = Step.from_callable(validate_step, max_retries=max_retries)

    # Compose the pipeline
    pipeline = review_step_s >> solution_step_s >> validate_step_s

    # Add reflection if provided
    if reflection_agent:

        async def reflection_step(data: Any, *, context: PipelineContext) -> Any:
            """Reflect on the solution and suggest improvements."""
            result = await _invoke(reflection_agent, data, context=context)
            return getattr(result, "output", result)

        reflection_step_s = Step.from_callable(reflection_step, max_retries=max_retries)
        pipeline = pipeline >> reflection_step_s

    return pipeline


def make_agentic_loop_pipeline(
    planner_agent: "AsyncAgentProtocol[Any, Any]",
    agent_registry: Dict[str, "AsyncAgentProtocol[Any, Any]"],
    max_loops: int = 10,
    max_retries: int = 3,
) -> Pipeline[str, Any]:
    """Create an agentic loop pipeline for explorative workflows.

    Args:
        planner_agent: Agent that decides what command to run next
        agent_registry: Dictionary of available agents to execute
        max_loops: Maximum number of loop iterations
        max_retries: Maximum retries for each step

    Returns:
        A Pipeline object with a LoopStep containing the agentic logic
    """

    class _CommandExecutor:
        """Internal class to execute commands from the planner."""

        def __init__(self, agent_registry: Dict[str, "AsyncAgentProtocol[Any, Any]"]):
            self.agent_registry = agent_registry

        async def run(self, data: Any, *, context: PipelineContext) -> ExecutedCommandLog:
            """Execute a command from the planner."""
            command = cast(AgentCommand, data)
            turn: int = len(getattr(context, "command_log", [])) + 1 if context is not None else 1
            if isinstance(command, FinishCommand):
                return ExecutedCommandLog(
                    turn=turn,
                    generated_command=command,
                    execution_result=getattr(command, "final_answer", None),
                )
            # Handle other command types as needed
            return ExecutedCommandLog(
                turn=turn,
                generated_command=command,
                execution_result="Command executed",
            )

    async def planner_step(data: str, *, context: PipelineContext) -> AgentCommand:
        """Get the next command from the planner."""
        result = await planner_agent.run(data, context=context)
        return cast(AgentCommand, getattr(result, "output", result))

    async def command_executor_step(
        data: AgentCommand, *, context: PipelineContext
    ) -> ExecutedCommandLog:
        executor: _CommandExecutor = _CommandExecutor(agent_registry)
        return await executor.run(data, context=context)

    # Create the loop body pipeline
    planner_step_s: Step[Any, Any] = Step.from_callable(planner_step, max_retries=max_retries)
    executor_step_s: Step[Any, Any] = Step.from_callable(
        command_executor_step, max_retries=max_retries
    )
    loop_body: Pipeline[Any, Any] = planner_step_s >> executor_step_s

    # Create the loop step with proper config
    def exit_condition(output: Any, context: Any) -> bool:
        return isinstance(output, ExecutedCommandLog) and isinstance(
            getattr(output, "generated_command", None), FinishCommand
        )

    loop_step: Any = Step.loop_until(
        name="AgenticExplorationLoop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=max_loops,
    )
    loop_step.config.max_retries = max_retries

    return Pipeline.from_step(loop_step)


async def run_default_pipeline(
    pipeline: Pipeline[str, Checklist],
    task: Task,
) -> Optional[Candidate]:
    """Run a default pipeline and return the result.

    Args:
        pipeline: Pipeline created by make_default_pipeline
        task: Task to execute

    Returns:
        Candidate with solution and checklist, or None if failed
    """
    runner: Flujo[str, Checklist, PipelineContext] = Flujo(pipeline)
    result = await gather_result(runner, task.prompt)

    # Extract solution and checklist from context
    context = result.final_pipeline_context
    solution = context.scratchpad.get("solution")
    checklist = context.scratchpad.get("checklist")

    if solution is None or checklist is None:
        return None

    # Calculate score
    score = ratio_score(checklist)

    return Candidate(
        solution=solution,
        checklist=checklist,
        score=score,
    )


async def run_agentic_loop_pipeline(
    pipeline: Pipeline[str, Any],
    initial_goal: str,
) -> Any:
    """Run an agentic loop pipeline and return the result.

    Args:
        pipeline: Pipeline created by make_agentic_loop_pipeline
        initial_goal: Initial goal for the agentic loop

    Returns:
        Final result from the agentic loop
    """
    runner: Flujo[str, Any, PipelineContext] = Flujo(pipeline)
    result = await gather_result(runner, initial_goal)
    output = result.step_history[-1].output
    if hasattr(output, "execution_result"):
        return output.execution_result
    return output


async def _invoke(
    agent: "AsyncAgentProtocol[Any, Any]",
    data: Any,
    *,
    context: Optional[PipelineContext] = None,
) -> Any:
    """Helper function to invoke an agent with proper error handling."""
    try:
        if context is not None:
            return await agent.run(data, context=context)
        else:
            return await agent.run(data)
    except Exception as e:
        # Handle specific exceptions as needed
        raise e

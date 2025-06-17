from __future__ import annotations

from typing import Any, Dict
import asyncio

from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.commands import (
    AgentCommand,
    FinishCommand,
    ExecutedCommandLog,
)
from ..domain.models import PipelineResult, PipelineContext
from ..domain.pipeline_dsl import Step, LoopStep
from ..application.flujo_engine import Flujo


class AgenticLoop:
    """High-level recipe for explorative agentic workflows."""

    def __init__(
        self,
        planner_agent: AsyncAgentProtocol[Any, AgentCommand],
        agent_registry: Dict[str, AsyncAgentProtocol],
        max_loops: int = 15,
    ) -> None:
        self.planner_agent = planner_agent
        self.agent_registry = agent_registry
        self.max_loops = max_loops
        self._pipeline = self._build_internal_pipeline()

    def _build_internal_pipeline(self) -> LoopStep:
        executor_step = Step("ExecuteCommand", _CommandExecutor(self.agent_registry))
        loop_body = Step("DecideNextCommand", self.planner_agent) >> executor_step

        def exit_condition(_: Any, context: PipelineContext) -> bool:
            last_cmd = context.command_log[-1].generated_command
            return isinstance(last_cmd, FinishCommand)

        return Step.loop_until(
            name="AgenticExplorationLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=exit_condition,
            max_loops=self.max_loops,
            iteration_input_mapper=lambda result, ctx, i: {"last_command_result": result},
        )

    def run(self, initial_goal: str) -> PipelineResult:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        return runner.run(
            {"last_command_result": None, "goal": initial_goal},
            initial_context_data={"initial_prompt": initial_goal},
        )

    async def run_async(self, initial_goal: str) -> PipelineResult:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        result: PipelineResult | None = None
        async for item in runner.run_async(
            {"last_command_result": None, "goal": initial_goal},
            initial_context_data={"initial_prompt": initial_goal},
        ):
            result = item
        assert result is not None
        return result

    def resume(self, paused_result: PipelineResult, human_input: Any) -> PipelineResult:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        async def _consume() -> PipelineResult:
            return await runner.resume_async(paused_result, human_input)
        return asyncio.run(_consume())

    async def resume_async(self, paused_result: PipelineResult, human_input: Any) -> PipelineResult:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        return await runner.resume_async(paused_result, human_input)


class _CommandExecutor:
    def __init__(self, agent_registry: Dict[str, AsyncAgentProtocol]):
        self.agent_registry = agent_registry

    async def run(self, command: AgentCommand, *, pipeline_context: PipelineContext) -> Any:
        turn = len(pipeline_context.command_log) + 1
        result: Any = "Command type not recognized."
        try:
            if command.type == "run_agent":
                agent = self.agent_registry.get(command.agent_name)
                if not agent:
                    result = f"Error: Agent '{command.agent_name}' not found."
                else:
                    result = await agent.run(command.input_data)
            elif command.type == "run_python":
                local_scope: Dict[str, Any] = {}
                exec(command.code, globals(), local_scope)
                result = local_scope.get("result", "Python code executed successfully.")
            elif command.type == "ask_human":
                from ..exceptions import PausedException
                if isinstance(pipeline_context, PipelineContext):
                    pipeline_context.scratchpad["paused_step_input"] = command
                raise PausedException(message=command.question)
            elif command.type == "finish":
                result = command.final_answer
        except PausedException:
            raise
        except Exception as e:  # noqa: BLE001
            result = f"Error during command execution: {e}"

        log_entry = ExecutedCommandLog(
            turn=turn,
            generated_command=command,
            execution_result=result,
        )
        pipeline_context.command_log.append(log_entry)
        return result

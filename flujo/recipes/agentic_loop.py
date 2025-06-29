from __future__ import annotations

from typing import Any, Dict

from ..domain.resources import AppResources
import ast
import asyncio
from pydantic import TypeAdapter, ValidationError

from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.commands import (
    AgentCommand,
    FinishCommand,
    ExecutedCommandLog,
)
from ..exceptions import PausedException
from ..domain.models import PipelineResult, PipelineContext
from ..domain.pipeline_dsl import Step, LoopStep
from ..application.flujo_engine import Flujo, _accepts_param

_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)


class AgenticLoop:
    """High-level recipe for explorative agentic workflows."""

    def __init__(
        self,
        planner_agent: AsyncAgentProtocol[Any, AgentCommand],
        agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]],
        max_loops: int = 15,
    ) -> None:
        self.planner_agent = planner_agent
        self.agent_registry = agent_registry
        self.max_loops = max_loops
        self._pipeline = self._build_internal_pipeline()

    def _build_internal_pipeline(self) -> LoopStep[PipelineContext]:
        executor_step: Step[Any, Any] = Step(
            "ExecuteCommand", _CommandExecutor(self.agent_registry)
        )
        loop_body = Step("DecideNextCommand", self.planner_agent) >> executor_step

        def exit_condition(_: Any, context: PipelineContext | None) -> bool:
            if not context or not context.command_log:
                return False
            last_cmd = context.command_log[-1].generated_command
            return isinstance(last_cmd, FinishCommand)

        return Step.loop_until(
            name="AgenticExplorationLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=exit_condition,
            max_loops=self.max_loops,
            iteration_input_mapper=lambda result, ctx, i: {
                "last_command_result": result,
                "goal": ctx.initial_prompt if ctx else "",
            },
        )

    def run(self, initial_goal: str) -> PipelineResult[PipelineContext]:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        return runner.run(
            {"last_command_result": None, "goal": initial_goal},
            initial_context_data={"initial_prompt": initial_goal},
        )

    async def run_async(self, initial_goal: str) -> PipelineResult[PipelineContext]:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        final_result: PipelineResult[PipelineContext] | None = None
        async for item in runner.run_async(
            {"last_command_result": None, "goal": initial_goal},
            initial_context_data={"initial_prompt": initial_goal},
        ):
            final_result = item
        assert final_result is not None
        return final_result

    def resume(
        self, paused_result: PipelineResult[PipelineContext], human_input: Any
    ) -> PipelineResult[PipelineContext]:
        runner = Flujo(self._pipeline, context_model=PipelineContext)

        async def _consume() -> PipelineResult[PipelineContext]:
            return await runner.resume_async(paused_result, human_input)

        return asyncio.run(_consume())

    async def resume_async(
        self, paused_result: PipelineResult[PipelineContext], human_input: Any
    ) -> PipelineResult[PipelineContext]:
        runner = Flujo(self._pipeline, context_model=PipelineContext)
        return await runner.resume_async(paused_result, human_input)

    def as_step(self, name: str, **kwargs: Any) -> Step[str, PipelineResult[PipelineContext]]:
        """Return this loop as a composable :class:`Step`.

        Parameters
        ----------
        name:
            Name of the resulting step.
        **kwargs:
            Additional ``Step`` configuration such as ``timeout_s`` or
            ``max_retries``.

        Returns
        -------
        Step
            A step that executes :meth:`run_async` when invoked.
        """

        async def _runner(
            initial_goal: str,
            *,
            context: PipelineContext | None = None,
            resources: AppResources | None = None,
        ) -> PipelineResult[PipelineContext]:
            runner = Flujo(
                self._pipeline,
                context_model=PipelineContext,
                resources=resources,
            )
            final_result: PipelineResult[PipelineContext] | None = None
            async for item in runner.run_async(
                {"last_command_result": None, "goal": initial_goal},
                initial_context_data={"initial_prompt": initial_goal},
            ):
                final_result = item
            if final_result is None:
                raise ValueError("The final result of the pipeline execution is None. Ensure the pipeline produces a valid result.")
            if context is not None:
                context.__dict__.update(final_result.final_pipeline_context.__dict__)
            return final_result

        return Step.from_callable(_runner, name=name, **kwargs)


class _CommandExecutor:
    def __init__(self, agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]]):
        self.agent_registry = agent_registry

    async def run(
        self,
        data: Any,
        *,
        context: PipelineContext | None = None,
        resources: AppResources | None = None,
        **kwargs: Any,
    ) -> Any:
        if context is None:
            raise ValueError("context must be a PipelineContext instance")
        return await self._run_command(data, context=context, resources=resources)

    async def run_async(
        self,
        data: Any,
        *,
        context: PipelineContext | None = None,
        resources: AppResources | None = None,
        **kwargs: Any,
    ) -> Any:
        if context is None:
            raise ValueError("context must be a PipelineContext instance")
        return await self._run_command(data, context=context, resources=resources)

    async def _run_command(
        self,
        data: Any,
        *,
        context: PipelineContext,
        resources: AppResources | None = None,
    ) -> Any:
        pipeline_context = context
        turn = len(pipeline_context.command_log) + 1
        try:
            cmd = _command_adapter.validate_python(data)
        except ValidationError as e:  # pragma: no cover - planner bug
            validation_error_result = f"Invalid command: {e}"
            pipeline_context.command_log.append(
                ExecutedCommandLog(
                    turn=turn,
                    generated_command=data,
                    execution_result=validation_error_result,
                )
            )
            return validation_error_result

        exec_result: Any = "Command type not recognized."
        try:
            if cmd.type == "run_agent":
                agent = self.agent_registry.get(cmd.agent_name)
                if not agent:
                    exec_result = f"Error: Agent '{cmd.agent_name}' not found."
                else:
                    agent_kwargs: Dict[str, Any] = {}
                    if _accepts_param(agent.run, "pipeline_context"):
                        agent_kwargs["pipeline_context"] = pipeline_context
                    if resources is not None and _accepts_param(agent.run, "resources"):
                        agent_kwargs["resources"] = resources
                    exec_result = await agent.run(cmd.input_data, **agent_kwargs)
            elif cmd.type == "run_python":
                tree = ast.parse(cmd.code, mode="exec")
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        raise ValueError("Imports are not allowed in run_python")
                local_scope: Dict[str, Any] = {}
                # WARNING: The following use of exec is sandboxed with empty __builtins__ for security.
                # Only trusted code should be executed here. Review all inputs to this exec carefully.
                exec(
                    compile(tree, filename="<agentic_loop>", mode="exec"),
                    {"__builtins__": {}},
                    local_scope,
                )
                exec_result = local_scope.get("result", "Python code executed successfully.")
            elif cmd.type == "ask_human":
                if isinstance(pipeline_context, PipelineContext):
                    pipeline_context.scratchpad["paused_step_input"] = cmd
                raise PausedException(message=cmd.question)
            elif cmd.type == "finish":
                exec_result = cmd.final_answer
        except PausedException:
            raise
        except Exception as e:  # noqa: BLE001
            exec_result = f"Error during command execution: {e}"

        pipeline_context.command_log.append(
            ExecutedCommandLog(
                turn=turn,
                generated_command=cmd,
                execution_result=exec_result,
            )
        )
        return exec_result

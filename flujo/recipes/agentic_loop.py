from __future__ import annotations

from typing import Any, Dict

from ..domain.resources import AppResources
import asyncio
from pydantic import TypeAdapter, ValidationError

from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.commands import (
    AgentCommand,
    FinishCommand,
    ExecutedCommandLog,
)
from ..exceptions import (
    PausedException,
    PipelineContextInitializationError,
    ContextInheritanceError,
)
from ..domain.models import PipelineResult, PipelineContext
from ..domain.dsl.step import Step
from ..domain.dsl.loop import LoopStep
from ..domain.dsl.pipeline import Pipeline
from ..application.runner import Flujo, _accepts_param, _extract_missing_fields

import warnings

_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)


class AgenticLoop:
    """AgenticLoop recipe for explorative agent workflows.

    DEPRECATED: This class-based approach is deprecated. Use the new factory functions
    for better transparency, composability, and future YAML/AI support:

    - Use `make_agentic_loop_pipeline()` to create a Pipeline object
    - Use `run_agentic_loop_pipeline()` to execute the pipeline

    See `flujo.recipes.factories` for the new approach.
    """

    def __init__(
        self,
        planner_agent: AsyncAgentProtocol[Any, Any],
        agent_registry: Dict[str, AsyncAgentProtocol[Any, Any]],
        max_loops: int = 10,
        max_retries: int = 3,
    ):
        warnings.warn(
            "The AgenticLoop class is deprecated. Use make_agentic_loop_pipeline() and run_agentic_loop_pipeline() "
            "from flujo.recipes.factories for better transparency, composability, and future YAML/AI support.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.planner_agent = planner_agent
        self.agent_registry = agent_registry
        self.max_loops = max_loops
        self.max_retries = max_retries
        self._pipeline = self._build_internal_pipeline()

    def _build_internal_pipeline(self) -> LoopStep[PipelineContext]:
        executor_step: Step[Any, Any] = Step.model_validate(
            {"name": "ExecuteCommand", "agent": _CommandExecutor(self.agent_registry)}
        )
        loop_body: Pipeline[Any, Any] = (
            Step.model_validate({"name": "DecideNextCommand", "agent": self.planner_agent})
            >> executor_step
        )

        def exit_condition(log: ExecutedCommandLog, _context: PipelineContext | None) -> bool:
            return isinstance(log.generated_command, FinishCommand)

        def _iter_mapper(
            log: ExecutedCommandLog, ctx: PipelineContext | None, _i: int
        ) -> dict[str, Any]:
            if ctx is not None:
                ctx.command_log.append(log)
                goal = ctx.initial_prompt
            else:
                goal = ""
            return {"last_command_result": log.execution_result, "goal": goal}

        def _output_mapper(log: ExecutedCommandLog, ctx: PipelineContext | None) -> Any:
            if ctx is not None:
                ctx.command_log.append(log)
                return ctx.command_log[-1].execution_result
            return log.execution_result

        return Step.loop_until(
            name="AgenticExplorationLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=exit_condition,
            max_loops=self.max_loops,
            iteration_input_mapper=_iter_mapper,
            loop_output_mapper=_output_mapper,
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

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> Step[str, PipelineResult[PipelineContext]]:
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
            init_ctx_data: Dict[str, Any] = {}
            if inherit_context and context is not None:
                init_ctx_data = context.model_dump()
            if "initial_prompt" not in init_ctx_data:
                init_ctx_data["initial_prompt"] = initial_goal

            try:
                runner = Flujo(
                    self._pipeline,
                    context_model=PipelineContext,
                    resources=resources,
                    initial_context_data=init_ctx_data,
                )
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=(list(context.model_dump().keys()) if context else []),
                    child_model_name=PipelineContext.__name__,
                ) from e

            final_result: PipelineResult[PipelineContext] | None = None
            try:
                async for item in runner.run_async(
                    {"last_command_result": None, "goal": initial_goal}
                ):
                    final_result = item
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=(list(context.model_dump().keys()) if context else []),
                    child_model_name=PipelineContext.__name__,
                ) from e
            if final_result is None:
                raise ValueError(
                    "The final result of the pipeline execution is None. Ensure the pipeline produces a valid result."
                )
            if (
                inherit_context
                and context is not None
                and final_result.final_pipeline_context is not None
            ):
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
        # Use context for clarity in internal logic
        context_obj = context
        turn = len(context_obj.command_log) + 1
        try:
            cmd = _command_adapter.validate_python(data)
        except ValidationError as e:  # pragma: no cover - planner bug
            validation_error_result = f"Invalid command: {e}"
            return ExecutedCommandLog(
                turn=turn,
                generated_command=data,
                execution_result=validation_error_result,
            )

        exec_result: Any = "Command type not recognized."
        try:
            if cmd.type == "run_agent":
                agent = self.agent_registry.get(cmd.agent_name)
                if not agent:
                    exec_result = f"Error: Agent '{cmd.agent_name}' not found."
                else:
                    agent_kwargs: Dict[str, Any] = {}
                    if _accepts_param(agent.run, "context"):
                        agent_kwargs["context"] = context_obj
                    if resources is not None and _accepts_param(agent.run, "resources"):
                        agent_kwargs["resources"] = resources
                    exec_result = await agent.run(cmd.input_data, **agent_kwargs)
            elif cmd.type == "ask_human":
                if isinstance(context_obj, PipelineContext):
                    context_obj.scratchpad["paused_step_input"] = cmd
                raise PausedException(message=cmd.question)
            elif cmd.type == "finish":
                exec_result = cmd.final_answer
        except PausedException:
            raise
        except Exception as e:  # noqa: BLE001
            exec_result = f"Error during command execution: {e}"
        log_entry = ExecutedCommandLog(
            turn=turn,
            generated_command=cmd,
            execution_result=exec_result,
        )
        context_obj.command_log.append(log_entry)
        return log_entry

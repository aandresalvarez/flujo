"""Pipeline factory functions for creating standard workflow patterns.

This module provides factory functions that return standard Pipeline objects,
making workflows transparent, inspectable, and composable. These factories
replace the class-based recipe approach to enable better serialization,
visualization, and AI-driven modification.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING
from pydantic import TypeAdapter

from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import Step
from ..domain.models import (
    PipelineContext,
    Task,
    Candidate,
    Checklist,
    ExecutedCommandLog,
)
from ..domain.commands import AgentCommand, FinishCommand
from ..application.runner import Flujo
from ..testing.utils import gather_result
from flujo.domain.models import PipelineResult
from ..domain.scoring import ratio_score

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..agents import AsyncAgentProtocol

# Type adapter for command validation
_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)
_checklist_adapter: TypeAdapter[Checklist] = TypeAdapter(Checklist)


def _extract_output_value(result: object) -> object:
    """Return `result.output` when present, else `result`."""
    return getattr(result, "output", result)


def _coerce_checklist(value: object) -> Checklist:
    if isinstance(value, Checklist):
        return value
    return _checklist_adapter.validate_python(value)


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    raise TypeError(f"Expected a string output, got {type(value).__name__}")


def make_default_pipeline(
    review_agent: "AsyncAgentProtocol[Any, Any]",
    solution_agent: "AsyncAgentProtocol[Any, Any]",
    validator_agent: "AsyncAgentProtocol[Any, Any]",
    reflection_agent: Optional["AsyncAgentProtocol[Any, Any]"] = None,
    max_retries: int = 3,
    k_variants: int = 1,
    max_iters: int = 3,
    reflection_limit: Optional[int] = None,
) -> Pipeline[str, Checklist]:
    """Create a default Review → Solution → Validate pipeline.

    Args:
        review_agent: Agent that creates a checklist of requirements
        solution_agent: Agent that generates a solution
        validator_agent: Agent that validates the solution against requirements
        reflection_agent: Optional agent for reflection/improvement
        max_retries: Maximum retries for each step
        k_variants: Number of solution variants to generate per iteration
        max_iters: Maximum number of iterations for improvement
        reflection_limit: Optional limit on reflection iterations

    Returns:
        A Pipeline object that can be inspected, composed, and executed
    """

    # --- Robust type validation for critical parameters ---
    if not isinstance(max_retries, int):
        raise TypeError(f"max_retries must be int, got {type(max_retries).__name__}")
    if not isinstance(k_variants, int):
        raise TypeError(f"k_variants must be int, got {type(k_variants).__name__}")
    if not isinstance(max_iters, int):
        raise TypeError(f"max_iters must be int, got {type(max_iters).__name__}")
    if reflection_limit is not None and not isinstance(reflection_limit, int):
        raise TypeError(
            f"reflection_limit must be int or None, got {type(reflection_limit).__name__}"
        )
    # -----------------------------------------------------

    async def review_step(data: str, *, context: PipelineContext) -> str:
        """Review the task and create a checklist."""
        result = await _invoke(review_agent, data, context=context)
        checklist = _coerce_checklist(_extract_output_value(result))
        context.checklist = checklist
        return data

    async def solution_step(data: str, *, context: PipelineContext) -> str:
        """Generate a solution based on the task."""
        result = await _invoke(solution_agent, data, context=context)
        solution = _coerce_text(_extract_output_value(result))
        context.solution = solution
        return solution

    async def validate_step(_data: Any, *, context: PipelineContext) -> Checklist:
        """Validate the solution against the checklist."""
        payload = {
            "solution": getattr(context, "solution", "") or "",
            "checklist": getattr(context, "checklist", None) or Checklist(items=[]),
        }
        result = await _invoke(validator_agent, payload, context=context)
        return _coerce_checklist(_extract_output_value(result))

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


def make_state_machine_pipeline(
    *,
    nodes: dict[str, Step[Any, Any] | Pipeline[Any, Any]],
    context_model: type[PipelineContext],
    router_field: str = "next_state",
    end_state_field: str = "is_complete",
    max_loops: int = 50,
) -> Pipeline[Any, Any]:
    """Create a simple state machine pipeline.

    Each iteration runs the state pipeline specified in ``context.<router_field>``.
    The loop exits when ``context.<end_state_field>`` evaluates to ``True``.
    """

    if router_field not in getattr(context_model, "model_fields", {}):
        raise AttributeError(
            f"{context_model.__name__!s} is missing required field {router_field!r}"
        )
    if end_state_field not in getattr(context_model, "model_fields", {}):
        raise AttributeError(
            f"{context_model.__name__!s} is missing required field {end_state_field!r}"
        )

    normalized: dict[str, Pipeline[Any, Any]] = {}
    for key, val in nodes.items():
        normalized[key] = Pipeline.from_step(val) if isinstance(val, Step) else val

    def _route_state(_last: Any, ctx: PipelineContext | None) -> str:
        if ctx is None:
            raise ValueError("State machine requires a pipeline context")
        value = getattr(ctx, router_field, None)
        if value is None:
            raise ValueError(f"State machine context is missing {router_field!r}")
        return str(value)

    route_state: Callable[[Any, PipelineContext | None], str] = _route_state

    from ..domain.dsl.conditional import ConditionalStep

    dispatcher: Step[Any, Any] = ConditionalStep[PipelineContext](
        name="state_dispatch",
        condition_callable=route_state,
        branches=normalized,
        default_branch_pipeline=None,
    )

    loop: Step[Any, Any] = Step.loop_until(
        name="state_machine",
        loop_body_pipeline=Pipeline.from_step(dispatcher),
        exit_condition_callable=lambda _out, ctx: bool(
            getattr(ctx, end_state_field, False) if ctx is not None else False
        ),
        max_loops=max_loops,
        iteration_input_mapper=lambda last, ctx, _i: last,
    )

    return Pipeline.from_step(loop)


def make_agentic_loop_pipeline(
    planner_agent: "AsyncAgentProtocol[Any, Any]",
    agent_registry: dict[str, "AsyncAgentProtocol[Any, Any]"],
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

        def __init__(self, agent_registry: dict[str, "AsyncAgentProtocol[Any, Any]"]):
            self.agent_registry = agent_registry

        async def run(self, data: Any, *, context: PipelineContext) -> ExecutedCommandLog:
            """Execute a command from the planner."""
            from flujo.application.runner import _accepts_param
            from flujo.exceptions import PausedException
            from pydantic import ValidationError

            turn = len(getattr(context, "command_log", [])) + 1 if context is not None else 1
            # If we already have an ExecutedCommandLog, treat it as terminal and avoid revalidating.
            try:
                if isinstance(data, ExecutedCommandLog):
                    _log_if_new(data, context)
                    return data
            except Exception:
                pass
            try:
                cmd = _command_adapter.validate_python(data)
            except ValidationError as e:  # pragma: no cover - planner bug
                validation_error_result = f"Invalid command: {e}"
                log_entry = ExecutedCommandLog(
                    turn=turn,
                    generated_command=data,
                    execution_result=validation_error_result,
                )
                _log_if_new(log_entry, context)
                return log_entry

            exec_result: Any = "Command type not recognized."
            try:
                if cmd.type == "run_agent":
                    agent = self.agent_registry.get(cmd.agent_name)
                    if not agent:
                        exec_result = f"Error: Agent '{cmd.agent_name}' not found."
                    else:
                        agent_kwargs: dict[str, Any] = {}
                        if _accepts_param(agent.run, "context"):
                            agent_kwargs["context"] = context
                        exec_result = await agent.run(cmd.input_data, **agent_kwargs)
                elif cmd.type == "ask_human":
                    if isinstance(context, PipelineContext):
                        # If we already have HITL data (resume path), consume it and continue.
                        # Prefer typed user_input field
                        hitl_data = getattr(context, "user_input", None)
                        if hitl_data is not None:
                            exec_result = hitl_data
                            log_entry = ExecutedCommandLog(
                                turn=turn,
                                generated_command=cmd,
                                execution_result=exec_result,
                            )
                            _log_if_new(log_entry, context)
                            # Clear resume marker now that we've consumed the payload.
                            if hasattr(context, "status"):
                                context.status = "running"
                            if hasattr(context, "loop_resume_requires_hitl_output"):
                                context.loop_resume_requires_hitl_output = False
                            return log_entry
                        if hasattr(context, "paused_step_input"):
                            context.paused_step_input = cmd
                        if hasattr(context, "loop_resume_requires_hitl_output"):
                            context.loop_resume_requires_hitl_output = True
                        if hasattr(context, "status"):
                            context.status = "paused"
                    # Do NOT create or append a log entry here; only log on resume
                    from flujo.infra import telemetry

                    telemetry.logfire.info(
                        f"_CommandExecutor raising PausedException for question: {cmd.question}"
                    )
                    pause_exc = PausedException(message=cmd.question)
                    setattr(pause_exc, "requires_resume_payload", True)
                    raise pause_exc
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
            _log_if_new(log_entry, context)
            return log_entry

    async def planner_step(data: str, *, context: PipelineContext) -> object:
        """Get the next command from the planner."""
        result = await _invoke(planner_agent, data, context=context)
        # Do not validate/normalize here: the executor is responsible for
        # validating commands and logging invalid payloads as ExecutedCommandLog entries.
        return _extract_output_value(result)

    async def command_executor_step(
        data: object, *, context: PipelineContext
    ) -> ExecutedCommandLog:
        executor: _CommandExecutor = _CommandExecutor(agent_registry)
        return await executor.run(data, context=context)

    # Create the loop body pipeline
    planner_step_s: Step[Any, Any] = Step.from_callable(planner_step, max_retries=max_retries)
    executor_step_s: Step[Any, Any] = Step.from_callable(
        command_executor_step,
        max_retries=0,  # No retries for command execution to allow HITL pausing
    )
    loop_body: Pipeline[Any, Any] = planner_step_s >> executor_step_s

    # Create the loop step with proper config
    def exit_condition(output: Any, context: PipelineContext | None) -> bool:
        """Determine whether the loop should exit.

        The primary signal is receiving an ``ExecutedCommandLog`` whose
        ``generated_command`` is a ``FinishCommand``.  In certain edge-cases
        (e.g. when the step executor returns ``None`` due to silent failures
        or validation short-circuits) `output` may be ``None``.  To remain
        robust we fall back to inspecting the latest entry in
        ``context.command_log`` if available.
        """

        # Direct check on the current output.
        if isinstance(output, ExecutedCommandLog) and isinstance(
            getattr(output, "generated_command", None), FinishCommand
        ):
            return True

        # Fallback: inspect the last command in the context log.
        if context is not None and context.command_log:
            last_cmd = context.command_log[-1].generated_command
            return isinstance(last_cmd, FinishCommand)

        return False

    def _log_if_new(log: ExecutedCommandLog, ctx: PipelineContext | None) -> None:
        """Safely append a new command log entry to the context.

        Notes
        -----
        1. Guard against ``log`` being ``None`` – this can happen during edge
           cases where the loop mappers are invoked with a placeholder value.
        2. Use *equality* (not identity) checks to prevent duplicate entries.
        """
        if ctx is None or log is None:
            # Nothing to do if there is no context or the log entry is None.
            return

        # Ignore unexpected payloads (e.g., raw strings injected during resume) to
        # prevent polluting command_log and downstream attribute access errors.
        if not isinstance(log, ExecutedCommandLog):
            return

        from flujo.domain.commands import (
            RunAgentCommand,
            AskHumanCommand,
            FinishCommand,
        )  # local import to avoid cycle

        if ctx.command_log:
            last = ctx.command_log[-1]
            # If the planner pre-logged a raw AgentCommand, replace it with the
            # richer ExecutedCommandLog representing its execution result.
            if (
                isinstance(last, (RunAgentCommand, AskHumanCommand, FinishCommand))
                and last == log.generated_command
            ):
                ctx.command_log[-1] = log
                return

        # Only append when the last entry differs to avoid duplicates.
        if not ctx.command_log or ctx.command_log[-1] != log:
            ctx.command_log.append(log)

    def _iter_mapper(
        log: ExecutedCommandLog | Any, ctx: PipelineContext | None, _i: int
    ) -> dict[str, Any]:
        """Transformation function that logs commands if not already present."""
        # Skip logging and return a default mapping if `log` is None to avoid
        # attribute errors in edge-cases where the body step failed and the
        # loop invokes the mapper with a placeholder value.
        if log is None:
            import logging

            logging.warning(
                "Log is None in _iter_mapper. Context: %s, Iteration index: %d",
                ctx,
                _i,
            )
            goal = ctx.initial_prompt if ctx is not None else ""
            return {"last_command_result": None, "goal": goal}

        # If we got a non-ExecutedCommandLog (e.g., resume payload), avoid logging it
        # and just pass through a lightweight mapping for the next iteration.
        if not isinstance(log, ExecutedCommandLog):
            goal = ctx.initial_prompt if ctx is not None else ""
            try:
                result_val = getattr(log, "execution_result", log)
            except Exception:
                result_val = log
            return {"last_command_result": result_val, "goal": goal}

        # If this is an AskHuman command, mark paused and store pending input so resume can log it
        try:
            from flujo.domain.commands import AskHumanCommand as _AskHuman
            from flujo.exceptions import PausedException as _Paused

            if isinstance(log, _AskHuman) and ctx is not None:
                if hasattr(ctx, "status"):
                    ctx.status = "paused"
                if hasattr(ctx, "pause_message"):
                    ctx.pause_message = getattr(log, "question", "Paused")
                # Save the pending command so resume can convert/log it (typed field only)
                if hasattr(ctx, "paused_step_input"):
                    ctx.paused_step_input = log
                raise _Paused(getattr(log, "question", "Paused"))
        except Exception:
            pass

        # If paused, do not log to preserve clean pause state
        if ctx is not None and getattr(ctx, "status", None) == "paused":
            goal = ctx.initial_prompt if ctx is not None else ""
            return {"last_command_result": None, "goal": goal}
        _log_if_new(log, ctx)
        goal = ctx.initial_prompt if ctx is not None else ""
        return {"last_command_result": log.execution_result, "goal": goal}

    def _output_mapper(log: ExecutedCommandLog | Any, ctx: PipelineContext | None) -> Any:
        """Transformation function that logs commands if not already present.
        Ensures the final output is always logged, even if the loop ends due to max_loops or error.
        """
        # Fallback: if `log` is None, attempt to use the latest command in the
        # context (if available) so the caller still receives a meaningful
        # `ExecutedCommandLog` instead of ``None``.
        if log is None:
            if ctx is not None and ctx.command_log:
                return ctx.command_log[-1]
            return None

        # If the loop hands us a resume payload or other unexpected type, avoid mutating
        # the log list and return the last known command entry when available.
        if not isinstance(log, ExecutedCommandLog):
            if ctx is not None and ctx.command_log:
                return ctx.command_log[-1]
            return log

        _log_if_new(log, ctx)
        return log  # Return the full log instead of just the execution result

    loop_step: Any = Step.loop_until(
        name="AgenticExplorationLoop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=max_loops,
        iteration_input_mapper=_iter_mapper,
        loop_output_mapper=_output_mapper,
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
    try:
        result = await gather_result(runner, task.prompt)
    except Exception as exc:
        from flujo.exceptions import TypeMismatchError

        # Treat type mismatch as pipeline failure for convenience helper
        if isinstance(exc, TypeMismatchError):
            return None
        raise

    # Extract solution from context and checklist from the validator step output
    context = result.final_pipeline_context
    solution = None
    checklist: Checklist | None = None

    try:
        # Primary sources filled by step callables
        if context is not None:
            solution = getattr(context, "solution", None)
            # Do not take checklist from context yet; prefer validator output below
    except Exception:
        # Best-effort: proceed to fallbacks below
        pass

    # Robust fallbacks sourced from recorded step history to avoid CI-only edge cases
    # 1) Prefer the validator step's output for the checklist
    if checklist is None:
        for step in reversed(result.step_history):
            try:
                if isinstance(step.output, Checklist):
                    checklist = step.output
                    break
            except Exception:
                continue

    # 2) If still missing, try to locate the explicit validate step by name
    if checklist is None:
        for step in result.step_history:
            try:
                if getattr(step, "name", "") == "validate_step" and isinstance(
                    step.output, Checklist
                ):
                    checklist = step.output
                    break
            except Exception:
                continue

    # 3) If still missing, fallback to checklist stored on context
    if checklist is None:
        try:
            if context is not None:
                checklist = getattr(context, "checklist", None)
        except Exception:
            pass

    # 4) Derive solution from the named solution step when missing
    if solution is None:
        for step in result.step_history:
            try:
                if getattr(step, "name", "") == "solution_step" and isinstance(step.output, str):
                    solution = step.output
                    break
            except Exception:
                continue

    # Do not silently coerce failures: preserve existing failure semantics
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
    resume_from: Optional[PipelineResult[PipelineContext]] = None,
    human_input: Optional[str] = "human",
) -> PipelineResult[PipelineContext]:
    """Run an agentic loop pipeline and return the result.

    Args:
        pipeline: Pipeline created by make_agentic_loop_pipeline
        initial_goal: Initial goal for the agentic loop
        resume_from: Optional paused result to resume from
        human_input: Human input to provide when resuming (default: "human")

    Returns:
        PipelineResult with the execution context and results
    """
    runner: Flujo[str, Any, PipelineContext] = Flujo(pipeline)

    if resume_from is not None:
        # Resume from a paused state
        result = await runner.resume_async(resume_from, human_input)
    else:
        # Start fresh execution
        result = await gather_result(runner, initial_goal)

    return result


async def _invoke(
    agent: "AsyncAgentProtocol[Any, Any]",
    data: Any,
    *,
    context: Optional[PipelineContext] = None,
) -> Any:
    """Helper function to invoke an agent with proper error handling."""
    from flujo.application.runner import _accepts_param
    from flujo.agents import AsyncAgentWrapper

    try:
        # If this is a pydantic-ai agent wrapper, never pass context (it will error)
        if isinstance(agent, AsyncAgentWrapper):
            return await agent.run(data)
        # Otherwise, only pass context if supported
        if context is not None and _accepts_param(agent.run, "context"):
            return await agent.run(data, context=context)
        else:
            return await agent.run(data)
    except Exception:
        # Handle specific exceptions as needed
        raise

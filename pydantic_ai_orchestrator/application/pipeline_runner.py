from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError


from ..infra.telemetry import logfire
from ..exceptions import (
    OrchestratorError,
    PipelineContextInitializationError,
)
from ..domain.pipeline_dsl import Pipeline, Step, LoopStep
from ..domain.plugins import PluginOutcome
from ..domain.models import PipelineResult, StepResult


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected."""


RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")


class PipelineRunner(Generic[RunnerInT, RunnerOutT]):
    """Execute a pipeline sequentially."""

    def __init__(
        self,
        pipeline: Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT],
        context_model: Optional[Type[BaseModel]] = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] = pipeline
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}

    async def _run_step(
        self,
        step: Step[Any, Any],
        data: Any,
        pipeline_context: Optional[BaseModel],
    ) -> StepResult:
        visited: set[Any] = set()
        if isinstance(step, LoopStep):
            return await self._execute_loop_step(step, data, pipeline_context)

        result = StepResult(name=step.name)
        original_agent = step.agent
        current_agent = original_agent
        last_feedback = None
        last_output = None
        for attempt in range(1, step.config.max_retries + 1):
            result.attempts = attempt
            if current_agent is None:
                raise OrchestratorError(f"Step {step.name} has no agent")

            start = time.monotonic()
            agent_kwargs = {}
            if pipeline_context is not None:
                agent_kwargs["pipeline_context"] = pipeline_context
            output = await current_agent.run(data, **agent_kwargs)
            result.latency_s += time.monotonic() - start
            last_output = output

            success = True
            feedback: str | None = None
            redirect_to = None
            final_plugin_outcome: PluginOutcome | None = None

            sorted_plugins = sorted(step.plugins, key=lambda p: p[1], reverse=True)
            for plugin, _ in sorted_plugins:
                try:
                    plugin_kwargs = {}
                    if pipeline_context is not None:
                        plugin_kwargs["pipeline_context"] = pipeline_context
                    plugin_result: PluginOutcome = await asyncio.wait_for(
                        plugin.validate({"input": data, "output": output}, **plugin_kwargs),
                        timeout=step.config.timeout_s,
                    )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Plugin timeout in step {step.name}") from e

                if not plugin_result.success:
                    success = False
                    feedback = plugin_result.feedback
                    redirect_to = plugin_result.redirect_to
                    final_plugin_outcome = plugin_result
                if plugin_result.new_solution is not None:
                    final_plugin_outcome = plugin_result

            if final_plugin_outcome and final_plugin_outcome.new_solution is not None:
                output = final_plugin_outcome.new_solution
                last_output = output

            if success:
                result.output = output
                result.success = True
                result.feedback = feedback
                result.token_counts += getattr(output, "token_counts", 1)
                result.cost_usd += getattr(output, "cost_usd", 0.0)
                return result

            # Call failure handlers on each failed attempt
            for handler in step.failure_handlers:
                handler()

            # Handle redirection for next attempt
            if redirect_to:
                if redirect_to in visited:
                    raise InfiniteRedirectError(f"Redirect loop detected in step {step.name}")
                visited.add(redirect_to)
                current_agent = redirect_to
            else:
                current_agent = original_agent

            # Update input with feedback for next attempt
            if feedback:
                if isinstance(data, dict):
                    data["feedback"] = data.get("feedback", "") + "\n" + feedback
                else:
                    data = f"{str(data)}\n{feedback}"
            last_feedback = feedback

        # If we get here, all retries failed
        result.output = last_output
        result.success = False
        result.feedback = last_feedback
        result.token_counts += getattr(last_output, "token_counts", 1) if last_output is not None else 0
        result.cost_usd += getattr(last_output, "cost_usd", 0.0) if last_output is not None else 0.0
        return result

    async def _execute_loop_step(
        self,
        loop_step: LoopStep,
        loop_step_initial_input: Any,
        pipeline_context: Optional[BaseModel],
    ) -> StepResult:
        loop_overall_result = StepResult(name=loop_step.name)

        if loop_step.initial_input_to_loop_body_mapper:
            current_body_input = loop_step.initial_input_to_loop_body_mapper(
                loop_step_initial_input, pipeline_context
            )
        else:
            current_body_input = loop_step_initial_input

        last_successful_iteration_body_output: Any = None
        final_body_output_of_last_iteration: Any = None
        loop_exited_successfully_by_condition = False

        for i in range(1, loop_step.max_loops + 1):
            loop_overall_result.attempts = i
            logfire.info(
                f"LoopStep '{loop_step.name}': Starting Iteration {i}/{loop_step.max_loops}"
            )

            iteration_succeeded_fully = True
            current_iteration_data_for_body_step = current_body_input

            for body_s in loop_step.loop_body_pipeline.steps:
                with logfire.span(
                    f"LoopStep '{loop_step.name}' Iteration {i} - Body Step '{body_s.name}'"
                ):
                    body_step_result_obj = await self._run_step(
                        body_s, current_iteration_data_for_body_step, pipeline_context
                    )

                loop_overall_result.latency_s += body_step_result_obj.latency_s
                loop_overall_result.cost_usd += getattr(body_step_result_obj, "cost_usd", 0.0)
                loop_overall_result.token_counts += getattr(body_step_result_obj, "token_counts", 0)

                if not body_step_result_obj.success:
                    logfire.warn(
                        f"Body Step '{body_s.name}' in LoopStep '{loop_step.name}' (Iteration {i}) failed."
                    )
                    iteration_succeeded_fully = False
                    final_body_output_of_last_iteration = body_step_result_obj.output
                    break

                current_iteration_data_for_body_step = body_step_result_obj.output

            if iteration_succeeded_fully:
                last_successful_iteration_body_output = current_iteration_data_for_body_step
            final_body_output_of_last_iteration = current_iteration_data_for_body_step

            try:
                should_exit = loop_step.exit_condition_callable(
                    final_body_output_of_last_iteration, pipeline_context
                )
            except Exception as e:
                logfire.error(
                    f"Error in exit_condition_callable for LoopStep '{loop_step.name}': {e}"
                )
                loop_overall_result.success = False
                loop_overall_result.feedback = (
                    f"Exit condition callable raised an exception: {e}"
                )
                break

            if should_exit:
                logfire.info(f"LoopStep '{loop_step.name}' exit condition met at iteration {i}.")
                loop_overall_result.success = iteration_succeeded_fully
                if not iteration_succeeded_fully:
                    loop_overall_result.feedback = (
                        "Loop exited by condition, but last iteration body failed."
                    )
                loop_exited_successfully_by_condition = True
                break

            if i < loop_step.max_loops:
                if loop_step.iteration_input_mapper:
                    try:
                        current_body_input = loop_step.iteration_input_mapper(
                            final_body_output_of_last_iteration, pipeline_context, i
                        )
                    except Exception as e:
                        logfire.error(
                            f"Error in iteration_input_mapper for LoopStep '{loop_step.name}': {e}"
                        )
                        loop_overall_result.success = False
                        loop_overall_result.feedback = (
                            f"Iteration input mapper raised an exception: {e}"
                        )
                        break
                else:
                    current_body_input = final_body_output_of_last_iteration
        else:
            logfire.warn(
                f"LoopStep '{loop_step.name}' reached max_loops ({loop_step.max_loops}) without exit condition being met."
            )
            loop_overall_result.success = False
            loop_overall_result.feedback = (
                f"Reached max_loops ({loop_step.max_loops}) without meeting exit condition."
            )

        if loop_overall_result.success and loop_exited_successfully_by_condition:
            if loop_step.loop_output_mapper:
                try:
                    loop_overall_result.output = loop_step.loop_output_mapper(
                        last_successful_iteration_body_output, pipeline_context
                    )
                except Exception as e:
                    logfire.error(
                        f"Error in loop_output_mapper for LoopStep '{loop_step.name}': {e}"
                    )
                    loop_overall_result.success = False
                    loop_overall_result.feedback = (
                        f"Loop output mapper raised an exception: {e}"
                    )
                    loop_overall_result.output = None
            else:
                loop_overall_result.output = last_successful_iteration_body_output
        else:
            loop_overall_result.output = final_body_output_of_last_iteration
            if not loop_overall_result.feedback:
                loop_overall_result.feedback = (
                    "Loop did not complete successfully or exit condition not met positively."
                )

        return loop_overall_result

    async def run_async(self, initial_input: RunnerInT) -> PipelineResult:
        current_pipeline_context_instance: Optional[BaseModel] = None
        if self.context_model is not None:
            try:
                current_pipeline_context_instance = self.context_model(
                    **self.initial_context_data
                )
            except ValidationError as e:
                logfire.error(
                    f"Pipeline context initialization failed for model {self.context_model.__name__}: {e}"
                )
                raise PipelineContextInitializationError(
                    f"Failed to initialize pipeline context with model {self.context_model.__name__} and initial data. Validation errors:\n{e}"
                ) from e

        data = initial_input
        pipeline_result_obj = PipelineResult()
        try:
            for step in self.pipeline.steps:
                with logfire.span(step.name):
                    step_result = await self._run_step(
                        step, data, pipeline_context=current_pipeline_context_instance
                    )
                pipeline_result_obj.step_history.append(step_result)
                pipeline_result_obj.total_cost_usd += step_result.cost_usd
                if not step_result.success:
                    logfire.warn(
                        f"Step '{step.name}' failed. Halting pipeline execution."
                    )
                    break
                data = step_result.output
        except asyncio.CancelledError:
            logfire.info("Pipeline cancelled")
            return pipeline_result_obj

        if current_pipeline_context_instance is not None:
            pipeline_result_obj.final_pipeline_context = current_pipeline_context_instance

        return pipeline_result_obj

    def run(self, initial_input: RunnerInT) -> PipelineResult:
        return asyncio.run(self.run_async(initial_input))

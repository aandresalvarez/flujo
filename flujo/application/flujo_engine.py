from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError


from ..infra.telemetry import logfire
from ..exceptions import (
    OrchestratorError,
    PipelineContextInitializationError,
    UsageLimitExceededError,
    PipelineAbortSignal,
)
from ..domain.pipeline_dsl import (
    Pipeline,
    Step,
    LoopStep,
    ConditionalStep,
    BranchKey,
)
from ..domain.plugins import PluginOutcome
from ..domain.models import PipelineResult, StepResult, UsageLimits
from ..domain.resources import AppResources
from ..domain.types import HookCallable


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected."""


RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")


class Flujo(Generic[RunnerInT, RunnerOutT]):
    """Execute a pipeline sequentially."""

    def __init__(
        self,
        pipeline: Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT],
        context_model: Optional[Type[BaseModel]] = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
        resources: Optional[AppResources] = None,
        usage_limits: Optional[UsageLimits] = None,
        hooks: Optional[list[HookCallable]] = None,
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] = pipeline
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}
        self.resources = resources
        self.usage_limits = usage_limits
        self.hooks = hooks or []

    async def _dispatch_hook(self, event_name: str, **kwargs: Any) -> None:
        for hook in self.hooks:
            try:
                await hook(event_name=event_name, **kwargs)
            except PipelineAbortSignal:
                raise
            except Exception as e:  # noqa: BLE001
                name = getattr(hook, "__name__", str(hook))
                logfire.error(f"Error in hook '{name}': {e}")

    async def _run_step(
        self,
        step: Step[Any, Any],
        data: Any,
        pipeline_context: Optional[BaseModel],
        resources: Optional[AppResources],
    ) -> StepResult:
        visited: set[Any] = set()
        if isinstance(step, LoopStep):
            return await self._execute_loop_step(
                step, data, pipeline_context, resources
            )
        elif isinstance(step, ConditionalStep):
            return await self._execute_conditional_step(
                step, data, pipeline_context, resources
            )

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
            if resources is not None:
                agent_kwargs["resources"] = resources
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
                    if resources is not None:
                        plugin_kwargs["resources"] = resources
                    plugin_result: PluginOutcome = await asyncio.wait_for(
                        plugin.validate(
                            {"input": data, "output": output}, **plugin_kwargs
                        ),
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
                    raise InfiniteRedirectError(
                        f"Redirect loop detected in step {step.name}"
                    )
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
        result.token_counts += (
            getattr(last_output, "token_counts", 1) if last_output is not None else 0
        )
        result.cost_usd += (
            getattr(last_output, "cost_usd", 0.0) if last_output is not None else 0.0
        )
        return result

    async def _execute_loop_step(
        self,
        loop_step: LoopStep,
        loop_step_initial_input: Any,
        pipeline_context: Optional[BaseModel],
        resources: Optional[AppResources],
    ) -> StepResult:
        loop_overall_result = StepResult(name=loop_step.name)

        if loop_step.initial_input_to_loop_body_mapper:
            try:
                current_body_input = loop_step.initial_input_to_loop_body_mapper(
                    loop_step_initial_input, pipeline_context
                )
            except Exception as e:  # noqa: BLE001
                logfire.error(
                    f"Error in initial_input_to_loop_body_mapper for LoopStep '{loop_step.name}': {e}"
                )
                loop_overall_result.success = False
                loop_overall_result.feedback = (
                    f"Initial input mapper raised an exception: {e}"
                )
                return loop_overall_result
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
                        body_s,
                        current_iteration_data_for_body_step,
                        pipeline_context,
                        resources,
                    )

                loop_overall_result.latency_s += body_step_result_obj.latency_s
                loop_overall_result.cost_usd += getattr(
                    body_step_result_obj, "cost_usd", 0.0
                )
                loop_overall_result.token_counts += getattr(
                    body_step_result_obj, "token_counts", 0
                )

                if not body_step_result_obj.success:
                    logfire.warn(
                        f"Body Step '{body_s.name}' in LoopStep '{loop_step.name}' (Iteration {i}) failed."
                    )
                    iteration_succeeded_fully = False
                    final_body_output_of_last_iteration = body_step_result_obj.output
                    break

                current_iteration_data_for_body_step = body_step_result_obj.output

            if iteration_succeeded_fully:
                last_successful_iteration_body_output = (
                    current_iteration_data_for_body_step
                )
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
                logfire.info(
                    f"LoopStep '{loop_step.name}' exit condition met at iteration {i}."
                )
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
            loop_overall_result.feedback = f"Reached max_loops ({loop_step.max_loops}) without meeting exit condition."

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
                loop_overall_result.feedback = "Loop did not complete successfully or exit condition not met positively."

        return loop_overall_result

    async def _execute_conditional_step(
        self,
        conditional_step: ConditionalStep,
        conditional_step_input: Any,
        pipeline_context: Optional[BaseModel],
        resources: Optional[AppResources],
    ) -> StepResult:
        conditional_overall_result = StepResult(name=conditional_step.name)
        executed_branch_key: BranchKey | None = None
        branch_output: Any = None
        branch_succeeded = False

        try:
            branch_key_to_execute = conditional_step.condition_callable(
                conditional_step_input, pipeline_context
            )
            logfire.info(
                f"ConditionalStep '{conditional_step.name}': Condition evaluated to branch key '{branch_key_to_execute}'."
            )
            executed_branch_key = branch_key_to_execute

            selected_branch_pipeline = conditional_step.branches.get(
                branch_key_to_execute
            )
            if selected_branch_pipeline is None:
                selected_branch_pipeline = conditional_step.default_branch_pipeline
                if selected_branch_pipeline is None:
                    err_msg = f"ConditionalStep '{conditional_step.name}': No branch found for key '{branch_key_to_execute}' and no default branch defined."
                    logfire.warn(err_msg)
                    conditional_overall_result.success = False
                    conditional_overall_result.feedback = err_msg
                    return conditional_overall_result
                logfire.info(
                    f"ConditionalStep '{conditional_step.name}': Executing default branch."
                )
            else:
                logfire.info(
                    f"ConditionalStep '{conditional_step.name}': Executing branch for key '{branch_key_to_execute}'."
                )

            if conditional_step.branch_input_mapper:
                input_for_branch = conditional_step.branch_input_mapper(
                    conditional_step_input, pipeline_context
                )
            else:
                input_for_branch = conditional_step_input

            current_branch_data = input_for_branch
            branch_pipeline_failed_internally = False

            for branch_s in selected_branch_pipeline.steps:
                with logfire.span(
                    f"ConditionalStep '{conditional_step.name}' Branch '{branch_key_to_execute}' - Step '{branch_s.name}'"
                ):
                    branch_step_result_obj = await self._run_step(
                        branch_s,
                        current_branch_data,
                        pipeline_context,
                        resources,
                    )

                conditional_overall_result.latency_s += branch_step_result_obj.latency_s
                conditional_overall_result.cost_usd += getattr(
                    branch_step_result_obj, "cost_usd", 0.0
                )
                conditional_overall_result.token_counts += getattr(
                    branch_step_result_obj, "token_counts", 0
                )

                if not branch_step_result_obj.success:
                    logfire.warn(
                        f"Step '{branch_s.name}' in branch '{branch_key_to_execute}' of ConditionalStep '{conditional_step.name}' failed."
                    )
                    branch_pipeline_failed_internally = True
                    branch_output = branch_step_result_obj.output
                    conditional_overall_result.feedback = f"Failure in branch '{branch_key_to_execute}', step '{branch_s.name}': {branch_step_result_obj.feedback}"
                    break

                current_branch_data = branch_step_result_obj.output

            if not branch_pipeline_failed_internally:
                branch_output = current_branch_data
                branch_succeeded = True

        except Exception as e:  # noqa: BLE001
            logfire.error(
                f"Error during ConditionalStep '{conditional_step.name}' execution: {e}",
                exc_info=True,
            )
            conditional_overall_result.success = False
            conditional_overall_result.feedback = (
                f"Error executing conditional logic or branch: {e}"
            )
            return conditional_overall_result

        conditional_overall_result.success = branch_succeeded
        if branch_succeeded:
            if conditional_step.branch_output_mapper:
                try:
                    conditional_overall_result.output = (
                        conditional_step.branch_output_mapper(
                            branch_output, executed_branch_key, pipeline_context
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logfire.error(
                        f"Error in branch_output_mapper for ConditionalStep '{conditional_step.name}': {e}"
                    )
                    conditional_overall_result.success = False
                    conditional_overall_result.feedback = (
                        f"Branch output mapper raised an exception: {e}"
                    )
                    conditional_overall_result.output = None
            else:
                conditional_overall_result.output = branch_output
        else:
            conditional_overall_result.output = branch_output

        conditional_overall_result.attempts = 1
        if executed_branch_key is not None:
            conditional_overall_result.metadata_ = (
                conditional_overall_result.metadata_ or {}
            )
            conditional_overall_result.metadata_["executed_branch_key"] = str(
                executed_branch_key
            )

        return conditional_overall_result

    def _check_usage_limits(
        self,
        pipeline_result: PipelineResult,
        span: Any | None,
    ) -> None:
        if self.usage_limits is None:
            return

        total_tokens = sum(sr.token_counts for sr in pipeline_result.step_history)

        if (
            self.usage_limits.total_cost_usd_limit is not None
            and pipeline_result.total_cost_usd > self.usage_limits.total_cost_usd_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception:  # noqa: BLE001
                    pass
            logfire.warn(
                f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded"
            )
            raise UsageLimitExceededError(
                f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded",
                pipeline_result,
            )

        if (
            self.usage_limits.total_tokens_limit is not None
            and total_tokens > self.usage_limits.total_tokens_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception:  # noqa: BLE001
                    pass
            logfire.warn(
                f"Token limit of {self.usage_limits.total_tokens_limit} exceeded"
            )
            raise UsageLimitExceededError(
                f"Token limit of {self.usage_limits.total_tokens_limit} exceeded",
                pipeline_result,
            )

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

        data: Optional[RunnerInT] = initial_input
        pipeline_result_obj = PipelineResult()
        try:
            await self._dispatch_hook(
                "pre_run",
                initial_input=initial_input,
                pipeline_context=current_pipeline_context_instance,
                resources=self.resources,
            )
            for step in self.pipeline.steps:
                await self._dispatch_hook(
                    "pre_step",
                    step=step,
                    step_input=data,
                    pipeline_context=current_pipeline_context_instance,
                    resources=self.resources,
                )
                with logfire.span(step.name) as span:
                    step_result = await self._run_step(
                        step,
                        data,
                        pipeline_context=current_pipeline_context_instance,
                        resources=self.resources,
                    )
                    if step_result.metadata_:
                        for key, value in step_result.metadata_.items():
                            try:
                                span.set_attribute(key, value)
                            except Exception:  # noqa: BLE001
                                pass
                    pipeline_result_obj.step_history.append(step_result)
                    pipeline_result_obj.total_cost_usd += step_result.cost_usd
                    self._check_usage_limits(pipeline_result_obj, span)
                if step_result.success:
                    await self._dispatch_hook(
                        "post_step",
                        step_result=step_result,
                        pipeline_context=current_pipeline_context_instance,
                        resources=self.resources,
                    )
                else:
                    await self._dispatch_hook(
                        "on_step_failure",
                        step_result=step_result,
                        pipeline_context=current_pipeline_context_instance,
                        resources=self.resources,
                    )
                    logfire.warn(
                        f"Step '{step.name}' failed. Halting pipeline execution."
                    )
                    break
                step_output: Optional[RunnerInT] = step_result.output
                data = step_output
        except asyncio.CancelledError:
            logfire.info("Pipeline cancelled")
            return pipeline_result_obj
        except PipelineAbortSignal as e:
            logfire.info(str(e))
        except UsageLimitExceededError as e:
            if current_pipeline_context_instance is not None:
                pipeline_result_obj.final_pipeline_context = (
                    current_pipeline_context_instance
                )
            raise e
        finally:
            if current_pipeline_context_instance is not None:
                pipeline_result_obj.final_pipeline_context = (
                    current_pipeline_context_instance
                )
            await self._dispatch_hook(
                "post_run",
                pipeline_result=pipeline_result_obj,
                pipeline_context=current_pipeline_context_instance,
                resources=self.resources,
            )

        return pipeline_result_obj

    def run(self, initial_input: RunnerInT) -> PipelineResult:
        return asyncio.run(self.run_async(initial_input))

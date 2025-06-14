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
from ..domain.pipeline_dsl import Pipeline, Step
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
            call_kwargs: dict[str, Any] = {}
            if pipeline_context is not None:
                call_kwargs["pipeline_context"] = pipeline_context
            try:
                output = await current_agent.run(data, **call_kwargs)
            except TypeError as e:
                if "pipeline_context" in str(e) and pipeline_context is not None:
                    err_msg = (
                        f"Agent '{current_agent.__class__.__name__}' in step '{step.name}' does not accept "
                        "'pipeline_context' keyword argument. Define it or use **kwargs."
                    )
                    logfire.error(err_msg)
                    raise TypeError(err_msg) from e
                raise
            result.latency_s += time.monotonic() - start
            last_output = output

            success = True
            feedback: str | None = None
            redirect_to = None
            final_plugin_outcome: PluginOutcome | None = None

            sorted_plugins = sorted(step.plugins, key=lambda p: p[1], reverse=True)
            for plugin, _ in sorted_plugins:
                try:
                    plugin_result: PluginOutcome = await asyncio.wait_for(
                        plugin.validate({"input": data, "output": output}, **call_kwargs),
                        timeout=step.config.timeout_s,
                    )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Plugin timeout in step {step.name}") from e
                except TypeError as e:
                    if "pipeline_context" in str(e) and pipeline_context is not None:
                        err_msg = (
                            f"Plugin '{plugin.__class__.__name__}' in step '{step.name}' does not accept "
                            "'pipeline_context' keyword argument. Define it or use **kwargs."
                        )
                        logfire.error(err_msg)
                        raise TypeError(err_msg) from e
                    raise

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
                with logfire.span(f"Step: {step.name}", pipeline_step_name=step.name):
                    if current_pipeline_context_instance is not None:
                        logfire.debug(
                            f"Context before step {step.name}: {current_pipeline_context_instance.model_dump_json(exclude_none=True)}"
                        )
                    step_result = await self._run_step(
                        step, data, pipeline_context=current_pipeline_context_instance
                    )
                    if current_pipeline_context_instance is not None:
                        logfire.debug(
                            f"Context after step {step.name}: {current_pipeline_context_instance.model_dump_json(exclude_none=True)}"
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

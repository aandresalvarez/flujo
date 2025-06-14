from __future__ import annotations

import asyncio
import time
from typing import Any


from ..infra.telemetry import logfire
from ..exceptions import OrchestratorError
from ..domain.pipeline_dsl import Pipeline, Step
from ..domain.plugins import PluginOutcome
from ..domain.models import PipelineResult, StepResult


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected."""


class PipelineRunner:
    """Execute a pipeline sequentially."""

    def __init__(self, pipeline: Pipeline | Step):
        if isinstance(pipeline, Step):
            pipeline = Pipeline(steps=[pipeline])
        self.pipeline = pipeline

    async def _run_step(self, step: Step, data: Any) -> StepResult:
        visited: set[Any] = set()
        result = StepResult(name=step.name)
        for attempt in range(1, step.config.max_retries + 1):
            result.attempts = attempt
            agent = step.agent
            if agent is None:
                raise OrchestratorError(f"Step {step.name} has no agent")
            start = time.monotonic()
            output = await agent.run(data)
            result.latency_s += time.monotonic() - start
            success = True
            feedback: str | None = None
            redirect_to = None
            final_plugin_outcome: PluginOutcome | None = None
            sorted_plugins = sorted(step.plugins, key=lambda p: p[1], reverse=True)
            for plugin, _ in sorted_plugins:
                try:
                    plugin_result: PluginOutcome = await asyncio.wait_for(
                        plugin.validate({"input": data, "output": output}),
                        timeout=step.config.timeout_s,
                    )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Plugin timeout in step {step.name}") from e
                if not plugin_result.success:
                    success = False
                    feedback = plugin_result.feedback
                    redirect_to = plugin_result.redirect_to
                    final_plugin_outcome = plugin_result
                elif plugin_result.new_solution is not None:
                    final_plugin_outcome = plugin_result
            if final_plugin_outcome and final_plugin_outcome.new_solution is not None:
                output = final_plugin_outcome.new_solution
            if success:
                result.output = output
                result.success = True
                result.feedback = feedback
                result.token_counts += getattr(output, "token_counts", 1)
                result.cost_usd += getattr(output, "cost_usd", 0.0)
                return result
            # failure -> prepare for retry
            if redirect_to:
                if redirect_to in visited:
                    raise InfiniteRedirectError(
                        f"Redirect loop detected in step {step.name}"
                    )
                visited.add(redirect_to)
                step.agent = redirect_to
            if feedback:
                if isinstance(data, dict):
                    data["feedback"] = data.get("feedback", "") + "\n" + feedback
                else:
                    data = f"{str(data)}\n{feedback}"
            for handler in step.failure_handlers:
                handler()
        result.output = output
        result.success = False
        result.feedback = feedback
        result.token_counts += getattr(output, "token_counts", 1)
        result.cost_usd += getattr(output, "cost_usd", 0.0)
        return result

    async def run_async(self, initial_input: Any) -> PipelineResult:
        data = initial_input
        result = PipelineResult()
        try:
            for step in self.pipeline.steps:
                with logfire.span(step.name):
                    step_result = await self._run_step(step, data)
                result.step_history.append(step_result)
                result.total_cost_usd += step_result.cost_usd
                data = step_result.output
        except asyncio.CancelledError:
            logfire.info("Pipeline cancelled")
            return result
        return result

    def run(self, initial_input: Any) -> PipelineResult:
        return asyncio.run(self.run_async(initial_input))

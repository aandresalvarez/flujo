from __future__ import annotations

import asyncio
import time
from typing import Any, List

from pydantic import BaseModel, Field

from ..infra.telemetry import logfire
from ..exceptions import OrchestratorError
from ..domain.pipeline_dsl import Pipeline, Step
from ..domain.plugins import ValidationPlugin, PluginOutcome


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected."""


class StepStats(BaseModel):
    attempts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


class StepResult(BaseModel):
    name: str
    output: Any | None = None
    success: bool = True
    feedback: str | None = None
    stats: StepStats = Field(default_factory=StepStats)
    plugin_results: List[PluginOutcome] = Field(default_factory=list)

    @property
    def attempts(self) -> int:  # pragma: no cover - passthrough
        return self.stats.attempts

    @property
    def latency_s(self) -> float:  # pragma: no cover - passthrough
        return self.stats.latency_s

    @property
    def cost_usd(self) -> float:  # pragma: no cover - passthrough
        return self.stats.cost_usd

    @property
    def token_counts(self) -> int:  # pragma: no cover - passthrough
        return self.stats.tokens_out


class PipelineResult(BaseModel):
    step_history: List[StepResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class PipelineRunner:
    """Execute a pipeline sequentially."""

    def __init__(self, pipeline: Pipeline | Step):
        if isinstance(pipeline, Step):
            pipeline = Pipeline.model_construct(steps=[pipeline])
        self.pipeline = pipeline

    async def _run_step(self, step: Step, data: Any) -> StepResult:
        visited: set[Any] = set()
        result = StepResult(name=step.name)
        for attempt in range(1, step.config.max_retries + 1):
            result.stats.attempts = attempt
            agent = step.agent
            if agent is None:
                raise OrchestratorError(f"Step {step.name} has no agent")
            start = time.monotonic()
            output = await agent.run(data)
            result.stats.latency_s += time.monotonic() - start
            success = True
            feedback: str | None = None
            redirect_agent = None
            for plugin in step.plugins:
                try:
                    plugin_result: PluginOutcome = await asyncio.wait_for(
                        plugin.validate({"input": data, "output": output}),
                        timeout=step.config.timeout_s,
                    )
                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Plugin timeout in step {step.name}") from e
                result.plugin_results.append(plugin_result)
                if not plugin_result.ok:
                    success = False
                    if plugin_result.new_feedback:
                        feedback = plugin_result.new_feedback
                    if plugin_result.redirect_agent:
                        redirect_agent = plugin_result.redirect_agent
                    if plugin_result.new_solution is not None:
                        output = plugin_result.new_solution
            if success:
                result.output = output
                result.success = True
                result.feedback = feedback
                result.stats.tokens_out += getattr(output, "token_counts", 1)
                result.stats.cost_usd += getattr(output, "cost_usd", 0.0)
                return result
            # failure -> prepare for retry
            if redirect_agent:
                if redirect_agent in visited:
                    raise InfiniteRedirectError(f"Redirect loop detected in step {step.name}")
                visited.add(redirect_agent)
                step.agent = redirect_agent
            if feedback:
                if isinstance(data, str):
                    data = f"{data}\n{feedback}"
            for handler in step.failure_handlers:
                handler()
        result.output = output
        result.success = False
        result.feedback = feedback
        result.stats.tokens_out += getattr(output, "token_counts", 1)
        result.stats.cost_usd += getattr(output, "cost_usd", 0.0)
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

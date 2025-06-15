"""Evaluation and self-improvement utilities."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable, Iterable, Optional

from pydantic_evals.reporting import EvaluationReport, ReportCase

from ..domain.models import (
    ImprovementReport,
    PipelineResult,
)
from ..domain.pipeline_dsl import Pipeline, Step
from ..utils.redact import summarize_and_redact_prompt


class SelfImprovementAgent:
    """Agent that analyzes failures and suggests improvements."""

    def __init__(self, agent: Any):
        self._agent = agent

    async def run(self, prompt: str) -> ImprovementReport:
        raw = await self._agent.run(prompt)
        if isinstance(raw, (dict, list)):
            data = json.dumps(raw)
        else:
            data = str(raw)
        return ImprovementReport.model_validate_json(data)


def _find_step(pipeline: Pipeline[Any, Any] | Step[Any, Any] | None, name: str) -> Step[Any, Any] | None:
    if pipeline is None:
        return None
    if isinstance(pipeline, Step):
        return pipeline if pipeline.name == name else None
    for step in pipeline.steps:
        if step.name == name:
            return step
    return None


def _build_context(
    failures: Iterable[ReportCase],
    success: ReportCase | None,
    pipeline_definition: Pipeline[Any, Any] | Step[Any, Any] | None = None,
) -> str:
    lines: list[str] = []
    for case in failures:
        pr: PipelineResult = case.output
        lines.append(f"Case: {case.name}")
        for step in pr.step_history:
            lines.append(f"- {step.name}: {step.output} (success={step.success})")
            step_obj = _find_step(pipeline_definition, step.name)
            if step_obj is not None:
                cfg = step_obj.config
                lines.append(f"  Config(retries={cfg.max_retries}, timeout={cfg.timeout_s}s)")
                if step_obj.agent is not None:
                    summary = summarize_and_redact_prompt(step_obj.agent.system_prompt)
                    lines.append(f'  SystemPromptSummary: "{summary}"')
    if success:
        lines.append("Successful example:")
        pr = success.output
        for step in pr.step_history:
            lines.append(f"- {step.name}: {step.output}")
            step_obj = _find_step(pipeline_definition, step.name)
            if step_obj is not None:
                cfg = step_obj.config
                lines.append(f"  Config(retries={cfg.max_retries}, timeout={cfg.timeout_s}s)")
                if step_obj.agent is not None:
                    summary = summarize_and_redact_prompt(step_obj.agent.system_prompt)
                    lines.append(f'  SystemPromptSummary: "{summary}"')
    return "\n".join(lines)


async def evaluate_and_improve(
    task_function: Callable[[Any], Awaitable[PipelineResult]],
    dataset: Any,
    improvement_agent: SelfImprovementAgent,
    pipeline_definition: Optional[Pipeline[Any, Any] | Step[Any, Any]] = None,
) -> ImprovementReport:
    """Run dataset evaluation and return improvement suggestions."""
    report: EvaluationReport = await dataset.evaluate(task_function)
    failures = [c for c in report.cases if any(not s.success for s in c.output.step_history)]
    success = next((c for c in report.cases if all(s.success for s in c.output.step_history)), None)
    prompt = _build_context(failures, success, pipeline_definition)
    return await improvement_agent.run(prompt)


__all__ = [
    "SelfImprovementAgent",
    "evaluate_and_improve",
    "ImprovementReport",
]

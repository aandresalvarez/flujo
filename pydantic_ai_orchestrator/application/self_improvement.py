"""Evaluation and self-improvement utilities."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable, Iterable

from pydantic_evals.reporting import EvaluationReport, ReportCase

from ..domain.models import (
    ImprovementReport,
    PipelineResult,
)


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


def _build_context(failures: Iterable[ReportCase], success: ReportCase | None) -> str:
    lines: list[str] = []
    for case in failures:
        pr: PipelineResult = case.output
        lines.append(f"Case: {case.name}")
        for step in pr.step_history:
            lines.append(f"- {step.name}: {step.output} (success={step.success})")
    if success:
        lines.append("Successful example:")
        pr = success.output
        for step in pr.step_history:
            lines.append(f"- {step.name}: {step.output}")
    return "\n".join(lines)


async def evaluate_and_improve(
    task_function: Callable[[Any], Awaitable[PipelineResult]],
    dataset: Any,
    improvement_agent: SelfImprovementAgent,
) -> ImprovementReport:
    """Run dataset evaluation and return improvement suggestions."""
    report: EvaluationReport = await dataset.evaluate(task_function)
    failures = [c for c in report.cases if any(not s.success for s in c.output.step_history)]
    success = next((c for c in report.cases if all(s.success for s in c.output.step_history)), None)
    prompt = _build_context(failures, success)
    return await improvement_agent.run(prompt)


__all__ = [
    'SelfImprovementAgent',
    'evaluate_and_improve',
    'ImprovementReport',
]

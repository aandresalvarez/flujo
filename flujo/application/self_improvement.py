"""Evaluation and self-improvement utilities."""

from __future__ import annotations

from typing import Any, Callable, Awaitable, Iterable, Optional

from pydantic_evals.reporting import EvaluationReport, ReportCase

from ..domain.models import (
    ImprovementReport,
    PipelineResult,
    StepResult,
)
from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import Step
from ..utils.redact import summarize_and_redact_prompt

MAX_STEP_OUTPUT_LENGTH = 150


class SelfImprovementAgent:
    """Agent that analyzes failures and suggests improvements."""

    def __init__(self, agent: Any):
        self._agent = agent

    async def run(self, prompt: str) -> ImprovementReport:
        result = await self._agent.run(prompt)
        if isinstance(result, ImprovementReport):
            return result
        return ImprovementReport.model_validate(result)


def _find_step(
    pipeline: Pipeline[Any, Any] | Step[Any, Any] | None, name: str
) -> Step[Any, Any] | None:
    """Return the step with ``name`` from ``pipeline`` if present.

    Parameters
    ----------
    pipeline:
        Pipeline or step to search within.
    name:
        Name of the desired step.

    Returns
    -------
    Step | None
        The matching step instance or ``None`` if not found.
    """
    if pipeline is None:
        return None
    if isinstance(pipeline, Step):
        return pipeline if pipeline.name == name else None
    for step in pipeline.steps:
        if step.name == name:
            return step
    return None


def _safe_str(obj: Any) -> str:
    """Convert obj to string, falling back to repr on error."""
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit, appending ellipsis only if truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_step_output(
    step: StepResult,
    pipeline_definition: Pipeline[Any, Any] | Step[Any, Any] | None,
) -> list[str]:
    """Return formatted lines describing a step result.

    Parameters
    ----------
    step:
        The :class:`StepResult` to summarize.
    pipeline_definition:
        Pipeline definition used to pull step metadata.

    Returns
    -------
    list[str]
        Human-readable lines describing the step's output and configuration.
    """
    lines: list[str] = []
    out_str = _truncate(_safe_str(step.output), MAX_STEP_OUTPUT_LENGTH)
    feedback = getattr(step, "feedback", None)
    feedback_str = f", feedback='{feedback}'" if feedback is not None else ""
    lines.append(f"- Step '{step.name}': Output='{out_str}' (success={step.success}{feedback_str})")
    step_obj = _find_step(pipeline_definition, step.name)
    if step_obj is not None:
        cfg = step_obj.config
        parts = [f"retries={cfg.max_retries}", f"timeout={cfg.timeout_s}s"]
        if cfg.temperature is not None:
            parts.append(f"temperature={cfg.temperature}")
        lines.append(f"  Config({', '.join(parts)})")
        if step_obj.agent is not None:
            summary = summarize_and_redact_prompt(getattr(step_obj.agent, "system_prompt", ""))
            lines.append(f'  SystemPromptSummary: "{summary}"')
    return lines


def _build_context(
    failures: Iterable[ReportCase],
    success: ReportCase | None,
    pipeline_definition: Pipeline[Any, Any] | Step[Any, Any] | None = None,
) -> str:
    """Construct the prompt fed to the improvement agent.

    Parameters
    ----------
    failures:
        Iterable of failing evaluation cases.
    success:
        Optional successful case for comparison.
    pipeline_definition:
        Pipeline or single step definition used for additional context.

    Returns
    -------
    str
        The text prompt summarizing the evaluation results.
    """
    lines: list[str] = []
    lines.append(
        "Analyze the following failed and successful pipeline runs to identify root causes and suggest improvements."
    )

    lines.append("")
    lines.append("--- FAILED CASES ---")
    lines.append("")
    for case in failures:
        pr: PipelineResult[Any] = case.output
        lines.append(f"Case: {case.name}")
        input_str = _truncate(_safe_str(case.inputs), 200)
        lines.append(f"Input: {input_str}")
        for step in pr.step_history:
            lines.extend(_format_step_output(step, pipeline_definition))
        lines.append("")

    if success:
        lines.append("\n--- SUCCESSFUL EXAMPLE FOR CONTRAST ---\n")
        pr = success.output
        lines.append(f"Case: {success.name}")
        input_str = _truncate(_safe_str(success.inputs), 200)
        lines.append(f"Input: {input_str}")
        for step in pr.step_history:
            lines.extend(_format_step_output(step, pipeline_definition))

    return "\n".join(lines)


async def evaluate_and_improve(
    task_function: Callable[[Any], Awaitable[PipelineResult[Any]]],
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

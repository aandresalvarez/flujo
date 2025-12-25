from __future__ import annotations

from typing import Any

from ..domain.pipeline_validation import ValidationFinding, ValidationReport
from .linters_base import BaseLinter, _load_rule_overrides, _override_severity, logfire
from .linters_control import (
    ExceptionLinter,
    HitlNestedContextLinter,
    LoopScopingLinter,
    TemplateControlStructureLinter,
)
from .linters_imports import AgentLinter, ImportLinter
from .linters_orchestration import OrchestrationLinter
from .linters_schema import ContextLinter, SchemaLinter
from .linters_template import TemplateLinter

__all__ = [
    "BaseLinter",
    "TemplateLinter",
    "SchemaLinter",
    "ContextLinter",
    "ImportLinter",
    "AgentLinter",
    "OrchestrationLinter",
    "ExceptionLinter",
    "LoopScopingLinter",
    "TemplateControlStructureLinter",
    "HitlNestedContextLinter",
    "run_linters",
    "_load_rule_overrides",
    "_override_severity",
    "ValidationFinding",
    "ValidationReport",
    "logfire",
]


def run_linters(pipeline: Any, visited: set[int] | None = None) -> ValidationReport:
    """Run linters and return a ValidationReport (always-on)."""
    if visited is None:
        visited = set()

    pid = id(pipeline)
    if pid in visited:
        return ValidationReport(errors=[], warnings=[])
    visited.add(pid)

    linters: list[BaseLinter] = [
        TemplateLinter(),
        SchemaLinter(),
        ContextLinter(),
        ImportLinter(),
        AgentLinter(),
        OrchestrationLinter(),
        ExceptionLinter(),
        LoopScopingLinter(),
        TemplateControlStructureLinter(),
        HitlNestedContextLinter(),
    ]
    errors: list[ValidationFinding] = []
    warnings: list[ValidationFinding] = []

    # Run for the current pipeline
    for lin in linters:
        try:
            for finding in lin.analyze(pipeline) or []:
                if finding.severity == "error":
                    errors.append(finding)
                else:
                    warnings.append(finding)
        except Exception:
            continue

    # Recurse into nested pipelines within steps
    for st in getattr(pipeline, "steps", []) or []:
        # Handle ParallelStep and ConditionalStep branches
        if hasattr(st, "branches") and isinstance(st.branches, dict):
            for bp in st.branches.values():
                if hasattr(bp, "steps"):
                    child_report = run_linters(bp, visited)
                    errors.extend(child_report.errors)
                    warnings.extend(child_report.warnings)

        # Handle LoopStep body
        if hasattr(st, "loop_body_pipeline") and st.loop_body_pipeline:
            child_report = run_linters(st.loop_body_pipeline, visited)
            errors.extend(child_report.errors)
            warnings.extend(child_report.warnings)

        # Handle StateMachineStep states
        if hasattr(st, "states") and isinstance(st.states, dict):
            for sp in st.states.values():
                if hasattr(sp, "steps"):
                    child_report = run_linters(sp, visited)
                    errors.extend(child_report.errors)
                    warnings.extend(child_report.warnings)

        # Handle ConditionalStep default branch
        if hasattr(st, "default_branch_pipeline") and st.default_branch_pipeline:
            # Check if it's a pipeline
            dbp = st.default_branch_pipeline
            if hasattr(dbp, "steps"):
                child_report = run_linters(dbp, visited)
                errors.extend(child_report.errors)
                warnings.extend(child_report.warnings)

    return ValidationReport(errors=errors, warnings=warnings)

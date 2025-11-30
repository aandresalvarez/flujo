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


def run_linters(pipeline: Any) -> ValidationReport:
    """Run linters and return a ValidationReport (always-on)."""
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
    for lin in linters:
        try:
            for finding in lin.analyze(pipeline) or []:
                if finding.severity == "error":
                    errors.append(finding)
                else:
                    warnings.append(finding)
        except Exception:
            continue

    return ValidationReport(errors=errors, warnings=warnings)

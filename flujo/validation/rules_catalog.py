from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class RuleInfo:
    id: str
    title: str
    description: str
    default_severity: str  # "error" | "warning"
    help_uri: Optional[str] = None


_BASE_URI = "https://aandresalvarez.github.io/flujo/reference/validation_rules/#"

_CATALOG: Dict[str, RuleInfo] = {
    "V-A1": RuleInfo(
        id="V-A1",
        title="Missing agent on simple step",
        description="Simple steps must define an agent to be executable.",
        default_severity="error",
        help_uri=_BASE_URI + "v-a1",
    ),
    "V-A2": RuleInfo(
        id="V-A2",
        title="Type mismatch between steps",
        description="Previous step output type is incompatible with next step input.",
        default_severity="error",
        help_uri=_BASE_URI + "v-a2",
    ),
    "V-A5": RuleInfo(
        id="V-A5",
        title="Unused previous output",
        description="Previous step output not consumed by next step and not merged into context.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-a5",
    ),
    "V-F1": RuleInfo(
        id="V-F1",
        title="Fallback input incompatible",
        description="Fallback step must accept the same input shape as the primary.",
        default_severity="error",
        help_uri=_BASE_URI + "v-f1",
    ),
    "V-P1": RuleInfo(
        id="V-P1",
        title="Parallel context merge conflict",
        description="Potential key conflicts with CONTEXT_UPDATE and no field_mapping.",
        default_severity="error",
        help_uri=_BASE_URI + "v-p1",
    ),
    "V-P2": RuleInfo(
        id="V-P2",
        title="Parallel explicit outputs conflict",
        description="Multiple branches map outputs to the same parent keys without coordination.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-p2",
    ),
    "V-P3": RuleInfo(
        id="V-P3",
        title="Parallel branch input heterogeneity",
        description="Branches should expect uniform input shape for determinism.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-p3",
    ),
    "V-S1": RuleInfo(
        id="V-S1",
        title="JSON schema structure issue",
        description="Basic issues like array without items or misplaced required.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-s1",
    ),
    "V-S2": RuleInfo(
        id="V-S2",
        title="Structured output stringified downstream",
        description="Agent declares structured output but the next step appears to stringify it.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-s2",
    ),
    "V-S3": RuleInfo(
        id="V-S3",
        title="Schema uses type=string",
        description="Awareness: pure string schema may limit downstream structure.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-s3",
    ),
    "V-L1": RuleInfo(
        id="V-L1",
        title="Loop exit coverage",
        description="Loop body/mappers may not allow exit condition to be satisfied.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-l1",
    ),
    "V-EX1": RuleInfo(
        id="V-EX1",
        title="Control flow exception handling",
        description="Custom skills must re-raise control flow exceptions (PausedException, PipelineAbortSignal, InfiniteRedirectError)",
        default_severity="warning",
        help_uri=_BASE_URI + "v-ex1",
    ),
    "V-SM1": RuleInfo(
        id="V-SM1",
        title="State machine unreachable end",
        description="No path from start_state to any end state.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-sm1",
    ),
    "V-C1": RuleInfo(
        id="V-C1",
        title="updates_context without mergeable output",
        description=(
            "Step sets updates_context but returns a non-mergeable value (not a dict or PipelineResult)."
        ),
        default_severity="warning",
        help_uri=_BASE_URI + "v-c1",
    ),
    "V-C2": RuleInfo(
        id="V-C2",
        title="Scratchpad shape conflict",
        description=(
            "Mapping a value into the scratchpad root may overwrite expected object shape."
        ),
        default_severity="warning",
        help_uri=_BASE_URI + "v-c2",
    ),
    "V-C3": RuleInfo(
        id="V-C3",
        title="Large literal in template",
        description=(
            "Template embeds an extremely large constant; consider referencing by key or file."
        ),
        default_severity="warning",
        help_uri=_BASE_URI + "v-c3",
    ),
    "V-I1": RuleInfo(
        id="V-I1",
        title="Import missing",
        description="Referenced child blueprint path cannot be resolved.",
        default_severity="error",
        help_uri=_BASE_URI + "v-i1",
    ),
    "V-I2": RuleInfo(
        id="V-I2",
        title="Import outputs mapping sanity",
        description="Parent mapping root appears invalid (e.g., unknown root).",
        default_severity="warning",
        help_uri=_BASE_URI + "v-i2",
    ),
    "V-I3": RuleInfo(
        id="V-I3",
        title="Import cycle detected",
        description="Detected a cycle in the import graph.",
        default_severity="error",
        help_uri=_BASE_URI + "v-i3",
    ),
    "V-I4": RuleInfo(
        id="V-I4",
        title="Aggregated child findings",
        description="Parent import step aggregates errors/warnings from the child blueprint.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-i4",
    ),
    "V-I5": RuleInfo(
        id="V-I5",
        title="Input projection coherence",
        description="Parent→child input projection may not match child's expected input shape.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-i5",
    ),
    "V-T1": RuleInfo(
        id="V-T1",
        title="previous_step.output misuse",
        description="previous_step is a raw value and has no .output attribute.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t1",
    ),
    "V-T2": RuleInfo(
        id="V-T2",
        title="'this' outside map body",
        description="'this' is defined only inside map bodies.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t2",
    ),
    "V-T3": RuleInfo(
        id="V-T3",
        title="Unknown/disabled template filter",
        description="Filter not enabled in settings; may be ignored or fail.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t3",
    ),
    "V-T4": RuleInfo(
        id="V-T4",
        title="Unknown step proxy",
        description="Template references steps.<name> that is not a prior step.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t4",
    ),
    "V-T5": RuleInfo(
        id="V-T5",
        title="Missing prior model field",
        description="Template references previous_step.<field> not present on the prior step's model.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t5",
    ),
    "V-T6": RuleInfo(
        id="V-T6",
        title="Non-JSON where JSON expected",
        description="Templated input looks like JSON but is not valid JSON while the consumer expects JSON.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-t6",
    ),
    "V-A6": RuleInfo(
        id="V-A6",
        title="Unknown agent id/import path",
        description="Unresolvable agent id or Python import path in step.agent.",
        default_severity="error",
        help_uri=_BASE_URI + "v-a6",
    ),
    "V-A7": RuleInfo(
        id="V-A7",
        title="Invalid max_retries/timeout coercion",
        description="Agent controls could not be coerced to integers; using defaults.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-a7",
    ),
    "V-A8": RuleInfo(
        id="V-A8",
        title="Structured output with non-JSON response mode",
        description=(
            "Agent declares structured output but step/provider is configured for a non-JSON response mode."
        ),
        default_severity="warning",
        help_uri=_BASE_URI + "v-a8",
    ),
    "V-CTX1": RuleInfo(
        id="V-CTX1",
        title="Missing context isolation in loop/parallel",
        description="Loop and parallel steps with custom skills should use ContextManager.isolate() to ensure idempotency",
        default_severity="warning",
        help_uri=_BASE_URI + "v-ctx1",
    ),
    "V-CF1": RuleInfo(
        id="V-CF1",
        title="Unconditional infinite loop",
        description="Loop has no exit condition or max_loops, creating potential infinite loop",
        default_severity="warning",
        help_uri=_BASE_URI + "v-cf1",
    ),
    "LOOP-001": RuleInfo(
        id="LOOP-001",
        title="Step reference in loop body",
        description="Step reference via steps['name'] inside loop body may not work as expected due to scoping",
        default_severity="warning",
        help_uri=_BASE_URI + "loop-001",
    ),
    "TEMPLATE-001": RuleInfo(
        id="TEMPLATE-001",
        title="Unsupported Jinja2 control structure",
        description="Jinja2 control structures ({% %}) are not supported in Flujo templates. Use expressions {{ }} and filters | instead",
        default_severity="error",
        help_uri=_BASE_URI + "template-001",
    ),
    "HITL-NESTED-001": RuleInfo(
        id="HITL-NESTED-001",
        title="HITL in nested context (CRITICAL)",
        description="HITL step in nested context (loop/conditional) will be SILENTLY SKIPPED at runtime, causing data loss. This is a known limitation.",
        default_severity="error",
        help_uri=_BASE_URI + "hitl-nested-001",
    ),
    # Legacy alias for backward compatibility
    "WARN-HITL-001": RuleInfo(
        id="WARN-HITL-001",
        title="HITL in nested context (DEPRECATED - use HITL-NESTED-001)",
        description="Legacy rule ID. Use HITL-NESTED-001 instead.",
        default_severity="error",
        help_uri=_BASE_URI + "hitl-nested-001",
    ),
}


def get_rule(rule_id: str) -> Optional[RuleInfo]:
    if not rule_id:
        return None
    return _CATALOG.get(rule_id.upper())

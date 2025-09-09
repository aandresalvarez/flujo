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
    "V-SM1": RuleInfo(
        id="V-SM1",
        title="State machine unreachable end",
        description="No path from start_state to any end state.",
        default_severity="warning",
        help_uri=_BASE_URI + "v-sm1",
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
}


def get_rule(rule_id: str) -> Optional[RuleInfo]:
    if not rule_id:
        return None
    return _CATALOG.get(rule_id.upper())

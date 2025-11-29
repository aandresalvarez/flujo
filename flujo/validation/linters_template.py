from __future__ import annotations

import os
import re
from typing import Any, Iterable

from ..domain.pipeline_validation import ValidationFinding
from ..utils.prompting import _get_enabled_filters
from .linters_base import BaseLinter, _override_severity, logfire


class TemplateLinter(BaseLinter):
    """Template-related lints: V-T1..V-T6."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []
        for idx, step in enumerate(steps):
            try:
                meta = getattr(step, "meta", None)
                if not isinstance(meta, dict):
                    continue
                templ = meta.get("templated_input")
                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}].input"
                fpath = (yloc or {}).get("file")
                line = (yloc or {}).get("line")
                col = (yloc or {}).get("column")
                has_tokens = isinstance(templ, str) and ("{{" in templ and "}}" in templ)

                if not isinstance(templ, str):
                    continue

                # V-T1: previous_step.output misuse
                if has_tokens and re.search(r"\bprevious_step\s*\.\s*output\b", templ):
                    sev = _override_severity("V-T1", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-T1",
                                severity=sev,
                                message=(
                                    "Template references previous_step.output, but previous_step is the raw value and has no .output attribute."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=(
                                    "Prefer using steps.<previous_step_name>.output | tojson, or use previous_step | tojson for raw value."
                                ),
                                location_path=loc_path,
                                file=fpath,
                                line=line,
                                column=col,
                            )
                        )

                # V-T2: 'this' outside map bodies (heuristic)
                if has_tokens and re.search(r"\bthis\b", templ):
                    sev = _override_severity("V-T2", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-T2",
                                severity=sev,
                                message=(
                                    "Template references 'this' outside a known map body context."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=(
                                    "Use 'this' only inside map bodies, or bind a variable explicitly."
                                ),
                                location_path=loc_path,
                                file=fpath,
                                line=line,
                                column=col,
                            )
                        )

                # V-T3: Unknown/disabled filters
                if has_tokens:
                    try:
                        enabled = {s.lower() for s in _get_enabled_filters()}
                    except Exception:
                        enabled = {"join", "upper", "lower", "length", "tojson"}
                    for m in re.finditer(r"\|\s*([a-zA-Z_][a-zA-Z0-9_]*)", templ):
                        fname = m.group(1).lower()
                        if fname not in enabled:
                            sev = _override_severity("V-T3", "warning")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-T3",
                                        severity=sev,
                                        message=f"Unknown or disabled template filter: {fname}",
                                        step_name=getattr(step, "name", None),
                                        suggestion=(
                                            "Add to [settings.enabled_template_filters] in flujo.toml or remove/misspelling fix."
                                        ),
                                        location_path=loc_path,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                    )
                                )

                # V-T4: Unknown step proxy name in steps.<name>
                if has_tokens:
                    prior_names = {getattr(s, "name", "") for s in steps[:idx]}
                    is_ci = os.getenv("CI") == "true"
                    if is_ci:
                        logfire.debug(f"[V-T4] Checking template: {templ[:100]}")
                        logfire.debug(f"[V-T4] Prior step names: {prior_names}")

                    for sm in re.finditer(r"steps\.([A-Za-z0-9_]+)\b", templ):
                        ref = sm.group(1)
                        if is_ci:
                            logfire.debug(f"[V-T4] Found reference: steps.{ref}")
                            logfire.debug(
                                f"[V-T4] ref={ref!r}, in prior_names={ref in prior_names}"
                            )

                        if ref and ref not in prior_names:
                            sev = _override_severity("V-T4", "warning")
                            if is_ci:
                                logfire.debug(f"[V-T4] Severity for V-T4: {sev}")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-T4",
                                        severity=sev,
                                        message=f"Template references steps.{ref} which is not a prior step.",
                                        step_name=getattr(step, "name", None),
                                        suggestion=(
                                            "Correct the step name or ensure the reference points to a prior step."
                                        ),
                                        location_path=loc_path,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                    )
                                )

                # V-T5: Prior model field existence for previous_step.<field>
                if has_tokens and idx > 0:
                    try:

                        def _is_model_type(t: Any) -> bool:
                            try:
                                return isinstance(t, type) and (
                                    hasattr(t, "model_fields") or hasattr(t, "__fields__")
                                )
                            except Exception:
                                return False

                        prev_type = getattr(steps[idx - 1], "__step_output_type__", Any)
                        if prev_type is not None and _is_model_type(prev_type):
                            if hasattr(prev_type, "model_fields"):
                                fields = set(getattr(prev_type, "model_fields", {}).keys())
                            else:
                                fields = set(getattr(prev_type, "__fields__", {}).keys())
                            seen_missing: set[str] = set()
                            try:
                                comp = "".join(
                                    ch for ch in templ if ch not in (" ", "\t", "\n", "\r")
                                )
                                key = "previous_step."
                                start = 0
                                while True:
                                    i2 = comp.find(key, start)
                                    if i2 == -1:
                                        break
                                    j2 = i2 + len(key)
                                    fld_chars: list[str] = []
                                    while j2 < len(comp) and (
                                        comp[j2].isalnum() or comp[j2] == "_"
                                    ):
                                        fld_chars.append(comp[j2])
                                        j2 += 1
                                    fld = "".join(fld_chars)
                                    if fld and fld != "output" and fld not in fields:
                                        seen_missing.add(fld)
                                    start = j2
                            except Exception:
                                pass
                            for fld in sorted(seen_missing):
                                sev = _override_severity("V-T5", "warning")
                                if sev is not None:
                                    out.append(
                                        ValidationFinding(
                                            rule_id="V-T5",
                                            severity=sev,
                                            message=(
                                                f"Template references previous_step.{fld} but field is not present on prior model {getattr(prev_type, '__name__', prev_type)}."
                                            ),
                                            step_name=getattr(step, "name", None),
                                            suggestion=(
                                                "Use an existing field or adapt the prior step to emit the needed attribute."
                                            ),
                                            location_path=loc_path,
                                            file=fpath,
                                            line=line,
                                            column=col,
                                        )
                                    )
                    except Exception:
                        pass

                # V-T6: Non-JSON where JSON expected
                def _expects_json(t: Any) -> bool:
                    try:
                        from typing import get_origin as _go

                        org = _go(t)
                    except Exception:
                        org = None
                    if t is dict or org is dict:
                        return True
                    try:
                        from pydantic import BaseModel as _PM

                        return isinstance(t, type) and issubclass(t, _PM)
                    except Exception:
                        return False

                in_type = getattr(step, "__step_input_type__", Any)
                if _expects_json(in_type):
                    if has_tokens:
                        cleaned = re.sub(r"\{\{.*?\}\}", "null", templ).strip()
                        if (cleaned.startswith("{") and cleaned.endswith("}")) or (
                            cleaned.startswith("[") and cleaned.endswith("]")
                        ):
                            import json as _json

                            try:
                                _json.loads(cleaned)
                            except Exception:
                                sev = _override_severity("V-T6", "warning")
                                if sev is not None:
                                    out.append(
                                        ValidationFinding(
                                            rule_id="V-T6",
                                            severity=sev,
                                            message=(
                                                "Templated input appears to be JSON but is not valid JSON; consumer expects JSON."
                                            ),
                                            step_name=getattr(step, "name", None),
                                            suggestion=(
                                                "Ensure valid JSON or use the tojson filter on variables."
                                            ),
                                            location_path=loc_path,
                                            file=fpath,
                                            line=line,
                                            column=col,
                                        )
                                    )
                    else:
                        s = templ.strip()
                        if (s.startswith("{") and s.endswith("}")) or (
                            s.startswith("[") and s.endswith("]")
                        ):
                            import json as _json

                            try:
                                _json.loads(s)
                            except Exception:
                                sev = _override_severity("V-T6", "warning")
                                if sev is not None:
                                    out.append(
                                        ValidationFinding(
                                            rule_id="V-T6",
                                            severity=sev,
                                            message=(
                                                "Input appears to be JSON but is not valid JSON; consumer expects JSON."
                                            ),
                                            step_name=getattr(step, "name", None),
                                            suggestion=(
                                                "Ensure valid JSON or use the tojson filter on variables."
                                            ),
                                            location_path=loc_path,
                                            file=fpath,
                                            line=line,
                                            column=col,
                                        )
                                    )
            except Exception:
                continue
        return out

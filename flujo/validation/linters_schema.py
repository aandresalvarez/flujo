from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional

from ..domain.pipeline_validation import ValidationFinding
from .linters_base import BaseLinter, _override_severity


class SchemaLinter(BaseLinter):
    """Surface agent schema warnings and V-S3 awareness."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []
        for idx, step in enumerate(steps):
            try:
                ag = getattr(step, "agent", None)
                if ag is None:
                    continue
                warns = getattr(ag, "_schema_warnings", None) or []
                meta = getattr(step, "meta", None)
                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}].agent.output_schema"
                for msg in warns:
                    sev = _override_severity("V-S1", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-S1",
                                severity=sev,
                                message=str(msg),
                                step_name=getattr(step, "name", None),
                                location_path=loc_path,
                                file=(yloc or {}).get("file"),
                                line=(yloc or {}).get("line"),
                                column=(yloc or {}).get("column"),
                            )
                        )
                schema = getattr(ag, "_declared_output_schema", None)
                if schema is None:
                    try:
                        candidate = getattr(ag, "output_schema", None)
                        if isinstance(candidate, dict):
                            schema = candidate
                    except Exception:
                        schema = None
                if (
                    isinstance(schema, dict)
                    and str(schema.get("type", "")).strip().lower() == "string"
                ):
                    sev = _override_severity("V-S3", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-S3",
                                severity=sev,
                                message=(
                                    "Agent output_schema uses type=string; consider structured schema if downstream expects objects."
                                ),
                                step_name=getattr(step, "name", None),
                                location_path=loc_path,
                                file=(yloc or {}).get("file"),
                                line=(yloc or {}).get("line"),
                                column=(yloc or {}).get("column"),
                            )
                        )
            except Exception:
                continue
        try:
            for i in range(1, len(steps)):
                prev = steps[i - 1]
                cur = steps[i]
                try:
                    ag_prev = getattr(prev, "agent", None)
                    prev_schema = getattr(ag_prev, "_declared_output_schema", None)
                    if prev_schema is None:
                        try:
                            candidate = getattr(ag_prev, "output_schema", None)
                            if isinstance(candidate, dict):
                                prev_schema = candidate
                        except Exception:
                            prev_schema = None
                    prev_structured = (isinstance(prev_schema, dict) and bool(prev_schema)) or bool(
                        getattr(ag_prev, "_schema_warnings", None)
                    )
                    if not prev_structured:
                        continue
                    next_in = getattr(cur, "__step_input_type__", Any)
                    next_out = getattr(cur, "__step_output_type__", Any)
                    is_str_in = next_in is str
                    is_str_out = next_out is str
                    agent_id = None
                    try:
                        agent_id = getattr(cur.agent, "__name__", None) or getattr(
                            cur.agent, "model_id", None
                        )
                        if isinstance(cur.agent, str):
                            agent_id = cur.agent
                    except Exception:
                        agent_id = None
                    is_stringify_agent = (
                        isinstance(agent_id, str) and "stringify" in agent_id.lower()
                    )
                    templated_input = None
                    try:
                        meta_cur = getattr(cur, "meta", None)
                        if isinstance(meta_cur, dict):
                            templated_input = meta_cur.get("templated_input")
                    except Exception:
                        templated_input = None
                    templated_hits_prev = False
                    try:
                        if isinstance(templated_input, str):
                            prev_name = getattr(prev, "name", "")
                            if prev_name and f"steps.{prev_name}.output" in templated_input:
                                templated_hits_prev = True
                    except Exception:
                        templated_hits_prev = False
                    if is_str_in or is_str_out or is_stringify_agent or templated_hits_prev:
                        meta = getattr(cur, "meta", None)
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get("path") or f"steps[{i}]"
                        sev = _override_severity("V-S2", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-S2",
                                    severity=sev,
                                    message=(
                                        f"Structured output from '{getattr(prev, 'name', None)}' appears to be stringified in next step '{getattr(cur, 'name', None)}'."
                                    ),
                                    step_name=getattr(cur, "name", None),
                                    suggestion=(
                                        "If you need fields, map them directly from the object; otherwise suppress if intended."
                                    ),
                                    location_path=loc_path,
                                    file=(yloc or {}).get("file"),
                                    line=(yloc or {}).get("line"),
                                    column=(yloc or {}).get("column"),
                                )
                            )
                except Exception:
                    continue
        except Exception:
            pass
        return out


class ContextLinter(BaseLinter):
    """Context-related lints: V-C1, V-C2, V-C3."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = list(getattr(pipeline, "steps", []) or [])

        try:
            from typing import get_origin as _go

            _PR: Optional[type[Any]] = None
            try:
                from ..domain.models import PipelineResult as _PR
            except Exception:  # pragma: no cover - defensive
                _PR = None

            for i, step in enumerate(steps):
                prev_step = steps[i - 1] if i > 0 else None
                if prev_step is None:
                    continue
                try:
                    if not bool(getattr(prev_step, "updates_context", False)):
                        continue
                    pout = getattr(prev_step, "__step_output_type__", Any)
                    is_mergeable = False
                    try:
                        is_mergeable = (pout is dict) or (_go(pout) is dict)
                    except Exception:
                        is_mergeable = pout is dict
                    if not is_mergeable and _PR is not None:
                        try:
                            is_mergeable = isinstance(pout, type) and issubclass(pout, _PR)
                        except Exception:
                            is_mergeable = False
                    if is_mergeable:
                        continue

                    def _consumes_prev(_step: Any) -> bool:
                        try:
                            in_t = getattr(_step, "__step_input_type__", Any)
                            return not (
                                in_t is Any or in_t is object or in_t is None or in_t is type(None)
                            )
                        except Exception:
                            return False

                    has_explicit_outputs_map = False
                    try:
                        from ..domain.dsl.import_step import ImportStep as _IS  # lazy import

                        if isinstance(prev_step, _IS):
                            outputs_map = getattr(prev_step, "outputs", None)
                            has_explicit_outputs_map = isinstance(outputs_map, list)
                    except Exception:
                        has_explicit_outputs_map = False

                    escalate = (not _consumes_prev(step)) and (not has_explicit_outputs_map)
                    out.append(
                        ValidationFinding(
                            rule_id="V-C1",
                            severity="error" if escalate else "warning",
                            message=(
                                f"Step '{getattr(prev_step, 'name', None)}' sets updates_context=True but its output type is not mergeable into context."
                            ),
                            step_name=getattr(prev_step, "name", None),
                            suggestion=(
                                "Emit a dict-like object or PipelineResult, or map specific fields via outputs."
                            ),
                        )
                    )
                except Exception:
                    continue
        except Exception:
            pass

        try:
            from ..domain.dsl.import_step import ImportStep as _IS2

            for st in steps:
                try:
                    if not isinstance(st, _IS2):
                        continue
                    outs = getattr(st, "outputs", None)
                    if not isinstance(outs, list):
                        continue
                    for om in outs:
                        try:
                            parent_path = str(getattr(om, "parent", ""))
                        except Exception:
                            parent_path = ""
                        if parent_path.strip() == "scratchpad":
                            sev = _override_severity("V-C2", "error")
                            out.append(
                                ValidationFinding(
                                    rule_id="V-C2",
                                    severity=sev or "error",
                                    message=(
                                        "Mapping into 'scratchpad' root is no longer supported; scratchpad has been removed."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Change parent to import_artifacts.<key> or another typed context field."
                                    ),
                                    location_path="steps[].config.outputs",
                                )
                            )
                except Exception:
                    continue
        except Exception:
            pass

        try:
            THRESH_DEFAULT = 50000
            try:
                _th = int(os.getenv("FLUJO_VALIDATE_LARGE_LITERAL_THRESHOLD", str(THRESH_DEFAULT)))
            except Exception:
                _th = THRESH_DEFAULT
            for idx, st in enumerate(steps):
                try:
                    meta = getattr(st, "meta", None)
                    templ = meta.get("templated_input") if isinstance(meta, dict) else None
                    if not isinstance(templ, str):
                        continue
                    yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                    loc_path = (yloc or {}).get("path") or f"steps[{idx}].input"
                    if len(templ) >= _th:
                        out.append(
                            ValidationFinding(
                                rule_id="V-C3",
                                severity="warning",
                                message=(
                                    f"Templated input string is very large (>= {_th} chars); consider referencing external data."
                                ),
                                step_name=getattr(st, "name", None),
                                location_path=loc_path,
                                file=(yloc or {}).get("file"),
                                line=(yloc or {}).get("line"),
                                column=(yloc or {}).get("column"),
                            )
                        )
                        continue
                    for m in re.finditer(r"\{\{(.*?)\}\}", templ, flags=re.S):
                        inner = m.group(1) or ""
                        mm = re.search(r"(['\"])(.*?)\1\s*\*\s*(\d{1,})", inner, re.S)
                        if mm:
                            lit = mm.group(2) or ""
                            try:
                                count = int(mm.group(3))
                            except Exception:
                                count = 1
                            est = len(lit) * max(count, 1)
                            if est >= _th:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-C3",
                                        severity="warning",
                                        message=(
                                            f"Template constructs a very large string (~{est} chars) via repetition."
                                        ),
                                        step_name=getattr(st, "name", None),
                                        location_path=loc_path,
                                        file=(yloc or {}).get("file"),
                                        line=(yloc or {}).get("line"),
                                        column=(yloc or {}).get("column"),
                                    )
                                )
                                break
                except Exception:
                    continue
        except Exception:
            pass

        return out

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional

from ..domain.pipeline_validation import ValidationFinding
from .linters_base import BaseLinter, _override_severity


class ImportLinter(BaseLinter):
    """Import-related lints that do not require recursive validation."""

    _ALLOWED_PARENT_ROOTS: ClassVar[set[str]] = {
        "import_artifacts",
        "step_outputs",
        "steps",
        "command_log",
        "hitl_history",
        "conversation_history",
        "yaml_text",
        "generated_yaml",
        "run_id",
    }

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        try:
            from ..domain.dsl import import_step as _import_step_module

            _IS = getattr(_import_step_module, "ImportStep")
        except Exception:  # pragma: no cover - if import system changes
            return out

        parent_path = getattr(pipeline, "_source_file", None)
        for st in getattr(pipeline, "steps", []) or []:
            try:
                if not isinstance(st, _IS):
                    continue
                try:
                    ch = getattr(st, "pipeline", None)
                    ch_path = getattr(ch, "_source_file", None)
                    if isinstance(ch_path, str):
                        import os as _os

                        if not _os.path.exists(ch_path):
                            sev = _override_severity("V-I1", "error")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-I1",
                                        severity=sev,
                                        message=(f"Import source file not found: {ch_path}"),
                                        step_name=getattr(st, "name", None),
                                        suggestion=(
                                            "Ensure the referenced child file exists and the path is correct relative to the parent YAML."
                                        ),
                                    )
                                )
                except Exception:
                    pass
                try:
                    ch = getattr(st, "pipeline", None)
                    ch_path = getattr(ch, "_source_file", None)
                    if isinstance(parent_path, str) and isinstance(ch_path, str):
                        import os as _os

                        if _os.path.realpath(parent_path) == _os.path.realpath(ch_path):
                            sev = _override_severity("V-I3", "error")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-I3",
                                        severity=sev,
                                        message=(
                                            "Cyclic import detected: parent and child refer to the same file."
                                        ),
                                        step_name=getattr(st, "name", None),
                                    )
                                )
                except Exception:
                    pass
                outs = getattr(st, "outputs", None)
                if isinstance(outs, list):
                    for om in outs:
                        try:
                            parent_path = str(getattr(om, "parent", ""))
                        except Exception:
                            parent_path = ""
                        if not parent_path:
                            continue
                        root = parent_path.split(".", 1)[0]
                        if root == "scratchpad":
                            out.append(
                                ValidationFinding(
                                    rule_id="V-I2",
                                    severity="error",
                                    message=(
                                        f"Import outputs mapping parent path '{parent_path}' targets removed scratchpad."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Map into import_artifacts.<key> or another typed context field."
                                    ),
                                    location_path="steps[].config.outputs",
                                )
                            )
                            continue
                        if root not in self._ALLOWED_PARENT_ROOTS:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-I2",
                                    severity="warning",
                                    message=(
                                        f"Import outputs mapping parent path '{parent_path}' has an unknown root; consider mapping under import_artifacts or a known context field."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Use import_artifacts.<key> for transient fields or ensure the root is a valid context field."
                                    ),
                                    location_path="steps[].config.outputs",
                                )
                            )
                try:
                    input_to = str(getattr(st, "input_to", "initial_prompt")).strip().lower()
                    meta_step = getattr(st, "meta", {}) or {}
                    t_in = meta_step.get("templated_input")
                    if input_to == "initial_prompt" and isinstance(t_in, dict):
                        out.append(
                            ValidationFinding(
                                rule_id="V-I5",
                                severity="warning",
                                message=(
                                    "Import projects an object literal to initial_prompt; consider projecting to import_artifacts or both."
                                ),
                                step_name=getattr(st, "name", None),
                                suggestion=(
                                    "Use input_to=import_artifacts or both to pass structured input."
                                ),
                            )
                        )
                except Exception:
                    pass
                try:
                    inherit_conversation = bool(getattr(st, "inherit_conversation", True))
                    outs2 = getattr(st, "outputs", None)
                    if isinstance(outs2, list) and not inherit_conversation:
                        for om in outs2:
                            try:
                                ch = str(getattr(om, "child", ""))
                                pr = str(getattr(om, "parent", ""))
                            except Exception:
                                ch = pr = ""
                            for path in (ch, pr):
                                root = path.split(".", 1)[0]
                                if root in {"conversation_history", "hitl_history"}:
                                    out.append(
                                        ValidationFinding(
                                            rule_id="V-I6",
                                            severity="warning",
                                            message=(
                                                "Import maps conversation-related fields but inherit_conversation=False; continuity may be lost."
                                            ),
                                            step_name=getattr(st, "name", None),
                                            suggestion=(
                                                "Set inherit_conversation=True or avoid mapping conversation history across the boundary."
                                            ),
                                        )
                                    )
                                    raise StopIteration
                except StopIteration:
                    pass
                except Exception:
                    pass
            except Exception:
                continue
        return out


class AgentLinter(BaseLinter):
    """Agent/provider-related lints."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        def _has_structured_intent(agent_obj: Any, step_obj: Any) -> bool:
            try:
                schema = getattr(agent_obj, "_declared_output_schema", None)
                if isinstance(schema, dict) and len(schema) > 0:
                    return True
            except Exception:
                pass
            try:
                meta = getattr(step_obj, "meta", {}) or {}
                proc = meta.get("processing", {}) if isinstance(meta, dict) else None
                if isinstance(proc, dict) and isinstance(proc.get("schema"), dict):
                    return True
            except Exception:
                pass
            return False

        def _is_json_mode(step_obj: Any, agent_obj: Any) -> bool:
            try:
                rf = getattr(agent_obj, "_structured_output_config", None)
                if isinstance(rf, dict):
                    t = str(rf.get("type", "")).strip().lower()
                    if t in {"json_object", "json_schema"}:
                        return True
            except Exception:
                pass

            mode_val: Optional[str] = None
            try:
                meta = getattr(step_obj, "meta", {}) or {}
                proc = meta.get("processing", {}) if isinstance(meta, dict) else None
                if isinstance(proc, dict):
                    mv = proc.get("structured_output")
                    if isinstance(mv, str):
                        mode_val = mv.strip().lower()
            except Exception:
                mode_val = None

            if not mode_val:
                try:
                    from ..infra.config_manager import get_aros_config as _get_aros

                    mode_val = _get_aros().structured_output_default.strip().lower()
                except Exception:
                    mode_val = "off"

            return mode_val in {"auto", "openai_json"}

        for idx, st in enumerate(steps):
            try:
                ag = getattr(st, "agent", None)
                if ag is None:
                    continue
                try:
                    if isinstance(ag, str):
                        try:
                            from ..domain.blueprint.loader import _import_object as _import_obj

                            _import_obj(ag)
                        except Exception as e:  # noqa: BLE001
                            meta = getattr(st, "meta", None)
                            yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                            loc_path = (yloc or {}).get("path") or f"steps[{idx}].agent"
                            sev = _override_severity("V-A6", "error")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-A6",
                                        severity=sev,
                                        message=(f"Unknown agent id/import path '{ag}': {e}"),
                                        step_name=getattr(st, "name", None),
                                        suggestion=(
                                            "Use a valid 'package.module:attr' or configure a declarative agent under agents.*"
                                        ),
                                        location_path=loc_path,
                                        file=(yloc or {}).get("file"),
                                        line=(yloc or {}).get("line"),
                                        column=(yloc or {}).get("column"),
                                    )
                                )
                except Exception:
                    pass

                try:
                    coerce_warns = getattr(ag, "_coercion_warnings", None)
                    if coerce_warns:
                        meta = getattr(st, "meta", None)
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get("path") or f"steps[{idx}].agent"
                        sev = _override_severity("V-A7", "warning")
                        if sev is not None:
                            for msg in coerce_warns:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-A7",
                                        severity=sev,
                                        message=str(msg),
                                        step_name=getattr(st, "name", None),
                                        location_path=loc_path,
                                        file=(yloc or {}).get("file"),
                                        line=(yloc or {}).get("line"),
                                        column=(yloc or {}).get("column"),
                                    )
                                )
                except Exception:
                    pass

                if _has_structured_intent(ag, st) and not _is_json_mode(st, ag):
                    meta = getattr(st, "meta", None)
                    yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                    loc_path = (yloc or {}).get(
                        "path"
                    ) or f"steps[{idx}].processing.structured_output"
                    sev = _override_severity("V-A8", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-A8",
                                severity=sev,
                                message=(
                                    "Structured output requested (schema present) but step is configured for a non-JSON response mode."
                                ),
                                step_name=getattr(st, "name", None),
                                suggestion=(
                                    "Set processing.structured_output: openai_json (or auto) and provide a schema under processing.schema,"
                                    " or programmatically enable JSON schema on the wrapper."
                                ),
                                location_path=loc_path,
                                file=(yloc or {}).get("file"),
                                line=(yloc or {}).get("line"),
                                column=(yloc or {}).get("column"),
                            )
                        )
            except Exception:
                continue

        return out

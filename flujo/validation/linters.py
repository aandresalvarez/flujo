from __future__ import annotations

import os
import re
from typing import Any, Iterable, ClassVar
import fnmatch
import json
from ..domain.pipeline_validation import ValidationFinding, ValidationReport
from ..infra.telemetry import logfire
from threading import RLock

# Conditional imports for TOML support (py311+ has tomllib, fallback to tomli)
try:
    import tomllib as toml_lib
except ImportError:
    try:
        import tomli as toml_lib  # type: ignore[no-redef]
    except ImportError:
        toml_lib = None  # type: ignore[assignment]

# Import config manager and template utils (used in validation rules)
from ..infra.config_manager import ConfigManager
from ..utils.prompting import _get_enabled_filters

# --- Rule overrides (profile/file/env) for early skip and severity adjustment ---
_OVERRIDE_CACHE: dict[str, str] | None = None
_OVERRIDE_CACHE_LOCK = RLock()


def _load_rule_overrides() -> dict[str, str]:
    """Load rule-id severity overrides from env/config in a thread-safe way.

    Logging is kept at debug level to avoid noise in CI. Any parsing errors are
    non-fatal and are recorded for troubleshooting.
    """
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE
    with _OVERRIDE_CACHE_LOCK:
        if _OVERRIDE_CACHE is not None:
            return _OVERRIDE_CACHE
        mapping: dict[str, str] = {}
        # 1) Env JSON mapping (highest precedence for early-skip)
        try:
            env_json = os.getenv("FLUJO_RULES_JSON")
            if env_json:
                data = json.loads(env_json)
                if isinstance(data, dict):
                    mapping.update({str(k).upper(): str(v).lower() for k, v in data.items()})
        except Exception as e:
            logfire.debug(f"[validate] Invalid FLUJO_RULES_JSON: {e!r}")
        # 2) Env file path mapping
        try:
            rules_file = os.getenv("FLUJO_RULES_FILE")
            if rules_file and os.path.exists(rules_file):
                try:
                    if rules_file.endswith((".json", ".JSON")):
                        with open(rules_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            mapping.update(
                                {str(k).upper(): str(v).lower() for k, v in data.items()}
                            )
                    elif rules_file.endswith((".toml", ".TOML")):
                        if toml_lib is None:
                            logfire.debug("[validate] TOML support not available (install tomli)")
                        else:
                            with open(rules_file, "rb") as f:
                                data = toml_lib.load(f)
                            # Expect { validation = { rules = {"V-T*"="off", ...} } }
                            try:
                                vm = data.get("validation", {}).get("rules", {})
                                if isinstance(vm, dict):
                                    mapping.update(
                                        {str(k).upper(): str(v).lower() for k, v in vm.items()}
                                    )
                            except Exception as e:
                                logfire.debug(
                                    f"[validate] TOML rules parse (validation.rules) failed: {e!r}"
                                )
                except Exception as e:
                    logfire.debug(
                        f"[validate] Failed reading FLUJO_RULES_FILE '{rules_file}': {e!r}"
                    )
        except Exception as e:
            logfire.debug(f"[validate] Error handling FLUJO_RULES_FILE: {e!r}")
        # 3) flujo.toml profile selected via FLUJO_RULES_PROFILE
        try:
            profile = os.getenv("FLUJO_RULES_PROFILE")
            if profile:
                cm = ConfigManager()
                cfg = cm.load_config()
                profiles = getattr(cfg, "validation", None)
                if profiles and getattr(profiles, "profiles", None):
                    raw = profiles.profiles.get(profile)
                    if isinstance(raw, dict):
                        mapping.update({str(k).upper(): str(v).lower() for k, v in raw.items()})
        except Exception as e:
            logfire.debug(f"[validate] Failed loading profile overrides: {e!r}")

        _OVERRIDE_CACHE = mapping
        return mapping


def _override_severity(rule_id: str, default: str) -> str | None:
    """Return overridden severity ('error'/'warning') or None to indicate OFF.

    - Exact match wins; then glob patterns (e.g., 'V-T*').
    - Values 'off' => return None; 'warning'/'error' => return that; unknown => keep default.
    """
    mp = _load_rule_overrides()
    rid = str(rule_id).upper()
    if rid in mp:
        val = mp[rid]
    else:
        val = None
        for pat, sev in mp.items():
            try:
                if fnmatch.fnmatch(rid, pat):
                    val = sev
                    break
            except Exception:
                continue
    if not val:
        return default
    if val == "off":
        return None
    if val in {"warning", "error"}:
        return val
    return default


class BaseLinter:
    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:  # pragma: no cover - interface
        return []


class TemplateLinter(BaseLinter):
    """Template-related lints: V-T1..V-T6.

    - V-T1: previous_step.output misuse
    - V-T2: 'this' outside map body
    - V-T3: Unknown/disabled filters
    - V-T4: Unknown step proxy (steps.<name>)
    - V-T5: Missing prior model field (previous_step.<field>)
    - V-T6: Non-JSON where JSON expected
    """

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
                    # Debug logging for CI
                    import os

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
        # V-S2: Structured output declared then likely stringified downstream (next step)
        try:
            for i in range(1, len(steps)):
                prev = steps[i - 1]
                cur = steps[i]
                try:
                    ag_prev = getattr(prev, "agent", None)
                    prev_schema = getattr(ag_prev, "_declared_output_schema", None)
                    prev_structured = isinstance(prev_schema, dict) and bool(prev_schema)
                    if not prev_structured:
                        continue
                    # Heuristics: next step clearly stringifies
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
                    if is_str_in or is_str_out or is_stringify_agent:
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

        # V-C1: updates_context without mergeable output (+ escalation heuristic)
        try:
            from typing import get_origin as _go

            try:
                from ..domain.models import PipelineResult as _PR
            except Exception:  # pragma: no cover - defensive
                _PR = None  # type: ignore

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

                    # Escalation when next step does not consume prev output (rough heuristic)
                    def _consumes_prev(_step: Any) -> bool:
                        try:
                            in_t = getattr(_step, "__step_input_type__", Any)
                            return not (
                                in_t is Any or in_t is object or in_t is None or in_t is type(None)
                            )
                        except Exception:
                            return False

                    # ImportStep outputs mapping counts as consumption for escalation suppression
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

        # V-C2: scratchpad shape conflicts (ImportStep outputs mapping to scratchpad root)
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
                            out.append(
                                ValidationFinding(
                                    rule_id="V-C2",
                                    severity="warning",
                                    message=(
                                        "Mapping into 'scratchpad' root may assign a non-dict and corrupt shape; map under scratchpad.<key>."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Change parent to 'scratchpad.<key>' or ensure the child value is an object."
                                    ),
                                    location_path="steps[].config.outputs",
                                )
                            )
                except Exception:
                    continue
        except Exception:
            pass

        # V-C3: Large literals in templates
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
                    # Within tokens, check for basic repetition pattern
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


class ImportLinter(BaseLinter):
    """Import-related lints that do not require recursive validation.

    - V-I2: outputs mapping sanity (unknown parent roots)
    - V-I5: input projection coherence (parent-side heuristics)
    - V-I6: inherit conversation consistency
    """

    _ALLOWED_PARENT_ROOTS: ClassVar[set[str]] = {
        "scratchpad",
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
            from ..domain.dsl.import_step import ImportStep as _IS
        except Exception:  # pragma: no cover - if import system changes
            return out

        parent_path = getattr(pipeline, "_source_file", None)
        for st in getattr(pipeline, "steps", []) or []:
            try:
                if not isinstance(st, _IS):
                    continue
                # V-I1: Missing import source file (best-effort) – when child pipeline exposes _source_file
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
                # V-I3: Immediate self-cycle (parent imports a child with the same path)
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
                # V-I2: Outputs mapping sanity – warn on obviously invalid parent roots
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
                        if root not in self._ALLOWED_PARENT_ROOTS:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-I2",
                                    severity="warning",
                                    message=(
                                        f"Import outputs mapping parent path '{parent_path}' has an unknown root; consider mapping under 'scratchpad' or a known context field."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Use scratchpad.<key> for transient fields or ensure the root is a valid context field."
                                    ),
                                    location_path="steps[].config.outputs",
                                )
                            )
                # V-I5: Input projection coherence (parent-side)
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
                                    "Import projects an object literal to initial_prompt; consider projecting to scratchpad or both."
                                ),
                                step_name=getattr(st, "name", None),
                                suggestion=(
                                    "Use input_to=scratchpad or both with input_scratchpad_key to pass structured input."
                                ),
                            )
                        )
                except Exception:
                    pass
                # V-I6: Inherit conversation consistency
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
    """Agent/provider-related lints.

    Implements:
    - V-A6: Unknown agent id/import path (string import resolution)
    - V-A7: Invalid max_retries/timeout coercion (surfaced from agent wrapper)
    - V-A8: Structured output with non-JSON response mode
    """

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        # Helper: determine structured-output intent
        def _has_structured_intent(agent_obj: Any, step_obj: Any) -> bool:
            try:
                # Declarative agents attach declared schema at compile time
                schema = getattr(agent_obj, "_declared_output_schema", None)
                if isinstance(schema, dict) and len(schema) > 0:
                    # Treat any declared schema as structured intent
                    return True
            except Exception:
                pass
            # Also consider step.meta.processing.schema
            try:
                meta = getattr(step_obj, "meta", {}) or {}
                proc = meta.get("processing", {}) if isinstance(meta, dict) else None
                if isinstance(proc, dict) and isinstance(proc.get("schema"), dict):
                    return True
            except Exception:
                pass
            return False

        # Helper: determine effective response mode (JSON vs non-JSON)
        def _is_json_mode(step_obj: Any, agent_obj: Any) -> bool:
            # If wrapper already configured, treat as JSON mode
            try:
                rf = getattr(agent_obj, "_structured_output_config", None)
                if isinstance(rf, dict):
                    t = str(rf.get("type", "")).strip().lower()
                    if t in {"json_object", "json_schema"}:
                        return True
            except Exception:
                pass

            # Inspect processing.structured_output for explicit mode
            mode_val: str | None = None
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
                # Fallback to project default from config manager
                try:
                    from ..infra.config_manager import get_aros_config as _get_aros

                    mode_val = _get_aros().structured_output_default.strip().lower()
                except Exception:
                    mode_val = "off"

            # JSON-capable modes
            return mode_val in {"auto", "openai_json"}

        for idx, st in enumerate(steps):
            try:
                ag = getattr(st, "agent", None)
                if ag is None:
                    continue
                # V-A6: Unknown import path
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

                # V-A7: coercion warnings surfaced from wrapper
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

                # V-A8: structured output vs non‑JSON mode
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


class OrchestrationLinter(BaseLinter):
    """Orchestration-related lints:
    - V-P2: Parallel explicit outputs conflicts
    - V-P3: Parallel branch input heterogeneity
    - V-L1: Loop exit coverage heuristic
    - V-CF1: Unconditional infinite loop heuristic
    - V-CTX1: Missing context isolation in loop/parallel steps with custom skills
    """

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        # Lazy imports to avoid cycles
        try:
            from ..domain.dsl.parallel import ParallelStep as _ParallelStep
        except Exception:
            _ParallelStep = None  # type: ignore
        try:
            from ..domain.dsl.import_step import ImportStep as _ImportStep
        except Exception:
            _ImportStep = None  # type: ignore
        try:
            from ..domain.dsl.loop import LoopStep as _LoopStep
        except Exception:
            _LoopStep = None  # type: ignore

        # V-P1/V-P2/V-P3
        if _ParallelStep is not None:
            for st in steps:
                if not isinstance(st, _ParallelStep):
                    continue
                # V-P1/V-P1-W: Merge conflicts under CONTEXT_UPDATE
                try:
                    from ..domain.dsl.step import MergeStrategy as _MergeStrategy

                    if getattr(st, "merge_strategy", None) == _MergeStrategy.CONTEXT_UPDATE:
                        candidate_fields: set[str] = set()
                        if getattr(st, "context_include_keys", None) is not None:
                            try:
                                candidate_fields.update(getattr(st, "context_include_keys"))
                            except Exception:
                                pass

                        # If no hints and no field_mapping: warn when multiple branches
                        if not candidate_fields and getattr(st, "field_mapping", None) is None:
                            try:
                                if len(getattr(st, "branches", {}) or {}) > 1:
                                    out.append(
                                        ValidationFinding(
                                            rule_id="V-P1-W",
                                            severity="warning",
                                            message=(
                                                f"ParallelStep '{getattr(st, 'name', None)}' uses CONTEXT_UPDATE without field_mapping; potential merge conflicts may occur."
                                            ),
                                            step_name=getattr(st, "name", None),
                                            suggestion=(
                                                "Provide a field_mapping per-branch or pick an explicit merge strategy like OVERWRITE or ERROR_ON_CONFLICT."
                                            ),
                                        )
                                    )
                            except Exception:
                                pass
                        else:
                            fm = getattr(st, "field_mapping", None)
                            if isinstance(fm, dict):
                                field_to_branches: dict[str, list[str]] = {}
                                for bname, fields in fm.items():
                                    try:
                                        for f in fields:
                                            field_to_branches.setdefault(str(f), []).append(
                                                str(bname)
                                            )
                                    except Exception:
                                        continue
                                for f, bnames in field_to_branches.items():
                                    try:
                                        if len(bnames) > 1 and not bool(
                                            getattr(st, "ignore_branch_names", False)
                                        ):
                                            out.append(
                                                ValidationFinding(
                                                    rule_id="V-P1",
                                                    severity="error",
                                                    message=(
                                                        f"Context merge conflict risk for key '{f}' in ParallelStep '{getattr(st, 'name', None)}': declared by branches {bnames}."
                                                    ),
                                                    step_name=getattr(st, "name", None),
                                                    suggestion=(
                                                        "Set an explicit MergeStrategy (e.g., OVERWRITE) or ensure only one branch writes each field via field_mapping."
                                                    ),
                                                )
                                            )
                                    except Exception:
                                        continue
                            else:
                                # No explicit field_mapping but candidate fields exist
                                try:
                                    if (
                                        candidate_fields
                                        and len(getattr(st, "branches", {}) or {}) > 1
                                    ):
                                        out.append(
                                            ValidationFinding(
                                                rule_id="V-P1",
                                                severity="error",
                                                message=(
                                                    f"ParallelStep '{getattr(st, 'name', None)}' may merge conflicting context fields {sorted(candidate_fields)} using CONTEXT_UPDATE without field_mapping."
                                                ),
                                                step_name=getattr(st, "name", None),
                                                suggestion=(
                                                    "Provide field_mapping for conflicting keys or choose OVERWRITE/ERROR_ON_CONFLICT explicitly."
                                                ),
                                            )
                                        )
                                except Exception:
                                    pass
                except Exception:
                    pass
                # V-P2: explicit outputs mapping conflicts across branches
                try:
                    parent_target_to_branches: dict[str, set[str]] = {}
                    for bname, bp in (getattr(st, "branches", {}) or {}).items():
                        try:
                            for _st in getattr(bp, "steps", []) or []:
                                if _ImportStep is not None and isinstance(_st, _ImportStep):
                                    outs = getattr(_st, "outputs", None)
                                    if isinstance(outs, list):
                                        for om in outs:
                                            try:
                                                parent_path = str(getattr(om, "parent", "") or "")
                                            except Exception:
                                                parent_path = ""
                                            if not parent_path:
                                                continue
                                            parent_target_to_branches.setdefault(
                                                parent_path, set()
                                            ).add(str(bname))
                        except Exception:
                            continue
                    conflicts = {k: v for k, v in parent_target_to_branches.items() if len(v) > 1}
                    if conflicts:
                        out.append(
                            ValidationFinding(
                                rule_id="V-P2",
                                severity="warning",
                                message=(
                                    f"ParallelStep '{getattr(st, 'name', None)}' branches map to the same parent keys: "
                                    + ", ".join(
                                        f"{k} <- {sorted(list(v))}" for k, v in conflicts.items()
                                    )
                                ),
                                step_name=getattr(st, "name", None),
                                suggestion=(
                                    "Map to distinct parent keys per branch or adjust merge strategy/field_mapping."
                                ),
                            )
                        )
                except Exception:
                    pass

                # V-P3: heterogeneous first-step input types across branches
                try:
                    branch_input_types: set[str] = set()
                    for bname, bp in (getattr(st, "branches", {}) or {}).items():
                        try:
                            steps_in = getattr(bp, "steps", []) or []
                            if not steps_in:
                                continue
                            first = steps_in[0]
                            category = None
                            # Prefer immediate templated_input literal category when non-template
                            meta = getattr(first, "meta", None)
                            if isinstance(meta, dict) and "templated_input" in meta:
                                tv = meta.get("templated_input")
                                if isinstance(tv, str) and ("{{" in tv and "}}" in tv):
                                    category = None
                                else:
                                    if isinstance(tv, bool):
                                        category = "bool"
                                    elif isinstance(tv, (int, float)):
                                        category = "number"
                                    elif isinstance(tv, str):
                                        category = "string"
                                    elif isinstance(tv, dict):
                                        category = "object"
                                    elif isinstance(tv, list):
                                        category = "array"
                            if category is None:
                                itype = getattr(first, "__step_input_type__", object)
                                category = str(itype)
                            branch_input_types.add(category)
                        except Exception:
                            continue
                    if len(branch_input_types) > 1:
                        out.append(
                            ValidationFinding(
                                rule_id="V-P3",
                                severity="warning",
                                message=(
                                    f"ParallelStep '{getattr(st, 'name', None)}' branches expect heterogeneous input types; "
                                    "the same input is passed to all branches."
                                ),
                                step_name=getattr(st, "name", None),
                                suggestion=(
                                    "Ensure branches handle the same input type or insert adapter steps per branch."
                                ),
                            )
                        )
                except Exception:
                    pass

        # Loop checks (V-L1, V-CF1)
        if _LoopStep is not None:
            for st in steps:
                if not isinstance(st, _LoopStep):
                    continue
                # V-CF1: extreme max_loops or constant-false exit condition
                try:
                    ml = 0
                    try:
                        ml = int(getattr(st, "max_loops", getattr(st, "max_retries", 0)) or 0)
                    except Exception:
                        ml = 0
                    if ml >= 1000:
                        sev = _override_severity("V-CF1", "error")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-CF1",
                                    severity=sev,
                                    message=(
                                        f"LoopStep '{getattr(st, 'name', None)}' declares max_loops={ml}, which may create a non-terminating loop."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Provide a stricter exit_condition or reduce max_loops to a reasonable bound."
                                    ),
                                )
                            )
                    else:
                        fn = getattr(st, "exit_condition_callable", None)
                        flag_const_false = False
                        if hasattr(fn, "__code__") and callable(fn):
                            try:
                                co = getattr(fn, "__code__")
                                consts = tuple(getattr(co, "co_consts", ()) or ())
                                names = tuple(getattr(co, "co_names", ()) or ())
                                if (False in consts) and (True not in consts) and (len(names) == 0):
                                    flag_const_false = True
                            except Exception:
                                flag_const_false = False
                        if flag_const_false:
                            sev = _override_severity("V-CF1", "error")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-CF1",
                                        severity=sev,
                                        message=(
                                            f"LoopStep '{getattr(st, 'name', None)}' exit condition appears to be constant false (non-terminating)."
                                        ),
                                        step_name=getattr(st, "name", None),
                                        suggestion=(
                                            "Ensure exit_condition depends on loop results or context and eventually returns True."
                                        ),
                                    )
                                )
                except Exception:
                    pass

                # V-L1: loop exit coverage heuristic
                try:
                    meta = getattr(st, "meta", None)
                    yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                    loc_path = (yloc or {}).get("path")
                    body = None
                    try:
                        getter = getattr(st, "get_loop_body_pipeline", None)
                        body = (
                            getter()
                            if callable(getter)
                            else getattr(st, "loop_body_pipeline", None)
                        )
                    except Exception:
                        body = getattr(st, "loop_body_pipeline", None)
                    body_steps = list(getattr(body, "steps", []) or [])
                    body_updates = any(
                        bool(getattr(bs, "updates_context", False)) for bs in body_steps
                    )
                    has_iter_mapper = getattr(st, "iteration_input_mapper", None) is not None
                    has_init_mapper = (
                        getattr(st, "initial_input_to_loop_body_mapper", None) is not None
                    )
                    has_output_mapper = getattr(st, "loop_output_mapper", None) is not None
                    ml = 0
                    try:
                        ml = int(getattr(st, "max_loops", getattr(st, "max_retries", 0)) or 0)
                    except Exception:
                        ml = 0
                    ml_small = ml and ml <= 5
                    if (not body_updates) and (not has_iter_mapper) and (not has_output_mapper):
                        sev = _override_severity("V-L1", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-L1",
                                    severity=sev,
                                    message=(
                                        f"LoopStep '{getattr(st, 'name', None)}' may not be able to reach its exit condition: "
                                        "no context updates in body, no iteration_input_mapper, and no loop_output_mapper."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    location_path=loc_path,
                                    file=(yloc or {}).get("file"),
                                    line=(yloc or {}).get("line"),
                                    column=(yloc or {}).get("column"),
                                    suggestion=(
                                        "Provide an iteration_input_mapper, update context in the body, or map outputs via loop_output_mapper so the exit condition can be satisfied."
                                    ),
                                )
                            )
                    elif (
                        (not body_updates)
                        and (not has_init_mapper)
                        and (not has_iter_mapper)
                        and (not ml_small)
                    ):
                        sev = _override_severity("V-L1", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-L1",
                                    severity=sev,
                                    message=(
                                        f"LoopStep '{getattr(st, 'name', None)}' has no input mappers and body seems side-effect free; consider exit coverage."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    location_path=loc_path,
                                    file=(yloc or {}).get("file"),
                                    line=(yloc or {}).get("line"),
                                    column=(yloc or {}).get("column"),
                                    suggestion=(
                                        "Add an iteration_input_mapper or ensure the body updates context or output that the exit condition uses."
                                    ),
                                )
                            )
                except Exception:
                    pass

        # V-CTX1: Context isolation in loop/parallel steps with custom skills
        # Warn when loops or parallel steps contain custom Python skills that receive context,
        # as they might mutate shared context without isolation (FLUJO_TEAM_GUIDE.md Section 3.5)
        self._check_context_isolation(steps, _LoopStep, _ParallelStep, out)

        # StateMachine checks (V-SM1)
        try:
            from ..domain.dsl.state_machine import StateMachineStep as _SM
        except Exception:
            _SM = None  # type: ignore
        if _SM is not None:
            for idx, st in enumerate(steps):
                try:
                    if not isinstance(st, _SM):
                        continue
                    meta = getattr(st, "meta", None)
                    yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                    loc_path = (yloc or {}).get("path") or f"steps[{idx}]"
                    fpath = (yloc or {}).get("file")
                    line = (yloc or {}).get("line")
                    col = (yloc or {}).get("column")

                    states: set[str] = set(getattr(st, "states", {}) or {})
                    start: str = str(getattr(st, "start_state", ""))
                    ends: set[str] = set(getattr(st, "end_states", []) or [])
                    transitions = list(getattr(st, "transitions", []) or [])

                    # Start exists
                    if start and start not in states:
                        sev = _override_severity("V-SM1", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-SM1",
                                    severity=sev,
                                    message=(
                                        f"StateMachine '{getattr(st, 'name', None)}' start_state '{start}' is not a defined state."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=("Ensure start_state matches a key in 'states'."),
                                )
                            )

                    # Build adjacency and reachability
                    adj: dict[str, set[str]] = {s: set() for s in states}
                    reachable_end = False
                    for tr in transitions:
                        try:
                            frm = str(getattr(tr, "from_state", ""))
                            to = str(getattr(tr, "to", ""))
                            if frm == "*":
                                from_candidates = set(states)
                            else:
                                from_candidates = {frm} if frm in states else set()
                            for s in from_candidates:
                                if to in states:
                                    adj.setdefault(s, set()).add(to)
                                elif to in ends:
                                    adj.setdefault(s, set())
                                    reachable_end = reachable_end or (s == start)
                        except Exception:
                            continue

                    visited: set[str] = set()
                    if start in states:
                        q: list[str] = [start]
                        while q:
                            cur = q.pop(0)
                            if cur in visited:
                                continue
                            visited.add(cur)
                            for nxt in adj.get(cur, set()):
                                if nxt not in visited:
                                    q.append(nxt)

                    unreachable = sorted(states - visited) if start in states else []
                    if unreachable:
                        sev = _override_severity("V-SM1", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="V-SM1",
                                    severity=sev,
                                    message=(
                                        f"StateMachine '{getattr(st, 'name', None)}' has unreachable states: {unreachable}"
                                    ),
                                    step_name=getattr(st, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=("Review transitions or remove unused states."),
                                )
                            )

                    if ends:
                        path_to_end = False
                        if start in states:
                            for s in visited or []:
                                for tr in transitions:
                                    try:
                                        frm2 = str(getattr(tr, "from_state", ""))
                                        to2 = str(getattr(tr, "to", ""))
                                        if (frm2 == s or frm2 == "*") and (to2 in ends):
                                            path_to_end = True
                                            break
                                    except Exception:
                                        continue
                                if path_to_end:
                                    break
                        path_to_end = path_to_end or reachable_end
                        if not path_to_end and start in states:
                            sev = _override_severity("V-SM1", "warning")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="V-SM1",
                                        severity=sev,
                                        message=(
                                            f"StateMachine '{getattr(st, 'name', None)}' has no transition path from start_state '{start}' to any end state {sorted(ends)}"
                                        ),
                                        step_name=getattr(st, "name", None),
                                        location_path=loc_path,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                        suggestion=(
                                            "Add a transition to an end state or adjust end_states."
                                        ),
                                    )
                                )
                except Exception:
                    continue
                except Exception:
                    pass

        return out

    def _check_context_isolation(
        self, steps: list[Any], LoopStepCls: Any, ParallelStepCls: Any, out: list[ValidationFinding]
    ) -> None:
        """Check for potential context isolation issues in loop/parallel steps (V-CTX1)."""
        if LoopStepCls is None and ParallelStepCls is None:
            return

        for idx, st in enumerate(steps):
            # Check both LoopStep and ParallelStep
            if not (
                (LoopStepCls and isinstance(st, LoopStepCls))
                or (ParallelStepCls and isinstance(st, ParallelStepCls))
            ):
                continue

            step_type = (
                "LoopStep" if (LoopStepCls and isinstance(st, LoopStepCls)) else "ParallelStep"
            )
            step_name = getattr(st, "name", None)

            # Get location info
            meta = getattr(st, "meta", None)
            yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
            loc_path = (yloc or {}).get("path") or f"steps[{idx}]"
            fpath = (yloc or {}).get("file")
            line = (yloc or {}).get("line")
            col = (yloc or {}).get("column")

            # Check if the step body contains custom skills
            has_custom_skills = False
            custom_skill_refs = []

            try:
                # For LoopStep, check loop_body_pipeline.steps
                if LoopStepCls and isinstance(st, LoopStepCls):
                    body_pipeline = getattr(st, "loop_body_pipeline", None)
                    if body_pipeline:
                        body_steps = getattr(body_pipeline, "steps", []) or []
                        for body_step in body_steps:
                            skill_ref = self._get_custom_skill_ref(body_step)
                            if skill_ref:
                                has_custom_skills = True
                                custom_skill_refs.append(skill_ref)

                # For ParallelStep, check branches
                elif ParallelStepCls and isinstance(st, ParallelStepCls):
                    branches = getattr(st, "branches", {}) or {}
                    for branch_name, branch_pipeline in branches.items():
                        if branch_pipeline:
                            branch_steps = getattr(branch_pipeline, "steps", []) or []
                            for branch_step in branch_steps:
                                skill_ref = self._get_custom_skill_ref(branch_step)
                                if skill_ref:
                                    has_custom_skills = True
                                    custom_skill_refs.append(f"{branch_name}:{skill_ref}")

            except Exception:
                continue

            # If custom skills found, warn about context isolation
            if has_custom_skills:
                sev = _override_severity("V-CTX1", "warning")
                if sev is not None:
                    skills_list = ", ".join(custom_skill_refs[:3])  # Show first 3
                    if len(custom_skill_refs) > 3:
                        skills_list += f" (and {len(custom_skill_refs) - 3} more)"

                    out.append(
                        ValidationFinding(
                            rule_id="V-CTX1",
                            severity=sev,
                            message=(
                                f"{step_type} '{step_name}' contains custom skills ({skills_list}). "
                                f"Ensure they don't mutate shared context without using ContextManager.isolate()."
                            ),
                            step_name=step_name,
                            location_path=loc_path,
                            file=fpath,
                            line=line,
                            column=col,
                            suggestion=(
                                f"In custom skills within {step_type.lower()}s, avoid direct context mutation. "
                                "If you need to modify context in parallel branches or loop iterations, "
                                "use ContextManager.isolate() to create isolated copies. "
                                "See: FLUJO_TEAM_GUIDE.md Section 3.5 'Idempotency in Step Policies'"
                            ),
                        )
                    )

    def _get_custom_skill_ref(self, step: Any) -> str | None:
        """Extract custom skill reference from a step, similar to V-EX1 logic."""
        try:
            # Check if 'uses' was specified in meta (preserved from YAML)
            step_meta = getattr(step, "meta", {}) or {}
            if "uses" in step_meta and isinstance(step_meta["uses"], str):
                uses_val = step_meta["uses"]
                # Custom skills are referenced with module paths (e.g., "my_module:function")
                # Skip built-ins and declarative agents
                if ":" in uses_val and not uses_val.startswith(
                    ("agents.", "imports.", "flujo.builtins.")
                ):
                    return uses_val

            # Also check the resolved agent object
            agent_spec = getattr(step, "agent", None)
            if agent_spec:
                # Check if agent is a _CallableAgent wrapper
                if hasattr(agent_spec, "_step_callable"):
                    wrapped_func = getattr(agent_spec, "_step_callable", None)
                    if wrapped_func:
                        module_name = getattr(wrapped_func, "__module__", None)
                        func_name = getattr(wrapped_func, "__name__", None)

                        # Custom skills typically aren't in flujo.* modules
                        if module_name and not module_name.startswith("flujo."):
                            if func_name:
                                return str(f"{module_name}:{func_name}")
                            else:
                                return str(module_name)
                # Check if agent is a direct callable from a custom module
                elif callable(agent_spec):
                    module_name = getattr(agent_spec, "__module__", None)
                    func_name = getattr(agent_spec, "__name__", None)

                    # Custom skills typically aren't in flujo.* modules
                    if module_name and not module_name.startswith("flujo."):
                        if func_name:
                            return str(f"{module_name}:{func_name}")
                        else:
                            return str(module_name)

        except Exception:
            pass

        return None


class ExceptionLinter(BaseLinter):
    """Control flow exception handling validation: V-EX1.

    This linter warns about potential control flow exception handling issues
    as described in the FLUJO_TEAM_GUIDE.md "Fatal Anti-Pattern" section.

    Control flow exceptions (PausedException, PipelineAbortSignal, InfiniteRedirectError)
    must be re-raised to maintain proper workflow orchestration. Converting them to
    StepResult(success=False) breaks pause/resume and other critical flows.
    """

    CONTROL_FLOW_EXCEPTIONS: ClassVar[set[str]] = {
        "PausedException",
        "PipelineAbortSignal",
        "InfiniteRedirectError",
    }

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        """Detect steps that use custom Python skills and warn about exception handling best practices."""
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        for idx, step in enumerate(steps):
            try:
                meta = getattr(step, "meta", None)
                if not isinstance(meta, dict):
                    continue

                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}]"
                fpath = (yloc or {}).get("file")
                line = (yloc or {}).get("line")
                col = (yloc or {}).get("column")

                # Check if this step uses custom Python skills
                step_name = getattr(step, "name", None)
                agent_spec = getattr(step, "agent", None)

                # Look for custom skill references - check both meta and agent
                uses_custom_skill = False
                skill_ref = None

                # First check if 'uses' was specified in meta (preserved from YAML)
                if "uses" in meta and isinstance(meta["uses"], str):
                    uses_val = meta["uses"]
                    # Custom skills are referenced with module paths (e.g., "my_module:function")
                    if ":" in uses_val and not uses_val.startswith(
                        ("agents.", "imports.", "flujo.builtins.")
                    ):
                        uses_custom_skill = True
                        skill_ref = uses_val

                # Also check the resolved agent object
                if not uses_custom_skill and agent_spec:
                    # Check if agent is a _CallableAgent wrapper
                    if hasattr(agent_spec, "_step_callable"):
                        wrapped_func = getattr(agent_spec, "_step_callable", None)
                        if wrapped_func:
                            module_name = getattr(wrapped_func, "__module__", None)
                            func_name = getattr(wrapped_func, "__name__", None)

                            # Custom skills typically aren't in flujo.* modules
                            if module_name and not module_name.startswith("flujo."):
                                uses_custom_skill = True
                                if func_name:
                                    skill_ref = f"{module_name}:{func_name}"
                                else:
                                    skill_ref = module_name
                    # Check if agent is a direct callable from a custom module
                    elif callable(agent_spec):
                        module_name = getattr(agent_spec, "__module__", None)
                        func_name = getattr(agent_spec, "__name__", None)

                        # Custom skills typically aren't in flujo.* modules
                        if module_name and not module_name.startswith("flujo."):
                            uses_custom_skill = True
                            if func_name:
                                skill_ref = f"{module_name}:{func_name}"
                            else:
                                skill_ref = module_name

                if uses_custom_skill and skill_ref:
                    sev = _override_severity("V-EX1", "warning")
                    if sev is not None:
                        out.append(
                            ValidationFinding(
                                rule_id="V-EX1",
                                severity=sev,
                                message=(
                                    f"Step '{step_name}' uses custom skill '{skill_ref}'. "
                                    f"Ensure it does not catch control flow exceptions "
                                    f"(PausedException, PipelineAbortSignal, InfiniteRedirectError) "
                                    f"without re-raising them."
                                ),
                                step_name=step_name,
                                location_path=loc_path,
                                file=fpath,
                                line=line,
                                column=col,
                                suggestion=(
                                    "In your custom skill, always re-raise control flow exceptions:\n"
                                    "try:\n"
                                    "    # your logic\n"
                                    "except (PausedException, PipelineAbortSignal, InfiniteRedirectError):\n"
                                    "    raise  # CRITICAL: must re-raise\n"
                                    "except Exception as e:\n"
                                    "    # handle other exceptions\n"
                                    "\n"
                                    "See: FLUJO_TEAM_GUIDE.md Section 2 'The Fatal Anti-Pattern'"
                                ),
                            )
                        )

            except Exception:
                continue

        return out


class LoopScopingLinter(BaseLinter):
    """Loop step scoping validation: LOOP-001.

    Detects step references (steps['name']) inside loop bodies that may not work as expected
    due to loop scoping. In loop bodies, only previous_step or context should be used.
    """

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        """Check for step references inside loop/map bodies."""
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        def _check_loop_body_steps(
            body_steps: list[Any], loop_name: str, _loop_meta: dict[str, Any]
        ) -> None:
            """Recursively check steps in a loop body for step references."""
            for _idx, step in enumerate(body_steps):
                try:
                    meta = getattr(step, "meta", {})

                    # Check for condition_expression or exit_expression with steps[] references
                    # These are stored in the meta dict for ConditionalStep
                    for field_name in ["condition_expression", "exit_expression"]:
                        expr = None

                        # Check both direct attribute and meta dict
                        if hasattr(step, field_name):
                            expr = getattr(step, field_name, None)
                        elif isinstance(meta, dict) and field_name in meta:
                            expr = meta.get(field_name)

                        if isinstance(expr, str) and (
                            "steps[" in expr
                            or "steps.get(" in expr
                            or re.search(r"\bsteps\.", expr)
                        ):
                            yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                            loc_path = (yloc or {}).get(
                                "path"
                            ) or f"loop '{loop_name}' body steps[{_idx}].{field_name}"
                            fpath = (yloc or {}).get("file")
                            line = (yloc or {}).get("line")
                            col = (yloc or {}).get("column")

                            sev = _override_severity("LOOP-001", "warning")
                            if sev is not None:
                                out.append(
                                    ValidationFinding(
                                        rule_id="LOOP-001",
                                        severity=sev,
                                        message=(
                                            f"Step reference detected in {field_name} inside loop body '{loop_name}'. "
                                            f"Loop body steps are scoped to the current iteration and may not be "
                                            f"accessible via steps['name']."
                                        ),
                                        step_name=getattr(step, "name", None),
                                        location_path=loc_path,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                        suggestion=(
                                            "Use 'previous_step' to reference the immediate previous step in the loop body.\n"
                                            "\n"
                                            "Example:\n"
                                            "  ❌ condition_expression: \"steps['process'].output.status == 'done'\"\n"
                                            "  ✅ condition_expression: \"previous_step.status == 'done'\"\n"
                                            "\n"
                                            "To access steps from outside the loop, use context to carry data."
                                        ),
                                    )
                                )

                    # Check templated_input for step references
                    meta = getattr(step, "meta", {})
                    templ = meta.get("templated_input")
                    if isinstance(templ, str) and (
                        "steps[" in templ or "steps.get(" in templ or re.search(r"\bsteps\.", templ)
                    ):
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get(
                            "path"
                        ) or f"loop '{loop_name}' body steps[{_idx}].input"
                        fpath = (yloc or {}).get("file")
                        line = (yloc or {}).get("line")
                        col = (yloc or {}).get("column")

                        sev = _override_severity("LOOP-001", "warning")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="LOOP-001",
                                    severity=sev,
                                    message=(
                                        f"Step reference detected in template inside loop body '{loop_name}'. "
                                        f"Loop body steps are scoped to the current iteration."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=(
                                        "Use '{{ previous_step }}' to reference the immediate previous step.\n"
                                        "\n"
                                        "Example:\n"
                                        '  ❌ input: "{{ steps.process.output }}"\n'
                                        '  ✅ input: "{{ previous_step }}"\n'
                                    ),
                                )
                            )

                    # Recursively check nested loop bodies
                    if hasattr(step, "loop_body_pipeline"):
                        nested_pipeline = getattr(step, "loop_body_pipeline", None)
                        if nested_pipeline and hasattr(nested_pipeline, "steps"):
                            nested_steps = getattr(nested_pipeline, "steps", [])
                            nested_meta = getattr(step, "meta", {})
                            _check_loop_body_steps(
                                nested_steps, getattr(step, "name", "nested_loop"), nested_meta
                            )

                except Exception as e:
                    # Log validation error but continue checking other steps
                    import logging

                    logging.getLogger(__name__).debug(f"Failed to validate loop body step: {e}")
                    continue

        # Check each top-level step for loops/maps
        for step in steps:
            try:
                # Check LoopStep
                if hasattr(step, "loop_body_pipeline"):
                    pipeline_body = getattr(step, "loop_body_pipeline", None)
                    if pipeline_body and hasattr(pipeline_body, "steps"):
                        body_steps = getattr(pipeline_body, "steps", [])
                        meta = getattr(step, "meta", {})
                        _check_loop_body_steps(body_steps, getattr(step, "name", "loop"), meta)

                # Check MapStep (inherits from LoopStep)
                if hasattr(step, "pipeline_to_run"):
                    pipeline_body = getattr(step, "pipeline_to_run", None)
                    if pipeline_body and hasattr(pipeline_body, "steps"):
                        body_steps = getattr(pipeline_body, "steps", [])
                        meta = getattr(step, "meta", {})
                        _check_loop_body_steps(body_steps, getattr(step, "name", "map"), meta)

            except Exception as e:
                # Log validation error but continue checking other steps
                import logging

                logging.getLogger(__name__).debug(f"Failed to validate step: {e}")
                continue

        return out


class TemplateControlStructureLinter(BaseLinter):
    """Template Jinja2 control structure validation: TEMPLATE-001.

    Detects unsupported Jinja2 control structures ({% %}) in templates.
    Flujo only supports {{ }} expressions and | filters, not control flow.
    """

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        """Check for unsupported Jinja2 control structures in templates."""
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        def _check_template_field(template_str: str, field_name: str, step: Any, idx: int) -> None:
            """Check a single template field for control structures."""
            if not isinstance(template_str, str):
                return

            # Detect {% %} control structures
            if "{%" in template_str or "%}" in template_str:
                meta = getattr(step, "meta", {})
                yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                loc_path = (yloc or {}).get("path") or f"steps[{idx}].{field_name}"
                fpath = (yloc or {}).get("file")
                line = (yloc or {}).get("line")
                col = (yloc or {}).get("column")

                # Try to extract the control structure name
                control_match = re.search(r"\{%\s*([a-z]+)", template_str)
                control_name = control_match.group(1) if control_match else "unknown"

                sev = _override_severity("TEMPLATE-001", "error")
                if sev is not None:
                    out.append(
                        ValidationFinding(
                            rule_id="TEMPLATE-001",
                            severity=sev,
                            message=(
                                f"Unsupported Jinja2 control structure '{{%{control_name}%}}' detected in {field_name}. "
                                f"Flujo templates support expressions {{{{ }}}} and filters |, but NOT control structures {{%{control_name}%}}."
                            ),
                            step_name=getattr(step, "name", None),
                            location_path=loc_path,
                            file=fpath,
                            line=line,
                            column=col,
                            suggestion=(
                                "Alternatives:\n"
                                "  1. Use template filters: {{ context.items | join('\\n') }}\n"
                                '  2. Use custom skill: uses: "skills:format_data"\n'
                                "  3. Use conditional steps for if/else logic\n"
                                "  4. Pre-format data in a previous step\n"
                                "\n"
                                "Supported:\n"
                                "  ✅ {{ variable }}\n"
                                "  ✅ {{ value | filter }}\n"
                                "  ✅ {{ context.nested.field }}\n"
                                "\n"
                                "NOT Supported:\n"
                                "  ❌ {% for %}, {% if %}, {% set %}\n"
                                "  ❌ {% macro %}, {% include %}\n"
                                "\n"
                                "Documentation: https://flujo.dev/docs/templates"
                            ),
                        )
                    )

        # Check all steps for template fields
        for idx, step in enumerate(steps):
            try:
                meta = getattr(step, "meta", {})

                # Check templated_input
                templ = meta.get("templated_input")
                if templ:
                    _check_template_field(templ, "input", step, idx)

                # Check message field (for HITL steps)
                if hasattr(step, "message"):
                    message = getattr(step, "message", None)
                    if message:
                        _check_template_field(message, "message", step, idx)

                # Check other common template fields
                for field in ["description", "prompt", "system_prompt"]:
                    if hasattr(step, field):
                        value = getattr(step, field, None)
                        if value:
                            _check_template_field(value, field, step, idx)

            except Exception as e:
                # Log validation error but continue checking other steps
                import logging

                logging.getLogger(__name__).debug(f"Failed to validate template field: {e}")
                continue

        return out


class HitlNestedContextLinter(BaseLinter):
    """HITL nested context validation: HITL-NESTED-001 (CRITICAL).

    Detects HITL (Human-In-The-Loop) steps inside nested contexts like loops and conditionals.
    HITL steps in nested contexts WILL BE SILENTLY SKIPPED at runtime, causing data loss.
    This is a known limitation and pipelines with this pattern will fail.
    """

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        """Check for HITL steps in nested contexts (loops, conditionals)."""
        out: list[ValidationFinding] = []

        # Import step types lazily to avoid circular dependencies
        try:
            from ..domain.dsl.step import HumanInTheLoopStep as _HITLStep
        except Exception as e:
            # Log import error but continue
            import logging

            logging.getLogger(__name__).debug(f"Failed to import HumanInTheLoopStep: {e}")
            _HITLStep = None  # type: ignore

        if _HITLStep is None:
            return out

        def _check_for_hitl_in_steps(
            steps: list[Any], context_chain: list[str], _parent_step_name: str | None = None
        ) -> None:
            """Recursively check steps for HITL in nested contexts."""
            for _idx, step in enumerate(steps):
                try:
                    # Check if this is a HITL step
                    is_hitl = isinstance(step, _HITLStep)

                    # Also check by kind field in meta (for YAML-loaded steps)
                    if not is_hitl:
                        meta = getattr(step, "meta", {})
                        if isinstance(meta, dict):
                            kind = meta.get("kind", "")
                            is_hitl = kind == "hitl"

                    # If this is a HITL step in a problematic nested context, warn
                    # Only block HITL in conditional branches inside loops (not HITL directly in loops)
                    if is_hitl and len(context_chain) > 0:
                        # Check if this is a problematic pattern: conditional inside loop
                        # We need to ensure a conditional appears AFTER the innermost loop
                        loop_indices = [
                            i
                            for i, ctx in enumerate(context_chain)
                            if ctx.startswith(("loop:", "map:"))
                        ]
                        if not loop_indices:
                            continue  # No loops present, allow HITL

                        innermost_loop_idx = max(loop_indices)
                        conditional_inside_loop = any(
                            idx > innermost_loop_idx
                            and context_chain[idx].startswith("conditional:")
                            for idx in range(innermost_loop_idx + 1, len(context_chain))
                        )

                        if not conditional_inside_loop:
                            continue  # Skip validation for HITL directly in loops or conditionals outside loops
                        meta = getattr(step, "meta", {})
                        yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                        loc_path = (yloc or {}).get(
                            "path"
                        ) or f"step '{getattr(step, 'name', 'unnamed')}'"
                        fpath = (yloc or {}).get("file")
                        line = (yloc or {}).get("line")
                        col = (yloc or {}).get("column")

                        context_desc = " > ".join(context_chain)

                        sev = _override_severity("HITL-NESTED-001", "error")
                        if sev is not None:
                            out.append(
                                ValidationFinding(
                                    rule_id="HITL-NESTED-001",
                                    severity=sev,
                                    message=(
                                        f"HITL step '{getattr(step, 'name', 'unnamed')}' will be SILENTLY SKIPPED at runtime. "
                                        f"Context: {context_desc}. "
                                        f"This is a known limitation: HITL steps in conditional branches inside loops do NOT execute. "
                                        f"The step will be filtered out silently with no error message, causing data loss."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    location_path=loc_path,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                    suggestion=(
                                        "CRITICAL: This pipeline will fail at runtime. Apply one of these workarounds:\n"
                                        "\n"
                                        "  1. Move HITL step outside the loop (RECOMMENDED)\n"
                                        "     - Collect user input before entering the loop\n"
                                        "     - Store result in context for use inside loop\n"
                                        "\n"
                                        "  2. Remove the conditional wrapper\n"
                                        "     - If the HITL must be in a loop, remove the conditional\n"
                                        "     - HITL directly in loop body should work\n"
                                        "\n"
                                        "  3. Use flujo.builtins.ask_user skill instead\n"
                                        "     - Built-in skills may have better nested context support\n"
                                        "\n"
                                        "Example fix:\n"
                                        "  # ❌ WILL FAIL - HITL in conditional in loop\n"
                                        "  - kind: loop\n"
                                        "    body:\n"
                                        "      - kind: conditional\n"
                                        "        branches:\n"
                                        "          true:\n"
                                        "            - kind: hitl  # <- SILENTLY SKIPPED!\n"
                                        "              message: 'Question?'\n"
                                        "\n"
                                        "  # ✅ WORKS - HITL at top-level\n"
                                        "  - kind: hitl\n"
                                        "    name: get_input\n"
                                        "    message: 'Question?'\n"
                                        "    sink_to: 'user_answer'\n"
                                        "  - kind: loop\n"
                                        "    body:\n"
                                        "      - kind: step\n"
                                        "        input: '{{ context.user_answer }}'\n"
                                        "\n"
                                        "Documentation: https://flujo.dev/docs/known-issues/hitl-nested\n"
                                        "Report hours lost debugging this? https://github.com/aandresalvarez/flujo/issues"
                                    ),
                                )
                            )

                    # Recursively check nested structures
                    # Check loop bodies
                    if hasattr(step, "loop_body_pipeline"):
                        loop_pipeline = getattr(step, "loop_body_pipeline", None)
                        if loop_pipeline and hasattr(loop_pipeline, "steps"):
                            nested_steps = getattr(loop_pipeline, "steps", [])
                            step_name = getattr(step, "name", "loop")
                            new_chain = [*context_chain, f"loop:{step_name}"]
                            _check_for_hitl_in_steps(nested_steps, new_chain, step_name)

                    # Check map bodies (similar to loops)
                    if hasattr(step, "pipeline_to_run"):
                        map_pipeline = getattr(step, "pipeline_to_run", None)
                        if map_pipeline and hasattr(map_pipeline, "steps"):
                            nested_steps = getattr(map_pipeline, "steps", [])
                            step_name = getattr(step, "name", "map")
                            new_chain = [*context_chain, f"map:{step_name}"]
                            _check_for_hitl_in_steps(nested_steps, new_chain, step_name)

                    # Check conditional branches
                    if hasattr(step, "branches"):
                        branches = getattr(step, "branches", {})
                        if isinstance(branches, dict):
                            step_name = getattr(step, "name", "conditional")
                            for branch_key, branch_pipeline in branches.items():
                                if branch_pipeline and hasattr(branch_pipeline, "steps"):
                                    branch_steps = getattr(branch_pipeline, "steps", [])
                                    new_chain = [
                                        *context_chain,
                                        f"conditional:{step_name}",
                                        f"branch:{branch_key}",
                                    ]
                                    _check_for_hitl_in_steps(branch_steps, new_chain, step_name)

                    # Check parallel branches
                    if hasattr(step, "parallel_branches"):
                        parallel_branches = getattr(step, "parallel_branches", {})
                        if isinstance(parallel_branches, dict):
                            step_name = getattr(step, "name", "parallel")
                            for branch_name, branch_pipeline in parallel_branches.items():
                                if branch_pipeline and hasattr(branch_pipeline, "steps"):
                                    branch_steps = getattr(branch_pipeline, "steps", [])
                                    new_chain = [
                                        *context_chain,
                                        f"parallel:{step_name}",
                                        f"branch:{branch_name}",
                                    ]
                                    _check_for_hitl_in_steps(branch_steps, new_chain, step_name)

                except Exception as e:
                    # Log validation error but continue checking other steps
                    import logging

                    logging.getLogger(__name__).debug(
                        f"Failed to validate HITL nested context: {e}"
                    )
                    continue

        # Start checking from top-level steps
        steps = getattr(pipeline, "steps", []) or []
        _check_for_hitl_in_steps(steps, [], None)

        return out


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
            for f in lin.analyze(pipeline) or []:
                if f.severity == "error":
                    errors.append(f)
                else:
                    warnings.append(f)
        except Exception:
            continue

    return ValidationReport(errors=errors, warnings=warnings)

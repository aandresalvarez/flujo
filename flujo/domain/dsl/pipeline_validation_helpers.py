from __future__ import annotations

import logging
from typing import Any, Set, TYPE_CHECKING

from ..pipeline_validation import ValidationFinding, ValidationReport
from ...exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Pipeline


def aggregate_import_validation(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
    *,
    include_imports: bool,
    visited_pipelines: Set[int],
    visited_paths: Set[str],
    report_cache: dict[str, ValidationReport],
) -> None:
    """Aggregate validation findings from imported child pipelines (V-I rules)."""
    if not include_imports:
        return

    try:
        from .import_step import ImportStep as _ImportStep
    except Exception:
        _ImportStep = None  # type: ignore

    if _ImportStep is None:
        return

    for step in getattr(pipeline, "steps", []) or []:
        try:
            if not isinstance(step, _ImportStep):
                continue

            child = getattr(step, "pipeline", None)
            if child is None or not hasattr(child, "validate_graph"):
                continue

            import os as _os

            ch_path = getattr(child, "_source_file", None)
            if isinstance(ch_path, str):
                ch_path = _os.path.realpath(ch_path)

            # Cycle detection (V-I3): path-based preferred; fallback to id
            is_cycle = False
            if isinstance(ch_path, str) and ch_path in visited_paths:
                is_cycle = True
            elif id(child) in visited_pipelines:
                is_cycle = True
            if is_cycle:
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-I3",
                        severity="error",
                        message=(
                            "Cyclic import detected while validating imports; import graph contains a cycle."
                        ),
                        step_name=getattr(step, "name", None),
                    )
                )
                continue

            # Cache lookup by path
            use_cache = isinstance(ch_path, str) and bool(ch_path)
            if use_cache and ch_path in report_cache:
                child_report = report_cache.get(ch_path) or ValidationReport()
            else:
                child_report = child.validate_graph(
                    include_imports=True,
                    _visited_pipelines=visited_pipelines,
                    _visited_paths=visited_paths,
                    _report_cache=report_cache,
                )
                if use_cache and ch_path:
                    report_cache[ch_path] = child_report

            # Aggregate child findings with step context
            meta = getattr(step, "meta", None)
            alias = meta.get("import_alias") if isinstance(meta, dict) else None
            import_tag = alias or getattr(step, "name", "")

            for f in child_report.errors:
                loc = f.location_path
                if import_tag:
                    loc = f"imports.{import_tag}::{loc}" if loc else f"imports.{import_tag}"
                report.errors.append(
                    ValidationFinding(
                        rule_id=f.rule_id,
                        severity=f.severity,
                        message=f"[import:{import_tag}] {f.message}",
                        step_name=f.step_name or getattr(step, "name", None),
                        suggestion=f.suggestion,
                        location_path=loc,
                        file=f.file,
                        line=f.line,
                        column=f.column,
                        import_alias=import_tag or None,
                        import_stack=(
                            ([import_tag] if import_tag else [])
                            + ((f.import_stack or []) if hasattr(f, "import_stack") else [])
                        ),
                    )
                )

            for w in child_report.warnings:
                loc = w.location_path
                if import_tag:
                    loc = f"imports.{import_tag}::{loc}" if loc else f"imports.{import_tag}"
                report.warnings.append(
                    ValidationFinding(
                        rule_id=w.rule_id,
                        severity=w.severity,
                        message=f"[import:{import_tag}] {w.message}",
                        step_name=w.step_name or getattr(step, "name", None),
                        suggestion=w.suggestion,
                        location_path=loc,
                        file=w.file,
                        line=w.line,
                        column=w.column,
                        import_alias=import_tag or None,
                        import_stack=(
                            ([import_tag] if import_tag else [])
                            + ((w.import_stack or []) if hasattr(w, "import_stack") else [])
                        ),
                    )
                )

            # V-I5: Input projection coherence heuristics
            try:
                child_steps = list(getattr(child, "steps", []) or [])
                if child_steps:
                    first = child_steps[0]
                    child_in = getattr(first, "__step_input_type__", object)
                    input_to = str(getattr(step, "input_to", "initial_prompt")).strip().lower()

                    def _is_objectish(t: Any) -> bool:
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

                    if input_to == "initial_prompt" and _is_objectish(child_in):
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-I5",
                                severity="warning",
                                message=(
                                    f"Import '{import_tag}' projects input to initial_prompt, "
                                    "but child first step expects an object."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=(
                                    "Use input_to=scratchpad or input_to=both (with input_scratchpad_key) "
                                    "to pass structured input."
                                ),
                            )
                        )
                    if input_to == "scratchpad" and child_in is str:
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-I5",
                                severity="warning",
                                message=(
                                    f"Import '{import_tag}' projects input to scratchpad only, "
                                    "but child first step expects a string input."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=(
                                    "Use input_to=both or input_to=initial_prompt to ensure the string input is provided."
                                ),
                            )
                        )
            except Exception:
                pass

            # V-I6: Inherit conversation/context consistency
            try:
                inherit_conversation = bool(getattr(step, "inherit_conversation", True))
                outs2 = getattr(step, "outputs", None)
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
                                report.warnings.append(
                                    ValidationFinding(
                                        rule_id="V-I6",
                                        severity="warning",
                                        message=(
                                            f"Import '{import_tag}' maps conversation-related fields but "
                                            "inherit_conversation=False; continuity may be lost."
                                        ),
                                        step_name=getattr(step, "name", None),
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

            # Additional V-I5 heuristic based on parent-provided input shape
            try:
                input_to2 = str(getattr(step, "input_to", "initial_prompt")).strip().lower()
                meta_step = getattr(step, "meta", {}) or {}
                t_in = meta_step.get("templated_input")
                if input_to2 == "initial_prompt" and isinstance(t_in, dict):
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-I5",
                            severity="warning",
                            message=(
                                "Import projects an object literal to initial_prompt; consider projecting to scratchpad or both."
                            ),
                            step_name=getattr(step, "name", None),
                            suggestion=(
                                "Use input_to=scratchpad or both with input_scratchpad_key to pass structured input."
                            ),
                        )
                    )
            except Exception:
                pass

            # Emit a summary V-I4 on the parent step to signal aggregation
            try:
                ce = len(child_report.errors)
                cw = len(child_report.warnings)
                if ce or cw:
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-I4",
                            severity="warning",
                            message=(
                                f"Aggregated child findings from import '{import_tag}': {ce} errors, {cw} warnings."
                            ),
                            step_name=getattr(step, "name", None),
                            location_path=f"imports.{import_tag}",
                        )
                    )
            except Exception:
                pass
        except Exception as import_err:
            logging.debug(
                "Import validation aggregation failed for %r: %s",
                getattr(step, "name", None),
                import_err,
            )
            continue


def apply_fallback_template_lints(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
) -> None:
    """
    Minimal in-process template lints (V-T rules) when external linters are unavailable.
    Mirrors the defensive fallback logic from Pipeline.validate_graph.
    """
    try:
        from ..pipeline_validation import ValidationFinding as _VF
        import re as _re
        import json as _json
    except Exception:
        return

    try:

        def _expects_json(_t: Any) -> bool:
            try:
                from typing import get_origin as _go

                org = _go(_t)
            except Exception:
                org = None
            if _t is dict or org is dict:
                return True
            try:
                from pydantic import BaseModel as _PM

                return isinstance(_t, type) and issubclass(_t, _PM)
            except Exception:
                return False

        for _idx, _st in enumerate(getattr(pipeline, "steps", []) or []):
            try:
                _meta = getattr(_st, "meta", None)
                _templ = _meta.get("templated_input") if isinstance(_meta, dict) else None
                if not isinstance(_templ, str):
                    continue
                _loc = (_meta.get("_yaml_loc") or {}) if isinstance(_meta, dict) else {}
                _fpath = _loc.get("file")
                _line = _loc.get("line")
                _col = _loc.get("column")
                _loc_path = (
                    (_meta.get("_yaml_loc") or {}).get("path") if isinstance(_meta, dict) else None
                )
                _has_tokens = bool(_re.search(r"\{\{.*\}\}", _templ))

                # V-T1: previous_step.output misuse
                if (
                    _has_tokens
                    and _re.search(r"previous_step\s*\.\s*output\b", _templ)
                    and _idx > 0
                ):
                    report.warnings.append(
                        _VF(
                            rule_id="V-T1",
                            severity="warning",
                            message=(
                                "Template references previous_step.output, but previous_step is the raw value and has no .output attribute."
                            ),
                            step_name=getattr(_st, "name", None),
                            location_path=_loc_path or f"steps[{_idx}].input",
                            file=_fpath,
                            line=_line,
                            column=_col,
                        )
                    )

                # V-T2: 'this' outside map body (heuristic)
                if _has_tokens:
                    _token_has_this = False
                    for _tm in _re.finditer(r"\{\{(.*?)\}\}", _templ, _re.DOTALL):
                        if _re.search(r"\bthis\b", _tm.group(1)):
                            _token_has_this = True
                            break
                    if _token_has_this:
                        report.warnings.append(
                            _VF(
                                rule_id="V-T2",
                                severity="warning",
                                message=(
                                    "Template references 'this' outside a known map body context."
                                ),
                                step_name=getattr(_st, "name", None),
                                location_path=_loc_path or f"steps[{_idx}].input",
                                file=_fpath,
                                line=_line,
                                column=_col,
                            )
                        )

                # V-T3: unknown/disabled filters
                if _has_tokens:
                    try:
                        from ...utils.prompting import _get_enabled_filters as _filters

                        _enabled = {s.lower() for s in _filters()}
                    except Exception:
                        _enabled = {"join", "upper", "lower", "length", "tojson"}
                    for _m in _re.finditer(r"\|\s*([a-zA-Z_][a-zA-Z0-9_]*)", _templ):
                        _fname = (_m.group(1) or "").lower()
                        if _fname and _fname not in _enabled:
                            report.warnings.append(
                                _VF(
                                    rule_id="V-T3",
                                    severity="warning",
                                    message=f"Unknown or disabled template filter: {_fname}",
                                    step_name=getattr(_st, "name", None),
                                    location_path=_loc_path or f"steps[{_idx}].input",
                                    file=_fpath,
                                    line=_line,
                                    column=_col,
                                )
                            )

                # V-T5: missing prior model field in previous_step.<field>
                if _has_tokens and _idx > 0:
                    try:
                        _prev_t = getattr(pipeline.steps[_idx - 1], "__step_output_type__", Any)
                        _fields: set[str] = set()
                        if hasattr(_prev_t, "model_fields"):
                            _fields = set(getattr(_prev_t, "model_fields", {}).keys())
                        elif hasattr(_prev_t, "__fields__"):
                            _fields = set(getattr(_prev_t, "__fields__", {}).keys())
                        _comp = "".join(ch for ch in _templ if ch not in (" ", "\t", "\n", "\r"))
                        _key = "previous_step."
                        _start = 0
                        _missing: set[str] = set()
                        while True:
                            _i = _comp.find(_key, _start)
                            if _i == -1:
                                break
                            _j = _i + len(_key)
                            _buf: list[str] = []
                            while _j < len(_comp) and (_comp[_j].isalnum() or _comp[_j] == "_"):
                                _buf.append(_comp[_j])
                                _j += 1
                            _fld = "".join(_buf)
                            if _fld and _fld != "output" and _fld not in _fields:
                                _missing.add(_fld)
                            _start = _j
                        for _fld in sorted(_missing):
                            report.warnings.append(
                                _VF(
                                    rule_id="V-T5",
                                    severity="warning",
                                    message=(
                                        f"Template references previous_step.{_fld} but field is not present on prior model {getattr(_prev_t, '__name__', _prev_t)}."
                                    ),
                                    step_name=getattr(_st, "name", None),
                                    location_path=_loc_path or f"steps[{_idx}].input",
                                    file=_fpath,
                                    line=_line,
                                    column=_col,
                                )
                            )
                    except Exception:
                        pass

                # V-T6: looks like JSON but fails to parse while input expects JSON
                _in_t = getattr(_st, "__step_input_type__", Any)
                if _expects_json(_in_t):
                    if _has_tokens:
                        _clean = _re.sub(r"\{\{.*?\}\}", "null", _templ).strip()
                        if (_clean.startswith("{") and _clean.endswith("}")) or (
                            _clean.startswith("[") and _clean.endswith("]")
                        ):
                            try:
                                _json.loads(_clean)
                            except Exception:
                                report.warnings.append(
                                    _VF(
                                        rule_id="V-T6",
                                        severity="warning",
                                        message=(
                                            "Templated input resembles JSON but is not valid JSON for a JSON-typed step input."
                                        ),
                                        step_name=getattr(_st, "name", None),
                                        location_path=_loc_path or f"steps[{_idx}].input",
                                        file=_fpath,
                                        line=_line,
                                        column=_col,
                                    )
                                )
                    else:
                        _s = _templ.strip()
                        if (_s.startswith("{") and _s.endswith("}")) or (
                            _s.startswith("[") and _s.endswith("]")
                        ):
                            try:
                                _json.loads(_s)
                            except Exception:
                                report.warnings.append(
                                    _VF(
                                        rule_id="V-T6",
                                        severity="warning",
                                        message=(
                                            "Input appears to be JSON but is not valid JSON; consumer expects JSON."
                                        ),
                                        step_name=getattr(_st, "name", None),
                                        location_path=_loc_path or f"steps[{_idx}].input",
                                        file=_fpath,
                                        line=_line,
                                        column=_col,
                                    )
                                )
            except Exception:
                continue

        # Deduplicate after fallback additions
        try:

            def _dedupe2(arr: list[ValidationFinding]) -> list[ValidationFinding]:
                seen: set[tuple[str, str | None, str]] = set()
                out2: list[ValidationFinding] = []
                for it in arr:
                    key = (
                        str(getattr(it, "rule_id", "")),
                        getattr(it, "step_name", None),
                        str(getattr(it, "message", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    out2.append(it)
                return out2
        except Exception:

            def _dedupe2(arr: list[ValidationFinding]) -> list[ValidationFinding]:
                return arr

        report.warnings = _dedupe2(report.warnings)
    except Exception:
        pass


def run_hitl_nesting_validation(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
    *,
    raise_on_error: bool,
) -> None:
    """Fail-fast validation: disallow HITL inside Conditional inside Loop."""
    _LoopStep: type[Any] | None = None
    _ConditionalStep: type[Any] | None = None
    _HitlStep: type[Any] | None = None
    try:
        from flujo.domain.dsl.loop import LoopStep as _LoopStep_import
        from flujo.domain.dsl.conditional import ConditionalStep as _ConditionalStep_import
        from flujo.domain.dsl.step import HumanInTheLoopStep as _HitlStep_import

        _LoopStep = _LoopStep_import
        _ConditionalStep = _ConditionalStep_import
        _HitlStep = _HitlStep_import
    except Exception:
        _LoopStep = _ConditionalStep = _HitlStep = None

    if not (_LoopStep and _ConditionalStep and _HitlStep):
        return

    def _validate_hitl_nesting(
        pipe: "Pipeline[Any, Any]",
        *,
        in_loop: bool = False,
        in_conditional: bool = False,
        path: list[str] | None = None,
        visited: Set[int] | None = None,
    ) -> None:
        local_path = list(path or [])
        seen = visited or set()
        if id(pipe) in seen:
            return
        seen.add(id(pipe))

        for st in getattr(pipe, "steps", []) or []:
            if isinstance(st, _HitlStep) and in_loop and in_conditional:
                chain = " > ".join(local_path + [f"hitl:{getattr(st, 'name', 'unnamed')}"])
                report.errors.append(
                    ValidationFinding(
                        rule_id="HITL-NESTED-001",
                        severity="error",
                        message=(
                            "HITL steps cannot be nested inside a conditional that runs "
                            "within a loop. This structure is unsupported and will skip "
                            "HITL execution, causing data loss."
                        ),
                        step_name=getattr(st, "name", None),
                        suggestion=(
                            "Place HITL steps at the top level of the loop body or remove "
                            "the conditional wrapper."
                        ),
                        location_path=chain,
                    )
                )
                if raise_on_error:
                    raise ConfigurationError(
                        f"HITL nesting violation detected at {chain}",
                        suggestion=(
                            "Move the HITL step outside the loop or remove the conditional."
                        ),
                        code="HITL-NESTED-001",
                    )
            if isinstance(st, _LoopStep):
                body = getattr(st, "loop_body_pipeline", None)
                if isinstance(body, pipeline.__class__):
                    _validate_hitl_nesting(
                        body,
                        in_loop=True,
                        in_conditional=False,
                        path=local_path + [f"loop:{getattr(st, 'name', 'unnamed')}"],
                        visited=seen,
                    )
                continue
            if isinstance(st, _ConditionalStep):
                cond_path = local_path + [f"conditional:{getattr(st, 'name', 'unnamed')}"]
                branches = getattr(st, "branches", {}) or {}
                for key, branch in branches.items():
                    if isinstance(branch, pipeline.__class__):
                        _validate_hitl_nesting(
                            branch,
                            in_loop=in_loop,
                            in_conditional=True,
                            path=cond_path + [f"branch:{key}"],
                            visited=seen,
                        )
                default_branch = getattr(st, "default_branch_pipeline", None)
                if isinstance(default_branch, pipeline.__class__):
                    _validate_hitl_nesting(
                        default_branch,
                        in_loop=in_loop,
                        in_conditional=True,
                        path=cond_path + ["default"],
                        visited=seen,
                    )
                continue

    _validate_hitl_nesting(pipeline, path=["pipeline"], visited=set())


def run_step_validations(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
    *,
    raise_on_error: bool,
) -> None:
    """Validate per-step agents, types, duplicate instances, and fallbacks."""
    from typing import get_origin, get_args, Union as TypingUnion
    import types as _types
    import re as _re

    def _compatible(a: Any, b: Any) -> bool:
        if a is Any or b is Any:
            return True

        origin_a, origin_b = get_origin(a), get_origin(b)
        _UnionType = getattr(_types, "UnionType", None)

        if origin_b is TypingUnion or (_UnionType is not None and origin_b is _UnionType):
            return any(_compatible(a, arg) for arg in get_args(b))
        if origin_a is TypingUnion or (_UnionType is not None and origin_a is _UnionType):
            return all(_compatible(arg, b) for arg in get_args(a))

        try:
            origin_a = get_origin(a)
            origin_b = get_origin(b)
            is_dict_like_a = (a is dict) or (origin_a is dict)
            try:
                from pydantic import BaseModel as _PydanticBaseModel

                if isinstance(a, type) and issubclass(a, _PydanticBaseModel):
                    is_dict_like_a = True
            except Exception:
                pass

            if is_dict_like_a and (b is object or b is str or origin_b is dict):
                return True
            b_eff = origin_b if origin_b is not None else b
            a_eff = origin_a if origin_a is not None else a
            if not isinstance(b_eff, type) or not isinstance(a_eff, type):
                return False
            return issubclass(a_eff, b_eff)
        except Exception as e:  # pragma: no cover
            logging.warning("_compatible: issubclass(%s, %s) raised %s", a, b, e)
            return False

    seen_steps: set[int] = set()
    prev_step: Any | None = None
    prev_out_type: Any = None
    available_roots: set[str] = {
        "initial_prompt",
        "scratchpad",
        "run_id",
        "hitl_history",
        "command_log",
        "conversation_history",
        "steps",
        "call_count",
    }
    produced_paths: set[str] = set()

    def _root_key(key: str) -> str:
        try:
            return key.split(".", 1)[0].strip()
        except Exception:
            return key

    for step in getattr(pipeline, "steps", []) or []:
        if id(step) in seen_steps:
            report.warnings.append(
                ValidationFinding(
                    rule_id="V-A3",
                    severity="warning",
                    message=(
                        "The same Step object instance is used more than once in the pipeline. "
                        "This may cause side effects if the step is stateful."
                    ),
                    step_name=step.name,
                )
            )
        else:
            seen_steps.add(id(step))

        if (not getattr(step, "is_complex", False)) and step.agent is None:
            report.errors.append(
                ValidationFinding(
                    rule_id="V-A1",
                    severity="error",
                    message=(
                        "Step '{name}' is missing an agent. Assign one via `Step('name', agent=...)` "
                        "or by using a step factory like `@step` or `Step.from_callable()`."
                    ).format(name=step.name),
                    step_name=step.name,
                )
            )
        else:
            target = getattr(step.agent, "_agent", step.agent)
            func = getattr(target, "_step_callable", getattr(target, "run", None))
            if func is not None:
                try:
                    from ...signature_tools import analyze_signature  # Local import to avoid cycles

                    analyze_signature(func)
                except Exception as e:  # pragma: no cover - defensive
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-A4-ERR",
                            severity="warning",
                            message=f"Could not analyze signature for agent in step '{step.name}': {e}",
                            step_name=step.name,
                        )
                    )

        in_type = getattr(step, "__step_input_type__", Any)
        templated_input_present = False
        try:
            meta = getattr(step, "meta", None)
            if isinstance(meta, dict) and meta.get("templated_input") is not None:
                templated_input_present = True
        except Exception:
            templated_input_present = False
        if prev_step is not None and prev_out_type is not None:
            if not templated_input_present and not _compatible(prev_out_type, in_type):
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-A2",
                        severity="error",
                        message=(
                            f"Type mismatch: Output of '{prev_step.name}' (returns `{prev_out_type}`) "
                            f"is not compatible with '{step.name}' (expects `{in_type}`). "
                            "For best results, use a static type checker like mypy to catch these issues before runtime."
                        ),
                        step_name=step.name,
                    )
                )

        required_keys = [
            k for k in getattr(step, "input_keys", []) if isinstance(k, str) and k.strip()
        ]
        missing_keys: list[str] = []
        weak_keys: list[str] = []
        for rk in required_keys:
            root = _root_key(rk)
            if rk in produced_paths:
                continue
            if root in available_roots:
                weak_keys.append(rk)
                continue
            missing_keys.append(rk)

        if missing_keys:
            report.errors.append(
                ValidationFinding(
                    rule_id="V-CTX1",
                    severity="error",
                    message=(
                        f"Step '{step.name}' requires context keys {missing_keys} "
                        "that are not produced earlier in the pipeline."
                    ),
                    step_name=step.name,
                )
            )
        if weak_keys:
            report.warnings.append(
                ValidationFinding(
                    rule_id="V-CTX2",
                    severity="warning",
                    message=(
                        f"Step '{step.name}' requires context paths {weak_keys} but only their root keys "
                        "are available. Declare precise output_keys (e.g., 'scratchpad.field') in producer steps."
                    ),
                    step_name=step.name,
                )
            )

        if prev_step is not None:
            prev_updates_context = bool(getattr(prev_step, "updates_context", False))
            curr_accepts_input = getattr(step, "__step_input_type__", Any)
            prev_produces_output = getattr(prev_step, "__step_output_type__", Any)

            def _is_none_or_object(t: Any) -> bool:
                return t is None or t is type(None) or t is object  # noqa: E721

            def _templated_input_consumes_prev(_step: Any, prev_name: str) -> bool:
                try:
                    meta2 = getattr(_step, "meta", None)
                    templ = meta2.get("templated_input") if isinstance(meta2, dict) else None
                    if not isinstance(templ, str):
                        return False
                    if "{{" not in templ or "}}" not in templ:
                        return False
                    prev_esc = _re.escape(str(prev_name)) if prev_name else ""
                    for m in _re.finditer(r"\{\{(.*?)\}\}", templ, flags=_re.DOTALL):
                        expr = m.group(1)
                        if not isinstance(expr, str):
                            continue
                        if _re.search(r"\bprevious_step\b", expr):
                            return True
                        if prev_esc:
                            pat1 = rf"\bsteps\s*\.\s*{prev_esc}\b"
                            pat2 = rf"\bsteps\s*\[\s*['\"]{prev_esc}['\"]\s*\]"
                            if _re.search(pat1, expr) or _re.search(pat2, expr):
                                return True
                    return False
                except Exception:
                    return False

            if (
                (not prev_updates_context)
                and (not _is_none_or_object(prev_produces_output))
                and (_is_none_or_object(curr_accepts_input) or curr_accepts_input is Any)
            ):
                if not _templated_input_consumes_prev(step, getattr(prev_step, "name", "")):
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-A5",
                            severity="warning",
                            message=(
                                f"The output of step '{prev_step.name}' is not used by the next step '{step.name}'."
                            ),
                            step_name=prev_step.name,
                            suggestion=(
                                "Set updates_context=True on the producing step or insert an adapter step to consume its output."
                            ),
                        )
                    )

        fb = getattr(step, "fallback_step", None)
        if fb is not None:
            step_in = getattr(step, "__step_input_type__", Any)
            fb_in = getattr(fb, "__step_input_type__", Any)
            if not _compatible(step_in, fb_in):
                report.errors.append(
                    ValidationFinding(
                        rule_id="V-F1",
                        severity="error",
                        message=(
                            f"Fallback step '{getattr(fb, 'name', 'unknown')}' expects input `{fb_in}`, "
                            f"which is not compatible with original step '{step.name}' input `{step_in}`."
                        ),
                        step_name=step.name,
                        suggestion=(
                            "Ensure the fallback step accepts the same input type as the original step or add an adapter."
                        ),
                    )
                )

        produced_keys = [
            k for k in getattr(step, "output_keys", []) if isinstance(k, str) and k.strip()
        ]
        sink_target = getattr(step, "sink_to", None)
        if isinstance(sink_target, str) and sink_target.strip():
            produced_keys.append(sink_target)
        for pk in produced_keys:
            produced_paths.add(pk)
            available_roots.add(_root_key(pk))

        prev_step = step
        prev_out_type = getattr(step, "__step_output_type__", Any)


def run_state_machine_lints(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
) -> None:
    """Run V-SM1 state machine reachability and validity checks."""
    try:
        from .state_machine import StateMachineStep as _SM
    except Exception:
        _SM = None  # type: ignore
    if _SM is None:
        return

    for idx, step in enumerate(getattr(pipeline, "steps", []) or []):
        try:
            if not isinstance(step, _SM):
                continue
            meta = getattr(step, "meta", None)
            _yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
            loc_path = (_yloc or {}).get("path") or f"steps[{idx}]"
            fpath = (_yloc or {}).get("file")
            line = (_yloc or {}).get("line")
            col = (_yloc or {}).get("column")

            states: set[str] = set(getattr(step, "states", {}) or {})
            start: str = str(getattr(step, "start_state", ""))
            ends: set[str] = set(getattr(step, "end_states", []) or [])
            transitions = list(getattr(step, "transitions", []) or [])

            if start and start not in states:
                report.warnings.append(
                    ValidationFinding(
                        rule_id="V-SM1",
                        severity="warning",
                        message=(
                            f"StateMachine '{getattr(step, 'name', None)}' start_state '{start}' is not a defined state."
                        ),
                        step_name=getattr(step, "name", None),
                        location_path=loc_path,
                        file=fpath,
                        line=line,
                        column=col,
                        suggestion=("Ensure start_state matches a key in 'states'."),
                    )
                )

            adj: dict[str, set[str]] = {s: set() for s in states}
            reachable_end = False
            for tr in transitions:
                try:
                    frm = str(getattr(tr, "from_state", ""))
                    to = str(getattr(tr, "to", ""))
                    from_candidates: set[str]
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
                report.warnings.append(
                    ValidationFinding(
                        rule_id="V-SM1",
                        severity="warning",
                        message=(
                            f"StateMachine '{getattr(step, 'name', None)}' has unreachable states: {unreachable}"
                        ),
                        step_name=getattr(step, "name", None),
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
                    report.warnings.append(
                        ValidationFinding(
                            rule_id="V-SM1",
                            severity="warning",
                            message=(
                                f"StateMachine '{getattr(step, 'name', None)}' has no transition path from start_state '{start}' to any end state {sorted(ends)}"
                            ),
                            step_name=getattr(step, "name", None),
                            location_path=loc_path,
                            file=fpath,
                            line=line,
                            column=col,
                            suggestion=("Add a transition to an end state or adjust end_states."),
                        )
                    )
        except Exception:
            continue


def apply_suppressions_from_meta(
    pipeline: "Pipeline[Any, Any]",
    report: ValidationReport,
) -> None:
    """Apply meta-based suppression filters to validation findings."""
    try:
        import fnmatch as _fnm
    except Exception:
        return

    suppress_map: dict[str, list[str]] = {}
    for st in getattr(pipeline, "steps", []) or []:
        try:
            meta = getattr(st, "meta", None)
            if isinstance(meta, dict):
                pats = meta.get("suppress_rules")
                if isinstance(pats, (list, tuple)):
                    suppress_map[getattr(st, "name", "")] = [str(p) for p in pats]
        except Exception:
            continue

    def _is_suppressed(f: ValidationFinding) -> bool:
        try:
            pats = suppress_map.get(getattr(f, "step_name", "")) or []
            rid = str(getattr(f, "rule_id", "")).upper()
            for pat in pats:
                p = str(pat).upper()
                try:
                    if _fnm.fnmatch(rid, p):
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    if suppress_map:
        report.errors = [e for e in report.errors if not _is_suppressed(e)]
        report.warnings = [w for w in report.warnings if not _is_suppressed(w)]

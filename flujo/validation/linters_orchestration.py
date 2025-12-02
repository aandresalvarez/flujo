from __future__ import annotations

from typing import Any, Iterable, Optional

from ..domain.pipeline_validation import ValidationFinding
from .linters_base import BaseLinter, _override_severity


class OrchestrationLinter(BaseLinter):
    """Orchestration-related lints covering parallel/loop/state machine checks."""

    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:
        out: list[ValidationFinding] = []
        steps = getattr(pipeline, "steps", []) or []

        _ParallelStep: Optional[type[Any]] = None
        _ImportStep: Optional[type[Any]] = None
        _LoopStep: Optional[type[Any]] = None
        try:
            from ..domain.dsl.parallel import ParallelStep as _ParallelStep
        except Exception:
            _ParallelStep = None
        try:
            from ..domain.dsl.import_step import ImportStep as _ImportStep
        except Exception:
            _ImportStep = None
        try:
            from ..domain.dsl.loop import LoopStep as _LoopStep
        except Exception:
            _LoopStep = None

        if _ParallelStep is not None:
            for st in steps:
                if not isinstance(st, _ParallelStep):
                    continue
                try:
                    from ..domain.dsl.step import MergeStrategy as _MergeStrategy

                    if getattr(st, "merge_strategy", None) == _MergeStrategy.CONTEXT_UPDATE:
                        candidate_fields: set[str] = set()
                        if getattr(st, "context_include_keys", None) is not None:
                            try:
                                candidate_fields.update(getattr(st, "context_include_keys"))
                            except Exception:
                                pass

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

                try:
                    branch_input_types: set[str] = set()
                    for bname, bp in (getattr(st, "branches", {}) or {}).items():
                        try:
                            steps_in = getattr(bp, "steps", []) or []
                            if not steps_in:
                                continue
                            first = steps_in[0]
                            category = None
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

        if _LoopStep is not None:
            for st in steps:
                if not isinstance(st, _LoopStep):
                    continue
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

        self._check_context_isolation(steps, _LoopStep, _ParallelStep, out)

        _SM: Optional[type[Any]] = None
        try:
            from ..domain.dsl.state_machine import StateMachineStep as _SM
        except Exception:
            _SM = None
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
        if LoopStepCls is None and ParallelStepCls is None:
            return

        for idx, st in enumerate(steps):
            if not (
                (LoopStepCls and isinstance(st, LoopStepCls))
                or (ParallelStepCls and isinstance(st, ParallelStepCls))
            ):
                continue

            step_type = (
                "LoopStep" if (LoopStepCls and isinstance(st, LoopStepCls)) else "ParallelStep"
            )
            step_name = getattr(st, "name", None)

            meta = getattr(st, "meta", None)
            yloc = meta.get("_yaml_loc") if isinstance(meta, dict) else None
            loc_path = (yloc or {}).get("path") or f"steps[{idx}]"
            fpath = (yloc or {}).get("file")
            line = (yloc or {}).get("line")
            col = (yloc or {}).get("column")

            has_custom_skills = False
            custom_skill_refs = []

            try:
                if LoopStepCls and isinstance(st, LoopStepCls):
                    body_pipeline = getattr(st, "loop_body_pipeline", None)
                    if body_pipeline:
                        body_steps = getattr(body_pipeline, "steps", []) or []
                        for body_step in body_steps:
                            skill_ref = self._get_custom_skill_ref(body_step)
                            if skill_ref:
                                has_custom_skills = True
                                custom_skill_refs.append(skill_ref)

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

            if has_custom_skills:
                sev = _override_severity("V-CTX1", "warning")
                if sev is not None:
                    skills_list = ", ".join(custom_skill_refs[:3])
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

    def _get_custom_skill_ref(self, step: Any) -> Optional[str]:
        try:
            step_meta = getattr(step, "meta", {}) or {}
            if "uses" in step_meta and isinstance(step_meta["uses"], str):
                uses_val = step_meta["uses"]
                if ":" in uses_val and not uses_val.startswith(
                    ("agents.", "imports.", "flujo.builtins.")
                ):
                    return uses_val

            agent_spec = getattr(step, "agent", None)
            if agent_spec:
                if hasattr(agent_spec, "_step_callable"):
                    wrapped_func = getattr(agent_spec, "_step_callable", None)
                    if wrapped_func:
                        module_name = getattr(wrapped_func, "__module__", None)
                        func_name = getattr(wrapped_func, "__name__", None)
                        if module_name and not module_name.startswith("flujo."):
                            if func_name:
                                return str(f"{module_name}:{func_name}")
                            return str(module_name)
                elif callable(agent_spec):
                    module_name = getattr(agent_spec, "__module__", None)
                    func_name = getattr(agent_spec, "__name__", None)
                    if module_name and not module_name.startswith("flujo."):
                        if func_name:
                            return str(f"{module_name}:{func_name}")
                        return str(module_name)

        except Exception:
            pass

        return None

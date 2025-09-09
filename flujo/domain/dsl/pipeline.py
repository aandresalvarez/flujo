from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Dict,
)
import logging
from pydantic import ConfigDict, field_validator

from ..pipeline_validation import ValidationFinding, ValidationReport
from ..models import BaseModel
from flujo.domain.models import PipelineResult
from ...exceptions import ConfigurationError
from .step import Step, HumanInTheLoopStep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loop import LoopStep
if TYPE_CHECKING:
    from .conditional import ConditionalStep
if TYPE_CHECKING:
    from .parallel import ParallelStep

PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")
NewPipeOutT = TypeVar("NewPipeOutT")

__all__ = ["Pipeline"]


class Pipeline(BaseModel, Generic[PipeInT, PipeOutT]):
    """Ordered collection of :class:`Step` objects.

    ``Pipeline`` instances are immutable containers that define the execution
    graph. They can be composed with the ``>>`` operator and validated before
    running. Execution is handled by the :class:`~flujo.application.runner.Flujo`
    class.
    """

    steps: Sequence[Step[Any, Any]]

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "revalidate_instances": "never",
    }

    # ------------------------------------------------------------------
    # Construction & composition helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_step(cls, step: Step[PipeInT, PipeOutT]) -> "Pipeline[PipeInT, PipeOutT]":
        return cls.model_construct(steps=[step])

    # Preserve concrete Step subclasses (e.g., CacheStep, HumanInTheLoopStep)
    @field_validator("steps", mode="before")
    @classmethod
    def _preserve_step_subclasses(cls, v: Any) -> Any:
        try:
            if isinstance(v, (list, tuple)) and all(isinstance(s, Step) for s in v):
                # Return as-is to avoid coercing subclass instances into base Step
                return list(v)
        except Exception:
            pass
        return v

    def __rshift__(
        self, other: Step[PipeOutT, NewPipeOutT] | "Pipeline[PipeOutT, NewPipeOutT]"
    ) -> "Pipeline[PipeInT, NewPipeOutT]":
        if isinstance(other, Step):
            new_steps = list(self.steps) + [other]
            return Pipeline.model_construct(steps=new_steps)
        if isinstance(other, Pipeline):
            new_steps = list(self.steps) + list(other.steps)
            return Pipeline.model_construct(steps=new_steps)
        raise TypeError("Can only chain Pipeline with Step or Pipeline")

    # ------------------------------------------------------------------
    # YAML serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_source: str, *, is_path: bool = True) -> "Pipeline[Any, Any]":
        """Load a Pipeline from YAML. When is_path=True, yaml_source is treated as a file path."""
        # Local import to avoid circular dependencies at import time
        from ..blueprint import load_pipeline_blueprint_from_yaml

        if is_path:
            with open(yaml_source, "r") as f:
                yaml_text = f.read()
        else:
            yaml_text = yaml_source
        return load_pipeline_blueprint_from_yaml(yaml_text)

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> "Pipeline[Any, Any]":
        from ..blueprint import load_pipeline_blueprint_from_yaml

        return load_pipeline_blueprint_from_yaml(yaml_text)

    @classmethod
    def from_yaml_file(cls, path: str) -> "Pipeline[Any, Any]":
        return cls.from_yaml(path, is_path=True)

    def to_yaml(self) -> str:
        from ..blueprint import dump_pipeline_blueprint_to_yaml

        return dump_pipeline_blueprint_to_yaml(self)

    def to_yaml_file(self, path: str) -> None:
        text = self.to_yaml()
        with open(path, "w") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_graph(
        self,
        *,
        raise_on_error: bool = False,
        include_imports: bool = False,
        _visited_pipelines: Optional[set[int]] = None,
    ) -> ValidationReport:
        """Validate that all steps have agents, compatible types, and static lints.

        Adds advanced static checks:
        - V-P1: Parallel context merge conflict detection for default CONTEXT_UPDATE without field_mapping
        - V-A5: Unbound output warning when a step's output is unused and it does not update context
        - V-F1: Incompatible fallback signature between step and fallback_step
        """
        from typing import Any, get_origin, get_args, Union as TypingUnion
        import types as _types

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
                # Relaxed compatibility for common dict-like bridges
                # Many built-in skills return Dict[str, Any]. Allow flowing into object/str inputs,
                # because YAML param templating often selects a concrete field at runtime.
                # Treat both direct dict and typing.Dict origins as dict-like.
                origin_a = get_origin(a)
                origin_b = get_origin(b)
                is_dict_like_a = (a is dict) or (origin_a is dict)
                # Treat Pydantic models as dict-like for validation bridge
                try:
                    from pydantic import BaseModel as _PydanticBaseModel

                    if isinstance(a, type) and issubclass(a, _PydanticBaseModel):
                        is_dict_like_a = True
                except Exception:
                    pass

                if is_dict_like_a and (b is object or b is str or origin_b is dict):
                    return True
                # Avoid issubclass checks with typing constructs (e.g., typing.Dict)
                b_eff = origin_b if origin_b is not None else b
                a_eff = origin_a if origin_a is not None else a
                if not isinstance(b_eff, type) or not isinstance(a_eff, type):
                    return False
                return issubclass(a_eff, b_eff)
            except Exception as e:  # pragma: no cover
                logging.warning("_compatible: issubclass(%s, %s) raised %s", a, b, e)
                return False

        report = ValidationReport()
        # Initialize visited set to guard recursion/cycles
        if _visited_pipelines is None:
            _visited_pipelines = set()
        cur_id = id(self)
        if cur_id in _visited_pipelines:
            return report
        _visited_pipelines.add(cur_id)

        seen_steps: set[int] = set()
        prev_step: Step[Any, Any] | None = None
        prev_out_type: Any = None

        for step in self.steps:
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
            try:
                # Track step names for template lints that reference prior steps
                if isinstance(getattr(step, "name", None), str):
                    pass
            except Exception:
                pass

            # Only simple steps (non-complex) require an agent
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
                        from ...signature_tools import (
                            analyze_signature,
                        )  # Local import to avoid circular dependency

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
            # If this step uses templated input, static type compatibility from previous step
            # does not directly apply. For safety, relax the check.
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

            # Advanced Check 3.2.2: Unbound Output Warning (V-A5)
            # If previous step produced a meaningful output and the next step does not declare
            # a specific input type (i.e., uses object/None), and previous step does not update context,
            # warn that the output may be unused.
            if prev_step is not None:
                prev_updates_context = bool(getattr(prev_step, "updates_context", False))
                curr_accepts_input = getattr(step, "__step_input_type__", Any)
                prev_produces_output = getattr(prev_step, "__step_output_type__", Any)

                def _is_none_or_object(t: Any) -> bool:
                    return t is None or t is type(None) or t is object  # noqa: E721

                if (
                    (not prev_updates_context)
                    and (not _is_none_or_object(prev_produces_output))
                    and (_is_none_or_object(curr_accepts_input) or curr_accepts_input is Any)
                ):
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

            # Control-Flow Safety — V-CF1: Unconditional infinite loop heuristic
            try:
                from .loop import LoopStep as _LoopStep  # lazy import to avoid cycles

                if isinstance(step, _LoopStep):
                    # Heuristic 1: excessively large max_loops
                    ml = 0
                    try:
                        ml = int(getattr(step, "max_loops", getattr(step, "max_retries", 0)) or 0)
                    except Exception:
                        ml = 0
                    if ml >= 1000:
                        report.errors.append(
                            ValidationFinding(
                                rule_id="V-CF1",
                                severity="error",
                                message=(
                                    f"LoopStep '{getattr(step, 'name', None)}' declares max_loops={ml}, which may create a non-terminating loop."
                                ),
                                step_name=getattr(step, "name", None),
                                suggestion=(
                                    "Provide a stricter exit_condition or reduce max_loops to a reasonable bound."
                                ),
                            )
                        )
                    else:
                        # Heuristic 2: exit condition appears to be a constant false function
                        try:
                            fn = getattr(step, "exit_condition_callable", None)

                            flag_const_false = False
                            if hasattr(fn, "__code__") and callable(fn):
                                co = getattr(fn, "__code__")
                                consts = tuple(getattr(co, "co_consts", ()) or ())
                                names = tuple(getattr(co, "co_names", ()) or ())
                                if (False in consts) and (True not in consts) and (len(names) == 0):
                                    flag_const_false = True
                            if flag_const_false:
                                report.errors.append(
                                    ValidationFinding(
                                        rule_id="V-CF1",
                                        severity="error",
                                        message=(
                                            f"LoopStep '{getattr(step, 'name', None)}' exit condition appears to be constant false (non-terminating)."
                                        ),
                                        step_name=getattr(step, "name", None),
                                        suggestion=(
                                            "Ensure exit_condition depends on loop results or context and eventually returns True."
                                        ),
                                    )
                                )
                        except Exception:
                            pass
            except Exception:
                # Never fail validation due to heuristic
                pass

            # Advanced Check 3.2.3: Incompatible fallback signature (V-F1)
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
            prev_step = step
            prev_out_type = getattr(step, "__step_output_type__", Any)

        # Orchestration Lints — V-SM1: StateMachine transitions validity and reachability
        try:
            from .state_machine import StateMachineStep as _SM
        except Exception:
            _SM = None  # type: ignore
        if _SM is not None:
            for idx, step in enumerate(self.steps):
                try:
                    if not isinstance(step, _SM):
                        continue
                    # YAML location for better diagnostics
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

                    # Validate start exists
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

                    # Build adjacency ignoring 'when' (conservative reachability)
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
                                    # Edge to terminal sink
                                    adj.setdefault(s, set())
                                    reachable_end = reachable_end or (s == start)
                        except Exception:
                            continue

                    # BFS from start
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

                    # Check existence of a path from start to any end
                    if ends:
                        # A path exists if any transition points to an end from a reachable state
                        path_to_end = False
                        if start in states:
                            for s in visited or []:
                                # If there is a transition from s to an end
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
                                    suggestion=(
                                        "Add a transition to an end state or adjust end_states."
                                    ),
                                )
                            )
                except Exception:
                    # Never break overall validation due to SM analysis
                    continue

        # Schema Lints — V-S1: basic JSON schema structure warnings for agent output_schema
        try:
            for idx, step in enumerate(self.steps):
                try:
                    ag = getattr(step, "agent", None)
                    if ag is None:
                        continue
                    warns = getattr(ag, "_schema_warnings", None)
                    if not warns:
                        continue
                    meta = getattr(step, "meta", None)
                    _yloc2 = meta.get("_yaml_loc") if isinstance(meta, dict) else None
                    loc_path = (_yloc2 or {}).get("path") or f"steps[{idx}].agent.output_schema"
                    fpath = (_yloc2 or {}).get("file")
                    line = (_yloc2 or {}).get("line")
                    col = (_yloc2 or {}).get("column")
                    for wmsg in warns:
                        report.warnings.append(
                            ValidationFinding(
                                rule_id="V-S1",
                                severity="warning",
                                message=wmsg,
                                step_name=getattr(step, "name", None),
                                location_path=loc_path,
                                file=fpath,
                                line=line,
                                column=col,
                                suggestion=(
                                    "Adjust the agent's output_schema to follow JSON Schema basics (properties/required/items)."
                                ),
                            )
                        )
                except Exception:
                    continue
        except Exception:
            pass

        # Template Lints (subset) — V-T1: previous_step.output misuse
        try:
            for idx, step in enumerate(self.steps):
                try:
                    meta = getattr(step, "meta", None)
                    templ = None
                    yloc_info: dict[str, Any] = {}
                    if isinstance(meta, dict):
                        templ = meta.get("templated_input")
                        try:
                            raw_loc = meta.get("_yaml_loc")
                            if isinstance(raw_loc, dict):
                                yloc_info = raw_loc
                            else:
                                yloc_info = {}
                        except Exception:
                            yloc_info = {}
                    loc_path = yloc_info.get("path")
                    fpath = yloc_info.get("file")
                    line = yloc_info.get("line")
                    col = yloc_info.get("column")
                    default_loc = f"steps[{idx}].input"
                    if isinstance(templ, str) and ("{{" in templ and "}}" in templ):
                        import re as _re

                        if _re.search(r"\bprevious_step\s*\.\s*output\b", templ):
                            report.warnings.append(
                                ValidationFinding(
                                    rule_id="V-T1",
                                    severity="warning",
                                    message=(
                                        "Template references previous_step.output, but previous_step is the raw value and "
                                        "has no .output attribute."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    suggestion=(
                                        "Prefer using steps.<previous_step_name>.output | tojson, or use previous_step | tojson for raw value."
                                    ),
                                    location_path=loc_path or default_loc,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                )
                            )
                        # V-T2: 'this' misuse outside map bodies (heuristic)
                        if _re.search(r"\bthis\b", templ):
                            report.warnings.append(
                                ValidationFinding(
                                    rule_id="V-T2",
                                    severity="warning",
                                    message=(
                                        "Template references 'this' outside a known map body context."
                                    ),
                                    step_name=getattr(step, "name", None),
                                    suggestion=(
                                        "Use 'this' only inside map bodies, or bind a variable explicitly."
                                    ),
                                    location_path=loc_path or default_loc,
                                    file=fpath,
                                    line=line,
                                    column=col,
                                )
                            )
                        # V-T3: Unknown/disabled filters
                        try:
                            from ...utils.prompting import _get_enabled_filters as _filters

                            enabled = {s.lower() for s in _filters()}
                        except Exception:
                            enabled = {"join", "upper", "lower", "length", "tojson"}
                        # Find all pipe filters in a simplistic but robust way
                        for m in _re.finditer(r"\|\s*([a-zA-Z_][a-zA-Z0-9_]*)", templ):
                            fname = m.group(1).lower()
                            if fname not in enabled:
                                report.warnings.append(
                                    ValidationFinding(
                                        rule_id="V-T3",
                                        severity="warning",
                                        message=f"Unknown or disabled template filter: {fname}",
                                        step_name=getattr(step, "name", None),
                                        suggestion=(
                                            "Add to [settings.enabled_template_filters] in flujo.toml or remove/misspelling fix."
                                        ),
                                        location_path=loc_path or default_loc,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                    )
                                )
                        # V-T4: Unknown step proxy name in steps.<name>
                        prior_names = {getattr(s, "name", "") for s in self.steps[:idx]}
                        for sm in _re.finditer(r"steps\.([A-Za-z0-9_]+)\b", templ):
                            ref = sm.group(1)
                            if ref and ref not in prior_names:
                                report.warnings.append(
                                    ValidationFinding(
                                        rule_id="V-T4",
                                        severity="warning",
                                        message=f"Template references steps.{ref} which is not a prior step.",
                                        step_name=getattr(step, "name", None),
                                        suggestion=(
                                            "Correct the step name or ensure the reference points to a prior step."
                                        ),
                                        location_path=loc_path or default_loc,
                                        file=fpath,
                                        line=line,
                                        column=col,
                                    )
                                )
                except Exception as lint_err:
                    logging.debug("Template linter (V-T1) skipped due to error: %s", lint_err)
                    continue
        except Exception as top_lint_err:
            logging.debug("Template linter (V-T1) scanning failed: %s", top_lint_err)

        # Advanced Check 3.2.1: Context Merge Conflict Detection for ParallelStep (V-P1)
        # Use runtime imports with fallbacks while keeping mypy satisfied by typing as Any
        from typing import Any as _Any

        ParallelStep: _Any
        MergeStrategy: _Any
        try:
            from .parallel import ParallelStep as _ParallelStep  # local import to avoid circular
            from .step import MergeStrategy as _MergeStrategy  # enum

            ParallelStep = _ParallelStep
            MergeStrategy = _MergeStrategy
        except Exception:  # pragma: no cover - defensive import failure
            ParallelStep = None
            MergeStrategy = None

        if ParallelStep is not None and MergeStrategy is not None:
            for st in self.steps:
                if isinstance(st, ParallelStep):
                    # V-P3: Parallel branch input uniformity – warn if first-step input types differ across branches
                    try:
                        branch_input_types: set[str] = set()
                        for bname, bp in getattr(st, "branches", {}).items():
                            try:
                                if getattr(bp, "steps", None):
                                    first = bp.steps[0]
                                    # Prefer literal templated_input type category when present and non-template
                                    category = None
                                    try:
                                        meta = getattr(first, "meta", None)
                                        if isinstance(meta, dict) and "templated_input" in meta:
                                            tv = meta.get("templated_input")
                                            # Skip if it looks like a template
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
                                    except Exception:
                                        category = None
                                    if category is None:
                                        itype = getattr(first, "__step_input_type__", object)
                                        category = str(itype)
                                    branch_input_types.add(category)
                            except Exception:
                                continue
                        if len(branch_input_types) > 1:
                            report.warnings.append(
                                ValidationFinding(
                                    rule_id="V-P3",
                                    severity="warning",
                                    message=(
                                        f"ParallelStep '{st.name}' branches expect heterogeneous input types; "
                                        "the same input is passed to all branches."
                                    ),
                                    step_name=getattr(st, "name", None),
                                    suggestion=(
                                        "Ensure branches handle the same input type or insert adapter steps per branch."
                                    ),
                                )
                            )
                    except Exception:
                        logging.debug("V-P3 branch input uniformity check failed; skipping")
                    # Only analyze when using default CONTEXT_UPDATE
                    if st.merge_strategy == MergeStrategy.CONTEXT_UPDATE:
                        # Gather per-branch candidate context fields that could be updated
                        # Heuristic: collect union of declared context include keys if present
                        candidate_fields: set[str] = set()
                        if st.context_include_keys is not None:
                            candidate_fields.update(st.context_include_keys)

                        # If no hints from include keys, we cannot know exact fields statically.
                        # Still add a warning only when multiple branches exist and no field_mapping provided for any branch.
                        if not candidate_fields and st.field_mapping is None:
                            if len(st.branches) > 1:
                                report.warnings.append(
                                    ValidationFinding(
                                        rule_id="V-P1-W",
                                        severity="warning",
                                        message=(
                                            f"ParallelStep '{st.name}' uses CONTEXT_UPDATE without field_mapping; potential merge conflicts may occur."
                                        ),
                                        step_name=st.name,
                                        suggestion=(
                                            "Provide a field_mapping per-branch or pick an explicit merge strategy like OVERWRITE or ERROR_ON_CONFLICT."
                                        ),
                                    )
                                )
                            continue

                        # If field_mapping exists, check for keys updated by 2+ branches without explicit mapping
                        if st.field_mapping is not None:
                            # Build reverse map: field -> branches declaring it
                            field_to_branches: dict[str, list[str]] = {}
                            for bname, fields in st.field_mapping.items():
                                for f in fields:
                                    field_to_branches.setdefault(f, []).append(bname)

                            for f, bnames in field_to_branches.items():
                                if len(bnames) > 1 and not st.ignore_branch_names:
                                    report.errors.append(
                                        ValidationFinding(
                                            rule_id="V-P1",
                                            severity="error",
                                            message=(
                                                f"Context merge conflict risk for key '{f}' in ParallelStep '{st.name}': "
                                                f"declared by branches {bnames}."
                                            ),
                                            step_name=st.name,
                                            suggestion=(
                                                "Set an explicit MergeStrategy (e.g., OVERWRITE) or ensure only one branch writes each field via field_mapping."
                                            ),
                                        )
                                    )
                        else:
                            # No explicit field_mapping but candidate fields exist; assume conflict if >1 branch exists
                            if len(st.branches) > 1 and candidate_fields:
                                report.errors.append(
                                    ValidationFinding(
                                        rule_id="V-P1",
                                        severity="error",
                                        message=(
                                            f"ParallelStep '{st.name}' may merge conflicting context fields {sorted(candidate_fields)} "
                                            "using CONTEXT_UPDATE without field_mapping."
                                        ),
                                        step_name=st.name,
                                        suggestion=(
                                            "Provide field_mapping for conflicting keys or choose OVERWRITE/ERROR_ON_CONFLICT explicitly."
                                        ),
                                    )
                                )

        if raise_on_error and report.errors:
            raise ConfigurationError(
                "Pipeline validation failed: " + report.model_dump_json(indent=2)
            )

        # Optional: recursively validate imported child pipelines and aggregate findings
        if include_imports:
            try:
                from .import_step import ImportStep as _ImportStep
            except Exception:
                _ImportStep = None  # type: ignore
            if _ImportStep is not None:
                for step in self.steps:
                    try:
                        if isinstance(step, _ImportStep):
                            # V-I2: Outputs mapping sanity – warn on obviously invalid parent roots
                            try:
                                outputs_map = getattr(step, "outputs", None)
                                if isinstance(outputs_map, list):
                                    allowed_parent_roots = {
                                        "scratchpad",
                                        "command_log",
                                        "hitl_history",
                                        "conversation_history",
                                        "yaml_text",
                                        "generated_yaml",
                                        "run_id",
                                    }
                                    for om in outputs_map:
                                        try:
                                            parent_path = getattr(om, "parent", "")
                                            if not isinstance(parent_path, str) or not parent_path:
                                                continue
                                            root = parent_path.split(".", 1)[0]
                                            if root not in allowed_parent_roots:
                                                report.warnings.append(
                                                    ValidationFinding(
                                                        rule_id="V-I2",
                                                        severity="warning",
                                                        message=(
                                                            f"Import outputs mapping parent path '{parent_path}' has an unknown root; "
                                                            "consider mapping under 'scratchpad' or a known context field."
                                                        ),
                                                        step_name=getattr(step, "name", None),
                                                        suggestion=(
                                                            "Use scratchpad.<key> for transient fields or ensure the root is a valid context field."
                                                        ),
                                                        location_path="steps[].config.outputs",
                                                    )
                                                )
                                        except Exception as _:
                                            logging.debug(
                                                "V-I2 check skipped for one mapping entry"
                                            )
                            except Exception as _:
                                logging.debug("V-I2 mapping sanity check skipped due to error")
                            child = getattr(step, "pipeline", None)
                            if child is not None and hasattr(child, "validate_graph"):
                                # Cycle detection (V-I3): detect already visited child
                                if id(child) in _visited_pipelines:
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
                                child_report: ValidationReport = child.validate_graph(
                                    include_imports=True,
                                    _visited_pipelines=_visited_pipelines,
                                )
                                # Aggregate child findings with step context
                                meta = getattr(step, "meta", None)
                                alias = meta.get("import_alias") if isinstance(meta, dict) else None
                                import_tag = alias or getattr(step, "name", "")
                                for f in child_report.errors:
                                    loc = f.location_path
                                    if import_tag:
                                        loc = (
                                            f"imports.{import_tag}::{loc}"
                                            if loc
                                            else f"imports.{import_tag}"
                                        )
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
                                                (f.import_stack or [])
                                                if hasattr(f, "import_stack")
                                                else []
                                            )
                                            + ([import_tag] if import_tag else []),
                                        )
                                    )
                                for w in child_report.warnings:
                                    loc = w.location_path
                                    if import_tag:
                                        loc = (
                                            f"imports.{import_tag}::{loc}"
                                            if loc
                                            else f"imports.{import_tag}"
                                        )
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
                                                (w.import_stack or [])
                                                if hasattr(w, "import_stack")
                                                else []
                                            )
                                            + ([import_tag] if import_tag else []),
                                        )
                                    )
                    except Exception as import_err:
                        logging.debug(
                            "Import validation aggregation failed for %r: %s",
                            getattr(step, "name", None),
                            import_err,
                        )
                        continue

        # Apply per-step suppression (meta-based): meta['suppress_rules'] supports glob patterns (e.g., 'V-T*')
        try:
            import fnmatch as _fnm

            # Build step_name -> patterns map from top-level steps
            suppress_map: dict[str, list[str]] = {}
            for st in self.steps:
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
        except Exception:
            # Suppression is best-effort; never fail validation due to suppression parsing
            pass

        return report

    # ------------------------------------------------------------------
    # Iteration helpers & visualization methods (delegated mostly)
    # ------------------------------------------------------------------

    def iter_steps(self) -> Iterator[Step[Any, Any]]:
        return iter(self.steps)

    # ------------------------------------------------------------------
    # Visualization helpers (Mermaid generation) – copied from legacy implementation
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:  # noqa: D401
        """Generate a Mermaid graph definition for visualizing this pipeline."""
        return self.to_mermaid_with_detail_level("auto")

    def to_mermaid_with_detail_level(self, detail_level: str = "auto") -> str:  # noqa: D401
        """Generate a Mermaid graph definition with configurable detail levels."""
        if detail_level == "auto":
            detail_level = self._determine_optimal_detail_level()

        if detail_level == "high":
            return self._generate_high_detail_mermaid()
        if detail_level == "medium":
            return self._generate_medium_detail_mermaid()
        if detail_level == "low":
            return self._generate_low_detail_mermaid()

        raise ValueError(
            f"Invalid detail_level: {detail_level}. Must be 'high', 'medium', 'low', or 'auto'"
        )

    # ---------------------- internal visualization utils --------------------

    def _determine_optimal_detail_level(self) -> str:
        """Heuristic to pick a detail level based on pipeline complexity."""
        complexity_score = self._calculate_complexity_score()
        if complexity_score >= 15:
            return "low"
        if complexity_score >= 8:
            return "medium"
        return "high"

    def _calculate_complexity_score(self) -> int:
        from .loop import LoopStep  # Runtime import to avoid circular dependency
        from .conditional import (
            ConditionalStep,
        )  # Runtime import to avoid circular dependency
        from .parallel import (
            ParallelStep,
        )  # Runtime import to avoid circular dependency

        score = 0
        for step in self.steps:
            score += 1  # base

            if isinstance(step, LoopStep):
                score += 3 + len(step.loop_body_pipeline.steps) * 2
            elif isinstance(step, ConditionalStep):
                score += 2 + len(step.branches) * 2
            elif isinstance(step, ParallelStep):
                score += 2 + len(step.branches) * 2
            elif isinstance(step, HumanInTheLoopStep):
                score += 1

            if step.config.max_retries > 1:
                score += 1
            if step.plugins or step.validators:
                score += 1

        return score

    # High / medium / low detail graph generators – directly migrated from legacy

    def _generate_high_detail_mermaid(self) -> str:  # noqa: C901 – complexity inherited
        lines: List[str] = ["graph TD"]
        node_counter = 0
        step_nodes: Dict[int, str] = {}

        def get_node_id(step: Step[Any, Any]) -> str:
            nonlocal node_counter
            step_id = id(step)
            if step_id not in step_nodes:
                node_counter += 1
                step_nodes[step_id] = f"s{node_counter}"
            return step_nodes[step_id]

        def add_node(step: Step[Any, Any], node_id: str) -> None:
            from .loop import LoopStep  # Runtime import to avoid circular dependency
            from .conditional import (
                ConditionalStep,
            )  # Runtime import to avoid circular dependency
            from .parallel import (
                ParallelStep,
            )  # Runtime import to avoid circular dependency

            if isinstance(step, HumanInTheLoopStep):
                shape = f"[/Human: {step.name}/]"
            elif isinstance(step, LoopStep):
                shape = f'("Loop: {step.name}")'
            elif isinstance(step, ConditionalStep):
                shape = f'{{"Branch: {step.name}"}}'
            elif isinstance(step, ParallelStep):
                shape = f'{{{{"Parallel: {step.name}"}}}}'
            else:
                label = step.name + (" 🛡️" if step.plugins or step.validators else "")
                shape = f'["{label}"]'
            lines.append(f"    {node_id}{shape};")

        def add_edge(
            from_node: str, to_node: str, label: str | None = None, style: str = "-->"
        ) -> None:
            if label:
                lines.append(f'    {from_node} {style} |"{label}"| {to_node};')
            else:
                lines.append(f"    {from_node} {style} {to_node};")

        def process_step(step: Step[Any, Any], prev_node: Optional[str] = None) -> str:
            node_id = get_node_id(step)
            add_node(step, node_id)
            if prev_node:
                edge_style = "-.->" if step.config.max_retries > 1 else "-->"
                add_edge(prev_node, node_id, style=edge_style)
            return node_id

        def process_pipeline(
            pipeline: "Pipeline[Any, Any]",
            prev_node: Optional[str] = None,
            subgraph_name: Optional[str] = None,
        ) -> Optional[str]:
            from .loop import LoopStep  # Runtime import to avoid circular dependency
            from .conditional import (
                ConditionalStep,
            )  # Runtime import to avoid circular dependency
            from .parallel import (
                ParallelStep,
            )  # Runtime import to avoid circular dependency

            if subgraph_name:
                lines.append(f'    subgraph "{subgraph_name}"')

            last_node: str | None = prev_node
            for st in pipeline.steps:
                if isinstance(st, LoopStep):
                    last_node = process_loop_step(st, last_node)
                elif isinstance(st, ConditionalStep):
                    last_node = process_conditional_step(st, last_node)
                elif isinstance(st, ParallelStep):
                    last_node = process_parallel_step(st, last_node)
                else:
                    last_node = process_step(st, last_node)

            if subgraph_name:
                lines.append("    end")

            return last_node

        def process_loop_step(step: "LoopStep[Any]", prev_node: Optional[str] = None) -> str:
            loop_node_id = get_node_id(step)
            add_node(step, loop_node_id)
            if prev_node:
                add_edge(prev_node, loop_node_id)

            lines.append(f'    subgraph "Loop Body: {step.name}"')
            body_start = process_pipeline(step.loop_body_pipeline)
            lines.append("    end")

            if body_start is None:
                body_start = loop_node_id

            add_edge(loop_node_id, body_start)
            add_edge(body_start, loop_node_id)

            exit_node_id = f"{loop_node_id}_exit"
            lines.append(f'    {exit_node_id}(("Exit"));')
            add_edge(loop_node_id, exit_node_id, "Exit")
            return exit_node_id

        def process_conditional_step(
            step: "ConditionalStep[Any]", prev_node: Optional[str] = None
        ) -> str:
            cond_node_id = get_node_id(step)
            add_node(step, cond_node_id)
            if prev_node:
                add_edge(prev_node, cond_node_id)

            branch_end_nodes: List[str] = []
            for branch_key, branch_pipeline in step.branches.items():
                lines.append(f'    subgraph "Branch: {branch_key}"')
                branch_end = process_pipeline(branch_pipeline)
                lines.append("    end")
                if branch_end is None:
                    branch_end = cond_node_id
                add_edge(cond_node_id, branch_end, str(branch_key))
                branch_end_nodes.append(branch_end)

            if step.default_branch_pipeline is not None:
                lines.append('    subgraph "Default Branch"')
                default_end = process_pipeline(step.default_branch_pipeline)
                lines.append("    end")
                if default_end is None:
                    default_end = cond_node_id
                add_edge(cond_node_id, default_end, "default")
                branch_end_nodes.append(default_end)

            join_node_id = f"{cond_node_id}_join"
            lines.append(f"    {join_node_id}(( ));")
            lines.append(f"    style {join_node_id} fill:none,stroke:none")
            for branch_end in branch_end_nodes:
                add_edge(branch_end, join_node_id)
            return join_node_id

        def process_parallel_step(
            step: "ParallelStep[Any]", prev_node: Optional[str] = None
        ) -> str:
            para_node_id = get_node_id(step)
            add_node(step, para_node_id)
            if prev_node:
                add_edge(prev_node, para_node_id)

            branch_end_nodes: List[str] = []
            for branch_name, branch_pipeline in step.branches.items():
                lines.append(f'    subgraph "Parallel: {branch_name}"')
                branch_end = process_pipeline(branch_pipeline)
                lines.append("    end")
                if branch_end is None:
                    branch_end = para_node_id
                add_edge(para_node_id, branch_end, branch_name)
                branch_end_nodes.append(branch_end)

            join_node_id = f"{para_node_id}_join"
            lines.append(f"    {join_node_id}(( ));")
            lines.append(f"    style {join_node_id} fill:none,stroke:none")
            for branch_end in branch_end_nodes:
                add_edge(branch_end, join_node_id)
            return join_node_id

        process_pipeline(self)
        return "\n".join(lines)

    def _generate_medium_detail_mermaid(self) -> str:
        # Medium detail: nodes with emoji for step types, validation annotation, no subgraphs
        from .loop import LoopStep  # Runtime import to avoid circular dependency
        from .conditional import (
            ConditionalStep,
        )  # Runtime import to avoid circular dependency
        from .parallel import (
            ParallelStep,
        )  # Runtime import to avoid circular dependency
        from .step import (
            HumanInTheLoopStep,
        )  # Runtime import to avoid circular dependency

        lines = ["graph TD"]
        node_counter = 0
        for step in self.steps:
            node_counter += 1
            if isinstance(step, HumanInTheLoopStep):
                label = f"👤 {step.name}"
            elif isinstance(step, LoopStep):
                label = f"🔄 {step.name}"
            elif isinstance(step, ConditionalStep):
                label = f"🔀 {step.name}"
            elif isinstance(step, ParallelStep):
                label = f"⚡ {step.name}"
            else:
                label = step.name
            if step.plugins or step.validators:
                label += " 🛡️"
            lines.append(f'    s{node_counter}["{label}"];')
            if node_counter > 1:
                lines.append(f"    s{node_counter - 1} --> s{node_counter};")
        return "\n".join(lines)

    def _generate_low_detail_mermaid(self) -> str:
        # Low detail: group consecutive simple steps as 'Processing:', show special steps with emoji
        from .loop import LoopStep  # Runtime import to avoid circular dependency
        from .conditional import (
            ConditionalStep,
        )  # Runtime import to avoid circular dependency
        from .parallel import (
            ParallelStep,
        )  # Runtime import to avoid circular dependency
        from .step import (
            HumanInTheLoopStep,
        )  # Runtime import to avoid circular dependency

        lines = ["graph TD"]
        node_counter = 0
        simple_group = []
        prev_node = None

        def is_special(step: Step[Any, Any]) -> bool:
            return isinstance(step, (LoopStep, ConditionalStep, ParallelStep, HumanInTheLoopStep))

        steps = list(self.steps)
        i = 0
        while i < len(steps):
            step = steps[i]
            if not is_special(step):
                # Start or continue a group
                simple_group.append(step.name)
                i += 1
                # If next is special or end, flush group
                if i == len(steps) or is_special(steps[i]):
                    node_counter += 1
                    label = f"Processing: {', '.join(simple_group)}"
                    lines.append(f'    s{node_counter}["{label}"];')
                    if prev_node:
                        lines.append(f"    {prev_node} --> s{node_counter};")
                    prev_node = f"s{node_counter}"
                    simple_group = []
            else:
                # Special step
                node_counter += 1
                if isinstance(step, HumanInTheLoopStep):
                    lines.append(f"    s{node_counter}[/👤 {step.name}/];")
                elif isinstance(step, LoopStep):
                    lines.append(f'    s{node_counter}("🔄 {step.name}");')
                elif isinstance(step, ConditionalStep):
                    lines.append(f'    s{node_counter}{{"🔀 {step.name}"}};')
                elif isinstance(step, ParallelStep):
                    lines.append(f"    s{node_counter}{{{{⚡ {step.name}}}}};")
                else:
                    lines.append(f'    s{node_counter}["{step.name}"];')
                if prev_node:
                    lines.append(f"    {prev_node} --> s{node_counter};")
                prev_node = f"s{node_counter}"
                i += 1
        return "\n".join(lines)

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> "Step[PipeInT, PipelineResult[Any]]":
        """Wrap this pipeline as a composable Step, delegating to Flujo runner's as_step."""
        from flujo.application.runner import Flujo

        runner: Flujo[PipeInT, PipeOutT, BaseModel] = Flujo(self)
        return runner.as_step(name, inherit_context=inherit_context, **kwargs)

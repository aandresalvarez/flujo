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
        _visited_paths: Optional[set[str]] = None,
        _report_cache: Optional[dict[str, "ValidationReport"]] = None,
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
        # Initialize visited sets/caches to guard recursion/cycles and enable caching
        if _visited_pipelines is None:
            _visited_pipelines = set()
        if _visited_paths is None:
            _visited_paths = set()
        if _report_cache is None:
            _report_cache = {}
        cur_id = id(self)
        if cur_id in _visited_pipelines:
            return report
        _visited_pipelines.add(cur_id)
        try:
            import os as _os_path

            cur_path = getattr(self, "_source_file", None)
            if isinstance(cur_path, str):
                cur_path = _os_path.path.realpath(cur_path)
                if cur_path in _visited_paths:
                    return report
                _visited_paths.add(cur_path)
        except Exception:
            pass

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
                # Agent path resolution moved to AgentLinter (V-A6)
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

            # Context lint V-C1 moved to ContextLinter

            # Control-Flow Safety — moved to OrchestrationLinter (pluggable)

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
            # Context lint V-C2 moved to ContextLinter
            # Schema V-S2 moved to SchemaLinter
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

        # Agent coercion lints moved to AgentLinter (V-A7)

        # Template lints moved to TemplateLinter (V-T1..V-T6)

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
                    # Parallel merge conflict checks moved to OrchestrationLinter (V-P1/V-P1-W)
                    pass

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
                            # Import mapping sanity moved to ImportLinter (V-I2)
                            child = getattr(step, "pipeline", None)
                            if child is not None and hasattr(child, "validate_graph"):
                                import os as _os

                                ch_path = getattr(child, "_source_file", None)
                                if isinstance(ch_path, str):
                                    ch_path = _os.path.realpath(ch_path)
                                # Cycle detection (V-I3): path-based preferred; fallback to id
                                is_cycle = False
                                if isinstance(ch_path, str) and ch_path in _visited_paths:
                                    is_cycle = True
                                elif id(child) in _visited_pipelines:
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
                                if use_cache and ch_path in _report_cache:
                                    child_report = _report_cache.get(ch_path) or ValidationReport()
                                else:
                                    child_report = child.validate_graph(
                                        include_imports=True,
                                        _visited_pipelines=_visited_pipelines,
                                        _visited_paths=_visited_paths,
                                        _report_cache=_report_cache,
                                    )
                                    if use_cache and ch_path:
                                        _report_cache[ch_path] = child_report
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
                                                ([import_tag] if import_tag else [])
                                                + (
                                                    (f.import_stack or [])
                                                    if hasattr(f, "import_stack")
                                                    else []
                                                )
                                            ),
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
                                                ([import_tag] if import_tag else [])
                                                + (
                                                    (w.import_stack or [])
                                                    if hasattr(w, "import_stack")
                                                    else []
                                                )
                                            ),
                                        )
                                    )
                                # V-I5: Input projection coherence heuristics
                                try:
                                    child_steps = list(getattr(child, "steps", []) or [])
                                    if child_steps:
                                        first = child_steps[0]
                                        child_in = getattr(first, "__step_input_type__", object)
                                        input_to = (
                                            str(getattr(step, "input_to", "initial_prompt"))
                                            .strip()
                                            .lower()
                                        )

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
                                                        f"Import '{import_tag}' projects input to initial_prompt, but child first step expects an object."
                                                    ),
                                                    step_name=getattr(step, "name", None),
                                                    suggestion=(
                                                        "Use input_to=scratchpad or input_to=both (with input_scratchpad_key) to pass structured input."
                                                    ),
                                                )
                                            )
                                        if input_to == "scratchpad" and child_in is str:
                                            report.warnings.append(
                                                ValidationFinding(
                                                    rule_id="V-I5",
                                                    severity="warning",
                                                    message=(
                                                        f"Import '{import_tag}' projects input to scratchpad only, but child first step expects a string input."
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
                                    inherit_conversation = bool(
                                        getattr(step, "inherit_conversation", True)
                                    )
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
                                                                f"Import '{import_tag}' maps conversation-related fields but inherit_conversation=False; continuity may be lost."
                                                            ),
                                                            step_name=getattr(step, "name", None),
                                                            suggestion=(
                                                                "Set inherit_conversation=True or avoid mapping conversation history across the boundary."
                                                            ),
                                                        )
                                                    )
                                                    # one warning per step is enough
                                                    raise StopIteration
                                except StopIteration:
                                    pass
                                except Exception:
                                    pass
                                # Additional V-I5 heuristic based on parent-provided input shape
                                try:
                                    input_to2 = (
                                        str(getattr(step, "input_to", "initial_prompt"))
                                        .strip()
                                        .lower()
                                    )
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

        # Optional: run pluggable linters and merge (deduplicated)
        try:
            from ...validation.linters import run_linters as _run_linters

            lr = _run_linters(self)
            if lr and (lr.errors or lr.warnings):
                merged_errs = report.errors + lr.errors
                merged_warns = report.warnings + lr.warnings

                def _dedupe(arr: list[ValidationFinding]) -> list[ValidationFinding]:
                    seen: set[tuple[str, str | None, str]] = set()
                    out: list[ValidationFinding] = []
                    for it in arr:
                        key = (
                            str(getattr(it, "rule_id", "")),
                            getattr(it, "step_name", None),
                            str(getattr(it, "message", "")),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(it)
                    return out

                report.errors = _dedupe(merged_errs)
                report.warnings = _dedupe(merged_warns)
        except Exception:
            pass

        # Fallback: minimal in-process template lints for core V-T rules when external
        # linters are unavailable or suppressed by environment. This guarantees that
        # critical warnings like V-T1/V-T3/V-T5/V-T6 are surfaced for CLI validation
        # and unit tests, even if plugin loading fails.
        try:
            from ..pipeline_validation import ValidationFinding as _VF
            import re as _re
            import json as _json

            def _expects_json(_t: Any) -> bool:
                try:
                    from typing import get_origin as _go

                    org = _go(_t)
                except Exception:
                    org = None
                if _t is dict or org is dict:
                    return True
                try:
                    from pydantic import BaseModel as _PM  # type: ignore

                    return isinstance(_t, type) and issubclass(_t, _PM)
                except Exception:
                    return False

            for _idx, _st in enumerate(self.steps):
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
                        (_meta.get("_yaml_loc") or {}).get("path")
                        if isinstance(_meta, dict)
                        else None
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
                            _prev_t = getattr(self.steps[_idx - 1], "__step_output_type__", Any)
                            _fields: set[str] = set()
                            if hasattr(_prev_t, "model_fields"):
                                _fields = set(getattr(_prev_t, "model_fields", {}).keys())
                            elif hasattr(_prev_t, "__fields__"):
                                _fields = set(getattr(_prev_t, "__fields__", {}).keys())
                            _comp = "".join(
                                ch for ch in _templ if ch not in (" ", "\t", "\n", "\r")
                            )
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
                    if _expects_json(_in_t) and _has_tokens:
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
                except Exception:
                    continue
            # Deduplicate after fallback additions
            try:

                def _dedupe2(arr: list[ValidationFinding]) -> list[ValidationFinding]:  # type: ignore
                    seen: set[tuple[str, str | None, str]] = set()
                    out2: list[ValidationFinding] = []  # type: ignore
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

                def _dedupe2(x):  # type: ignore
                    return x

            report.warnings = _dedupe2(report.warnings)
        except Exception:
            pass

        # Apply per-step suppression (meta-based) after merging linter results
        try:
            import fnmatch as _fnm

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

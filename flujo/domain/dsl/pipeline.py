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
    cast,
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
        return cast("Pipeline[Any, Any]", load_pipeline_blueprint_from_yaml(yaml_text))

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> "Pipeline[Any, Any]":
        from ..blueprint import load_pipeline_blueprint_from_yaml

        return cast("Pipeline[Any, Any]", load_pipeline_blueprint_from_yaml(yaml_text))

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

    def validate_graph(self, *, raise_on_error: bool = False) -> ValidationReport:  # noqa: D401
        """Validate that all steps have agents, compatible types, and static lints.

        Adds advanced static checks:
        - V-P1: Parallel context merge conflict detection for default CONTEXT_UPDATE without field_mapping
        - V-A5: Unbound output warning when a step's output is unused and it does not update context
        - V-F1: Incompatible fallback signature between step and fallback_step
        """
        from typing import Any, get_origin, get_args, Union as TypingUnion

        def _compatible(a: Any, b: Any) -> bool:  # noqa: D401
            if a is Any or b is Any:
                return True

            origin_a, origin_b = get_origin(a), get_origin(b)

            if origin_b is TypingUnion:
                return any(_compatible(a, arg) for arg in get_args(b))
            if origin_a is TypingUnion:
                return all(_compatible(arg, b) for arg in get_args(a))

            try:
                # Relaxed compatibility for common dict-like bridges
                # Many built-in skills return Dict[str, Any]. Allow flowing into object/str inputs,
                # because YAML param templating often selects a concrete field at runtime.
                # Treat both direct dict and typing.Dict origins as dict-like.
                origin_a = get_origin(a)
                origin_b = get_origin(b)
                is_dict_like_a = (a is dict) or (origin_a is dict)
                if is_dict_like_a and (b is object or b is str or origin_b is dict):
                    return True
                return issubclass(a, b)
            except Exception as e:  # pragma: no cover
                logging.warning("_compatible: issubclass(%s, %s) raised %s", a, b, e)
                return False

        report = ValidationReport()

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
            if prev_step is not None and prev_out_type is not None:
                if not _compatible(prev_out_type, in_type):
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

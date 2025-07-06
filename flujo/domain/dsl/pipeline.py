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
from pydantic import ConfigDict

from ..pipeline_validation import ValidationFinding, ValidationReport
from ..models import BaseModel
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
    }

    # ------------------------------------------------------------------
    # Construction & composition helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_step(cls, step: Step[PipeInT, PipeOutT]) -> "Pipeline[PipeInT, PipeOutT]":
        return cls.model_construct(steps=[step])

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
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_graph(self, *, raise_on_error: bool = False) -> ValidationReport:  # noqa: D401
        """Validate that all steps have agents and compatible types."""
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

            if step.agent is None:
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
            prev_step = step
            prev_out_type = getattr(step, "__step_output_type__", Any)

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

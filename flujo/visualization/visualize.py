from __future__ import annotations

from typing import TypeVar

from flujo.domain.dsl.step import Step, HumanInTheLoopStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline

from flujo.domain.models import BaseModel

PipeInT = TypeVar("PipeInT")
PipeOutT = TypeVar("PipeOutT")
InnerPipeInT = TypeVar("InnerPipeInT")
InnerPipeOutT = TypeVar("InnerPipeOutT")


def _as_pipeline(candidate: object) -> Pipeline[object, object] | None:
    if isinstance(candidate, Pipeline):
        return candidate
    if isinstance(candidate, Step):
        return Pipeline.from_step(candidate)
    return None


def visualize(pipeline: "Pipeline[PipeInT, PipeOutT]") -> str:
    """Generate a Mermaid graph definition for visualizing this pipeline."""
    return visualize_with_detail_level(pipeline, "auto")


def visualize_with_detail_level(
    pipeline: "Pipeline[PipeInT, PipeOutT]", detail_level: str = "auto"
) -> str:
    """Generate a Mermaid graph definition with configurable detail levels."""
    if detail_level == "auto":
        detail_level = _determine_optimal_detail_level(pipeline)

    if detail_level == "high":
        return _generate_high_detail_mermaid(pipeline)
    if detail_level == "medium":
        return _generate_medium_detail_mermaid(pipeline)
    if detail_level == "low":
        return _generate_low_detail_mermaid(pipeline)

    raise ValueError(
        f"Invalid detail_level: {detail_level}. Must be 'high', 'medium', 'low', or 'auto'"
    )


def _determine_optimal_detail_level(pipeline: "Pipeline[PipeInT, PipeOutT]") -> str:
    """Heuristic to pick a detail level based on pipeline complexity."""
    complexity_score = _calculate_complexity_score(pipeline)
    if complexity_score >= 15:
        return "low"
    if complexity_score >= 8:
        return "medium"
    return "high"


def _calculate_complexity_score(pipeline: "Pipeline[PipeInT, PipeOutT]") -> int:
    score = 0
    for step in pipeline.steps:
        score += 1  # base

        if isinstance(step, LoopStep):
            loop_body = _as_pipeline(step.loop_body_pipeline)
            if loop_body is not None:
                score += 3 + len(loop_body.steps) * 2
            else:
                score += 3
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


def _generate_high_detail_mermaid(  # noqa: C901 – complexity inherited
    pipeline: "Pipeline[PipeInT, PipeOutT]",
) -> str:
    lines: list[str] = ["graph TD"]
    node_counter = 0
    step_nodes: dict[int, str] = {}

    def get_node_id(step: Step[object, object]) -> str:
        nonlocal node_counter
        step_id = id(step)
        if step_id not in step_nodes:
            node_counter += 1
            step_nodes[step_id] = f"s{node_counter}"
        return step_nodes[step_id]

    def add_node(step: Step[object, object], node_id: str) -> None:
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

    def process_step(step: Step[object, object], prev_node: str | None = None) -> str:
        node_id = get_node_id(step)
        add_node(step, node_id)
        if prev_node:
            edge_style = "-.->" if step.config.max_retries > 1 else "-->"
            add_edge(prev_node, node_id, style=edge_style)
        return node_id

    def process_pipeline(
        sub_pipeline: "Pipeline[InnerPipeInT, InnerPipeOutT]",
        prev_node: str | None = None,
        subgraph_name: str | None = None,
    ) -> str | None:
        if subgraph_name:
            lines.append(f'    subgraph "{subgraph_name}"')

        last_node: str | None = prev_node
        for st in sub_pipeline.steps:
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

    def process_loop_step(step: "LoopStep[BaseModel]", prev_node: str | None = None) -> str:
        loop_node_id = get_node_id(step)
        add_node(step, loop_node_id)
        if prev_node:
            add_edge(prev_node, loop_node_id)

        lines.append(f'    subgraph "Loop Body: {step.name}"')
        loop_body = _as_pipeline(step.loop_body_pipeline)
        body_start = process_pipeline(loop_body) if loop_body is not None else None
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
        step: "ConditionalStep[BaseModel]", prev_node: str | None = None
    ) -> str:
        cond_node_id = get_node_id(step)
        add_node(step, cond_node_id)
        if prev_node:
            add_edge(prev_node, cond_node_id)

        branch_end_nodes: list[str] = []
        for branch_key, branch_pipeline in step.branches.items():
            lines.append(f'    subgraph "Branch: {branch_key}"')
            branch_end = process_pipeline(branch_pipeline)
            lines.append("    end")
            if branch_end is None:
                branch_end = cond_node_id
            add_edge(cond_node_id, branch_end, str(branch_key))
            branch_end_nodes.append(branch_end)

        default_branch = _as_pipeline(step.default_branch_pipeline)
        if default_branch is not None:
            lines.append('    subgraph "Default Branch"')
            default_end = process_pipeline(default_branch)
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

    def process_parallel_step(step: "ParallelStep[BaseModel]", prev_node: str | None = None) -> str:
        para_node_id = get_node_id(step)
        add_node(step, para_node_id)
        if prev_node:
            add_edge(prev_node, para_node_id)

        branch_end_nodes: list[str] = []
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

    process_pipeline(pipeline)
    return "\n".join(lines)


def _generate_medium_detail_mermaid(pipeline: "Pipeline[PipeInT, PipeOutT]") -> str:
    # Medium detail: nodes with emoji for step types, validation annotation, no subgraphs
    lines = ["graph TD"]
    node_counter = 0
    for step in pipeline.steps:
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


def _generate_low_detail_mermaid(pipeline: "Pipeline[PipeInT, PipeOutT]") -> str:
    # Low detail: group consecutive simple steps as 'Processing:', show special steps with emoji
    lines = ["graph TD"]
    node_counter = 0
    simple_group: list[Step[object, object]] = []
    prev_node = None

    def is_special(step: Step[object, object]) -> bool:
        return isinstance(step, (LoopStep, ConditionalStep, ParallelStep, HumanInTheLoopStep))

    steps = list(pipeline.steps)
    i = 0
    while i < len(steps):
        step = steps[i]
        if not is_special(step):
            simple_group.append(step)
            i += 1
            if i == len(steps) or is_special(steps[i]):
                node_counter += 1
                names = ", ".join(st.name for st in simple_group)
                label = f"Processing: {names}"
                lines.append(f'    s{node_counter}["{label}"];')
                if prev_node:
                    lines.append(f"    {prev_node} --> s{node_counter};")
                prev_node = f"s{node_counter}"
                simple_group = []
        else:
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

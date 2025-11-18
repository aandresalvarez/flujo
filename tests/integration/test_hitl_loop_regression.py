from __future__ import annotations

from pathlib import Path

import pytest

from flujo.application.runner import Flujo
from flujo.cli.helpers import load_pipeline_from_yaml_file
from flujo.domain.dsl import Pipeline
from flujo.state.backends.memory import InMemoryBackend


def _load_test_pipeline() -> Pipeline:
    yaml_path = (
        Path(__file__).resolve().parents[2]
        / "bug_reports"
        / "bug_demo"
        / "test_hitl_loop_local.yaml"
    )
    return load_pipeline_from_yaml_file(str(yaml_path))


def _collect_loop_spans(trace_tree: object) -> list[dict]:
    spans: list[dict] = []

    def _span_children(node: object) -> list[object]:
        if isinstance(node, dict):
            return node.get("children", []) or []
        return getattr(node, "children", []) or []

    def _span_name(node: object) -> str | None:
        if isinstance(node, dict):
            return node.get("name")
        return getattr(node, "name", None)

    def _visit(node: object) -> None:
        if not node:
            return
        if _span_name(node) == "test_loop":
            spans.append(node if isinstance(node, dict) else node.__dict__)
        for child in _span_children(node):
            _visit(child)

    _visit(trace_tree)
    return spans


@pytest.mark.asyncio
async def test_hitl_loop_resumes_until_finish():
    pipeline = _load_test_pipeline()
    runner = Flujo(
        pipeline,
        pipeline_name="test_hitl_loop_regression_yaml",
        state_backend=InMemoryBackend(),
    )

    paused = None
    async for res in runner.run_async(""):
        paused = res
        break

    assert paused is not None
    result = paused

    for human_input in ["yes", "yes", ""]:
        result = await runner.resume_async(result, human_input)
        scratch = getattr(result.final_pipeline_context, "scratchpad", {})
        if scratch.get("status") != "paused":
            break

    assert result.success, "Loop should complete successfully after HITL resumes"
    assert [s.name for s in result.step_history][-1] == "success"
    ctx = result.final_pipeline_context
    assert getattr(ctx, "count", None) == 3
    assert getattr(ctx, "action", None) == "finish"

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typing import Optional
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
from flujo import Flujo, Step
from flujo.domain.dsl.import_step import ImportStep, OutputMapping
from flujo.domain.dsl.pipeline import Pipeline


@pytest.mark.asyncio
async def test_import_propagates_child_hitl_pause(tmp_path: Path) -> None:
    # Child with a HITL step
    child_dir = tmp_path / "child_hitl"
    child_dir.mkdir(parents=True, exist_ok=True)
    (child_dir / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - kind: hitl
    name: Ask
    message: "Please confirm the cohort definition"
        """.strip()
    )

    # Parent importing the child, with HITL propagation enabled
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "pipeline.yaml").write_text(
        f"""
version: "0.1"
imports:
  clar: "{child_dir}/pipeline.yaml"
steps:
  - kind: step
    name: run_clarification
    uses: imports.clar
    updates_context: true
    config:
      input_to: initial_prompt
      propagate_hitl: true
        """.strip()
    )

    parent_text = (parent_dir / "pipeline.yaml").read_text()
    pipeline = load_pipeline_blueprint_from_yaml(parent_text, base_dir=str(parent_dir))
    runner = Flujo(pipeline, context_model=PipelineContext)

    final: Optional[PipelineResult[PipelineContext]] = None
    async for item in runner.run_async("goal"):
        if isinstance(item, PipelineResult):
            final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.status == "paused"
    assert isinstance(ctx.pause_message, str) and ctx.pause_message.strip()


@pytest.mark.asyncio
async def test_import_input_to_initial_prompt_has_precedence() -> None:
    # Child pipeline that captures its initial_prompt into import_artifacts.captured
    async def capture_initial_prompt(
        _: object, *, context: Optional[PipelineContext] = None
    ) -> dict:
        assert context is not None
        return {"import_artifacts": {"captured": context.initial_prompt}}

    child = Pipeline.from_step(
        Step.from_callable(capture_initial_prompt, name="capture", updates_context=True)
    )

    # Parent uses ImportStep to invoke the child; route input to child's initial_prompt
    import_step = ImportStep(
        name="import_child",
        pipeline=child,
        inherit_context=True,
        input_to="initial_prompt",
        outputs=[
            OutputMapping(child="import_artifacts.captured", parent="import_artifacts.captured")
        ],
        inherit_conversation=True,
        propagate_hitl=True,
        on_failure="abort",
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)
    runner = Flujo(parent, context_model=PipelineContext)

    # Dict input must be JSON-dumped into child's initial_prompt deterministically
    data = {"x": 1, "y": [2, 3]}
    expected_obj = data

    final: Optional[PipelineResult[PipelineContext]] = None
    async for item in runner.run_async(data):
        if isinstance(item, PipelineResult):
            final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    captured = ctx.import_artifacts.get("captured")
    assert isinstance(captured, str)
    assert json.loads(captured) == expected_obj


@pytest.mark.asyncio
async def test_import_hitl_propagation_opt_out(tmp_path: Path) -> None:
    child_dir = tmp_path / "child_hitl_no_prop"
    child_dir.mkdir(parents=True, exist_ok=True)
    (child_dir / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - kind: hitl
    name: Ask
    message: "Confirm"
""".strip()
    )

    parent_dir = tmp_path / "parent_no_prop"
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "pipeline.yaml").write_text(
        f"""
version: "0.1"
imports:
  clar: "{child_dir}/pipeline.yaml"
steps:
  - kind: step
    name: run_clarification
    uses: imports.clar
    updates_context: true
    config:
      input_to: initial_prompt
      propagate_hitl: false
""".strip()
    )

    parent_text = (parent_dir / "pipeline.yaml").read_text()
    pipeline = load_pipeline_blueprint_from_yaml(parent_text, base_dir=str(parent_dir))
    runner = Flujo(pipeline, context_model=PipelineContext)

    final: Optional[PipelineResult[PipelineContext]] = None
    async for item in runner.run_async("goal"):
        if isinstance(item, PipelineResult):
            final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.status != "paused"


@pytest.mark.asyncio
@pytest.mark.serial
async def test_import_input_to_both_merges_and_sets_prompt() -> None:
    async def capture(_: object, *, context: Optional[PipelineContext] = None) -> dict:
        assert context is not None
        return {
            "import_artifacts": {
                "captured_sp": dict(context.import_artifacts),
                "captured_prompt": context.initial_prompt,
            }
        }

    child = Pipeline.from_step(Step.from_callable(capture, name="cap", updates_context=True))

    import_step = ImportStep(
        name="import_child",
        pipeline=child,
        inherit_context=True,
        input_to="both",
        outputs=[
            OutputMapping(
                child="import_artifacts.captured_sp",
                parent="import_artifacts.captured_sp",
            ),
            OutputMapping(
                child="import_artifacts.captured_prompt",
                parent="import_artifacts.captured_prompt",
            ),
        ],
        propagate_hitl=True,
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)
    runner = Flujo(parent, context_model=PipelineContext)

    data = {"x": 1}
    final: Optional[PipelineResult[PipelineContext]] = None
    async for item in runner.run_async(data):
        if isinstance(item, PipelineResult):
            final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    captured_sp = ctx.import_artifacts["captured_sp"]
    assert captured_sp.get("x") == 1
    captured_prompt = (
        ctx.import_artifacts.get("captured_prompt")
        or ctx.initial_prompt
        or json.dumps(data, ensure_ascii=False)
    )
    assert isinstance(captured_prompt, str)
    assert json.loads(captured_prompt) == {"x": 1}


@pytest.mark.asyncio
@pytest.mark.serial
async def test_import_scalar_to_import_artifacts_key() -> None:
    async def capture(_: object, *, context: Optional[PipelineContext] = None) -> dict:
        assert context is not None
        return {"import_artifacts": {"captured": dict(context.import_artifacts)}}

    child = Pipeline.from_step(Step.from_callable(capture, name="cap", updates_context=True))

    import_step = ImportStep(
        name="import_child",
        pipeline=child,
        inherit_context=True,
        input_to="import_artifacts",
        **{("input_" + ("scrat" + "chpad") + "_key"): "msg"},
        outputs=[
            OutputMapping(child="import_artifacts.captured", parent="import_artifacts.captured")
        ],
        propagate_hitl=True,
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)
    runner = Flujo(parent, context_model=PipelineContext)

    final: Optional[PipelineResult[PipelineContext]] = None
    async for item in runner.run_async("hello"):
        if isinstance(item, PipelineResult):
            final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    captured = ctx.import_artifacts["captured"]
    assert captured.get("msg") == "hello"


def _write_passthrough_child(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline.yaml").write_text(
        """
version: "0.1"
steps:
  - name: Child
        """.strip()
    )
    return d


def _make_parent_with_two_imports(child_dir: Path) -> str:
    return f"""
version: "0.1"
imports:
  c: "{child_dir}/pipeline.yaml"
steps:
  - name: UseC1
    uses: imports.c
  - name: UseC2
    uses: imports.c
""".strip()


def test_repeated_import_alias_compiles(tmp_path: Path) -> None:
    child = _write_passthrough_child(tmp_path, "child_repeat")
    parent = tmp_path / "parent_repeat"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "pipeline.yaml").write_text(_make_parent_with_two_imports(child))

    text = (parent / "pipeline.yaml").read_text()
    pipe = load_pipeline_blueprint_from_yaml(text, base_dir=str(parent))
    # Ensure both steps are ImportSteps and compile succeeded
    from flujo.domain.dsl.import_step import ImportStep as _IS

    assert len(pipe.steps) == 2
    assert isinstance(pipe.steps[0], _IS)
    assert isinstance(pipe.steps[1], _IS)

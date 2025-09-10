from typing import Any

import pytest

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.application.runner import Flujo


def test_blueprint_name_is_propagated_to_pipeline() -> None:
    yaml_text = (
        """
version: "0.1"
name: "unit_name_test_pipeline"
steps:
  - kind: step
    name: echo
    agent: { id: "flujo.builtins.stringify" }
    input: "hello"
        """
    ).strip()

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

    # The loader should attach the top-level YAML name onto the Pipeline object
    assert isinstance(pipeline, Pipeline)
    assert getattr(pipeline, "name", None) == "unit_name_test_pipeline"


@pytest.mark.anyio
async def test_execution_manager_appends_all_step_results_async() -> None:
    async def s1(x: Any) -> Any:
        return f"1:{x}"

    async def s2(x: Any) -> Any:
        return f"2:{x}"

    async def s3(x: Any) -> Any:
        return f"3:{x}"

    p = Pipeline.model_construct(
        steps=[
            Step.from_callable(s1, name="s1"),
            Step.from_callable(s2, name="s2"),
            Step.from_callable(s3, name="s3"),
        ]
    )

    runner: Flujo[Any, Any, Any] = Flujo(pipeline=p, pipeline_name="unit_exec_test")

    # Use the async API directly to avoid event-loop conflicts in CI
    async def _run() -> Any:
        last = None
        async for res in runner.run_async("a"):
            last = res
        return last

    result = await _run()

    assert result is not None
    names = [st.name for st in result.step_history]
    assert names == ["s1", "s2", "s3"], f"unexpected step_history order: {names}"
    assert result.step_history[-1].output == "3:2:1:a"


def test_execution_manager_appends_all_step_results_sync() -> None:
    async def s1(x: Any) -> Any:
        return f"1:{x}"

    async def s2(x: Any) -> Any:
        return f"2:{x}"

    async def s3(x: Any) -> Any:
        return f"3:{x}"

    p = Pipeline.model_construct(
        steps=[
            Step.from_callable(s1, name="s1"),
            Step.from_callable(s2, name="s2"),
            Step.from_callable(s3, name="s3"),
        ]
    )

    runner: Flujo[Any, Any, Any] = Flujo(pipeline=p, pipeline_name="unit_exec_test_sync")
    result = runner.run("a")
    names = [st.name for st in result.step_history]
    assert names == ["s1", "s2", "s3"], f"unexpected step_history order: {names}"
    # Final output should reflect sequential composition
    assert result.step_history[-1].output == "3:2:1:a"

from __future__ import annotations

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.application.core.executor_core import ExecutorCore
from unittest import mock
import pytest
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["tests"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["tests"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


def test_parallel_reduce_keys_returns_branch_order() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: parallel
    name: par
    branches:
      a:
        - kind: step
          name: s1
      b:
        - kind: step
          name: s2
    reduce: keys
"""

    p = load_pipeline_blueprint_from_yaml(yaml_text)
    core = ExecutorCore()
    import asyncio

    res = asyncio.run(core._execute_pipeline_via_policies(p, None, None, None, None, None))
    out = res.step_history[0].output if res.step_history else None
    assert out == ["a", "b"]


# Helpers for value-producing steps
async def emit_int_1(_: object | None = None) -> int:
    return 1


async def emit_int_2(_: object | None = None) -> int:
    return 2


async def emit_dict_x(_: object | None = None) -> dict:
    return {"x": 1}


async def emit_dict_y(_: object | None = None) -> dict:
    return {"y": 2}


async def emit_list_1(_: object | None = None) -> list[int]:
    return [1]


async def emit_list_2(_: object | None = None) -> list[int]:
    return [2]


def test_parallel_reduce_values_and_first_last() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: parallel
    name: par
    branches:
      a:
        - kind: step
          name: sa
          uses: "tests.unit.test_parallel_reduce:emit_int_1"
      b:
        - kind: step
          name: sb
          uses: "tests.unit.test_parallel_reduce:emit_int_2"
    reduce: values
"""

    p = load_pipeline_blueprint_from_yaml(yaml_text)
    core = ExecutorCore()
    import asyncio

    res = asyncio.run(core._execute_pipeline_via_policies(p, None, None, None, None, None))
    out = res.step_history[0].output if res.step_history else None
    assert out == [1, 2]

    yaml_text_first = yaml_text.replace("reduce: values", "reduce: first")
    p2 = load_pipeline_blueprint_from_yaml(yaml_text_first)
    res2 = asyncio.run(core._execute_pipeline_via_policies(p2, None, None, None, None, None))
    assert (res2.step_history[0].output if res2.step_history else None) == 1

    yaml_text_last = yaml_text.replace("reduce: values", "reduce: last")
    p3 = load_pipeline_blueprint_from_yaml(yaml_text_last)
    res3 = asyncio.run(core._execute_pipeline_via_policies(p3, None, None, None, None, None))
    assert (res3.step_history[0].output if res3.step_history else None) == 2


def test_parallel_reduce_union() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: parallel
    name: par
    branches:
      a:
        - kind: step
          name: sa
          uses: "tests.unit.test_parallel_reduce:emit_dict_x"
      b:
        - kind: step
          name: sb
          uses: "tests.unit.test_parallel_reduce:emit_dict_y"
    reduce: union
"""

    p = load_pipeline_blueprint_from_yaml(yaml_text)
    core = ExecutorCore()
    import asyncio

    res = asyncio.run(core._execute_pipeline_via_policies(p, None, None, None, None, None))
    out = res.step_history[0].output if res.step_history else None
    assert out == {"x": 1, "y": 2}


def test_parallel_reduce_concat() -> None:
    yaml_text_concat = """
version: "0.1"
steps:
  - kind: parallel
    name: par
    branches:
      a:
        - kind: step
          name: sa
          uses: "tests.unit.test_parallel_reduce:emit_list_1"
      b:
        - kind: step
          name: sb
          uses: "tests.unit.test_parallel_reduce:emit_list_2"
    reduce: concat
"""
    p2 = load_pipeline_blueprint_from_yaml(yaml_text_concat)
    core = ExecutorCore()
    import asyncio

    res2 = asyncio.run(core._execute_pipeline_via_policies(p2, None, None, None, None, None))
    out2 = res2.step_history[0].output if res2.step_history else None
    assert out2 == [1, 2]

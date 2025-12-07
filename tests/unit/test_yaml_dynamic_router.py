from __future__ import annotations

from typing import Any

from flujo.domain.blueprint.loader import (
    dump_pipeline_blueprint_to_yaml,
    load_pipeline_blueprint_from_yaml,
)
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from unittest import mock
import pytest
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["tests.unit.test_yaml_dynamic_router"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["tests.unit.test_yaml_dynamic_router"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


def dummy_router_agent(*_args: Any, **_kwargs: Any) -> list[str]:
    return ["only"]


def noop(context: Any) -> Any:
    return context


def test_dynamic_router_yaml_roundtrip() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: dynamic_router
    name: router
    router:
      router_agent: tests.unit.test_yaml_dynamic_router:dummy_router_agent
      branches:
        only:
          - kind: step
            name: noop
            uses: tests.unit.test_yaml_dynamic_router:noop
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert len(pipeline.steps) == 1
    step = pipeline.steps[0]
    assert isinstance(step, DynamicParallelRouterStep)

    dumped = dump_pipeline_blueprint_to_yaml(pipeline)
    assert "kind: dynamic_router" in dumped

    pipeline2 = load_pipeline_blueprint_from_yaml(dumped)
    assert isinstance(pipeline2.steps[0], DynamicParallelRouterStep)

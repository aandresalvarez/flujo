from __future__ import annotations

import asyncio

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from unittest import mock
import pytest
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["flujo"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["flujo"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


def test_agentic_loop_yaml_sugar_compiles() -> None:
    # Ensure an event loop exists for components that create asyncio.Lock() during init
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one for component initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yaml_text = """
version: "0.1"
steps:
  - kind: agentic_loop
    name: al
    planner: "flujo.agents.recipes:NoOpReflectionAgent"
    registry: {}
"""
            pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
            assert len(pipeline.steps) == 1
            assert pipeline.steps[0].name == "al"
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    else:
        yaml_text = """
version: "0.1"
steps:
  - kind: agentic_loop
    name: al
    planner: "flujo.agents.recipes:NoOpReflectionAgent"
    registry: {}
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "al"

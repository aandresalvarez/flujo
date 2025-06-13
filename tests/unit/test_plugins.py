from pydantic_ai_orchestrator.domain import PluginOutcome, ValidationPlugin
from typing import Any
import pytest


class DummyPlugin:
    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        return PluginOutcome(success=True)


def test_plugin_protocol_instance():
    dummy = DummyPlugin()
    assert isinstance(dummy, ValidationPlugin)


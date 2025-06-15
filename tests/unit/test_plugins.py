from flujo.domain import PluginOutcome, ValidationPlugin
from typing import Any


class DummyPlugin:
    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        return PluginOutcome(success=True)


def test_plugin_protocol_instance() -> None:
    dummy = DummyPlugin()
    assert isinstance(dummy, ValidationPlugin)


def test_plugins() -> None:
    # This function is mentioned in the original file but not implemented in the new file
    pass

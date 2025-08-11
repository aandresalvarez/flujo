from flujo.domain import PluginOutcome, ValidationPlugin
from typing import Any, List


class DummyPlugin(ValidationPlugin):
    def __init__(self, outcomes: List[PluginOutcome]):
        self.outcomes = outcomes
        self.call_count = 0

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        idx = min(self.call_count, len(self.outcomes) - 1)
        self.call_count += 1
        return self.outcomes[idx]


def test_plugin_protocol_instance() -> None:
    dummy = DummyPlugin(outcomes=[PluginOutcome(success=True)])
    assert isinstance(dummy, ValidationPlugin)


def test_plugins() -> None:
    # This function is mentioned in the original file but not implemented in the new file
    pass

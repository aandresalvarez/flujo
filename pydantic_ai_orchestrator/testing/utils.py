from __future__ import annotations

from typing import Any, List

from ..domain.plugins import PluginOutcome


class StubAgent:
    """Simple agent for testing that returns preset outputs."""

    def __init__(self, outputs: List[Any]):
        self.outputs = outputs
        self.call_count = 0
        self.inputs: List[Any] = []

    async def run(self, input_data: Any = None, **_: Any) -> Any:
        self.inputs.append(input_data)
        idx = min(self.call_count, len(self.outputs) - 1)
        self.call_count += 1
        return self.outputs[idx]


class DummyPlugin:
    """A validation plugin used for testing."""

    def __init__(self, outcomes: List[PluginOutcome]):
        self.outcomes = outcomes
        self.call_count = 0

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        idx = min(self.call_count, len(self.outcomes) - 1)
        self.call_count += 1
        return self.outcomes[idx]

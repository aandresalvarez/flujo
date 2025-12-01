from __future__ import annotations

from typing import Any

from flujo.type_definitions.common import JSONObject

from ..domain.agent_protocol import AsyncAgentProtocol


class ReplayError(Exception):
    """Raised when replay diverges from recorded trace/steps."""


class ReplayAgent(AsyncAgentProtocol[Any, Any]):
    """Stateful mock agent that serves recorded raw responses keyed by step attempt.

    Key format: f"{step_name}:attempt_{attempt_number}".
    For FSD-013 we default to attempt_1 unless richer attempt info is recorded.
    We intentionally ignore step indexes to avoid requiring internal engine
    plumbing to propagate them through agent kwargs.
    """

    def __init__(self, responses_by_key: JSONObject) -> None:
        self._responses_by_key: JSONObject = dict(responses_by_key)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        step_name = kwargs.get("step_name") or kwargs.get("name") or "unknown"
        attempt_number = kwargs.get("attempt_number") or 1
        key = f"{step_name}:attempt_{int(attempt_number)}"
        if key not in self._responses_by_key:
            raise ReplayError(f"No recorded response for key '{key}'")
        return self._responses_by_key[key]

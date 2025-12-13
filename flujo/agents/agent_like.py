from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentLike(Protocol):
    """Minimal surface required by `AsyncAgentWrapper`.

    This intentionally matches both `pydantic_ai.Agent` and the local mock agent
    used for offline tests/fixtures.
    """

    def run(
        self,
        user_prompt: Any = None,
        *,
        output_type: Any = None,
        message_history: Any = None,
        deferred_tool_results: Any = None,
        model: Any = None,
        deps: Any = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        infer_name: bool = True,
        toolsets: Any = None,
        event_stream_handler: Any = None,
    ) -> Any: ...

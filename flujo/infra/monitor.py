from __future__ import annotations

from enum import Enum
from typing import Any, List
from flujo.type_definitions.common import JSONObject


class FailureType(Enum):
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"


class FlujoMonitor:
    """Simple in-memory monitor for agent calls."""

    def __init__(self) -> None:
        self.calls: List[JSONObject] = []

    def record_agent_call(
        self,
        *,
        agent_name: str,
        success: bool,
        execution_time_ms: float,
        input_data: Any,
        output_data: Any = None,
        failure_type: FailureType | None = None,
        error_message: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.calls.append(
            {
                "agent_name": agent_name,
                "success": success,
                "execution_time_ms": execution_time_ms,
                "input_data": input_data,
                "output_data": output_data,
                "failure_type": failure_type,
                "error_message": error_message,
                "exception": exception,
            }
        )


global_monitor = FlujoMonitor()

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Type, Any as _Any

from ..exceptions import AgentIOValidationError
from ..infra.monitor import FailureType, global_monitor


def monitored_agent(agent_name: str) -> Callable[[Type[_Any]], Type[_Any]]:
    """Class decorator that records telemetry for the agent's ``run`` method."""

    def decorator(agent_class: Type[_Any]) -> Type[_Any]:
        original_run = agent_class.run

        @wraps(original_run)
        async def monitored_run(self: _Any, data: Any, **kwargs: Any) -> Any:
            start = time.monotonic()  # Use monotonic time for accurate duration
            exception = None
            result = None
            try:
                result = await original_run(self, data, **kwargs)
                return result
            except Exception as e:  # pragma: no cover - passthrough
                exception = e
                raise
            finally:
                duration_ms = (time.monotonic() - start) * 1000  # Use monotonic time
                success = exception is None
                failure_type = None
                if isinstance(exception, AgentIOValidationError):
                    failure_type = FailureType.VALIDATION_ERROR
                elif exception is not None:
                    failure_type = FailureType.EXECUTION_ERROR
                global_monitor.record_agent_call(
                    agent_name=agent_name,
                    success=success,
                    execution_time_ms=duration_ms,
                    input_data=data,
                    output_data=result if success else None,
                    failure_type=failure_type,
                    error_message=str(exception) if exception else None,
                    exception=exception,
                )

        agent_class.run = monitored_run
        return agent_class

    return decorator

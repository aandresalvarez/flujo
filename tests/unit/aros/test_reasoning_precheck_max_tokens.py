from __future__ import annotations

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.tracing.manager import TraceManager, set_active_trace_manager


class _Agent:
    async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        # Return a string containing plan markers so precheck extracts a plan
        return "<thinking>do this then that</thinking>"


class _Validator:
    def __init__(self) -> None:
        self.last_max = None

    async def run(self, payload, max_tokens=None, **kwargs):  # type: ignore[no-untyped-def]
        self.last_max = max_tokens
        # Return verdict-like shape
        return {"is_valid": True, "score": 0.9, "feedback": "looks fine"}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_precheck_passes_max_tokens_to_validator():
    val = _Validator()
    step = Step(name="s1", agent=_Agent())
    step.meta["processing"] = {
        "reasoning_precheck": {
            "enabled": True,
            "validator_agent": val,
            "delimiters": ["<thinking>", "</thinking>"],
            "max_tokens": 123,
        }
    }

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    frame = make_execution_frame(
        core,
        step,
        "<thinking>plan</thinking>",
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    _ = await execu.execute(core=core, frame=frame)

    # Validator should have seen the max_tokens value
    assert val.last_max == 123

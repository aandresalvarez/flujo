from __future__ import annotations

import pytest

from flujo.tracing.manager import TraceManager
from flujo.domain.events import PreRunPayload, PreStepPayload, PostStepPayload
from flujo.domain.models import StepResult


class _Step:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.asyncio
async def test_resume_post_step_closes_lingering_span() -> None:
    mgr = TraceManager()

    # Start run and first step (e.g., HITL get_name) which then pauses
    await mgr.hook(
        PreRunPayload(event_name="pre_run", initial_input="", context=None, resources=None)
    )
    await mgr.hook(
        PreStepPayload(
            event_name="pre_step",
            step=_Step("get_name"),
            step_input="",
            context=None,
            resources=None,
        )
    )

    # Emulate resume path finalizing the paused step via post_step with a synthetic success result
    await mgr.hook(
        PostStepPayload(
            event_name="post_step",
            step_result=StepResult(name="get_name", output="alvaro", success=True),
            context=None,
            resources=None,
        )
    )

    # Next pre_step should attach as a sibling under root, not as a child of get_name
    await mgr.hook(
        PreStepPayload(
            event_name="pre_step",
            step=_Step("generate_greeting"),
            step_input="alvaro",
            context=None,
            resources=None,
        )
    )

    assert mgr._root_span is not None
    names = [ch.name for ch in mgr._root_span.children]
    # Expect two siblings under root
    assert names == ["get_name", "generate_greeting"]  # noqa: S101

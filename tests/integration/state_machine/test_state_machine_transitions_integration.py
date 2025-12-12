from __future__ import annotations

import pytest

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from tests.conftest import create_test_flujo
from flujo.testing.utils import gather_result
from flujo.domain.models import PipelineContext


def _yaml_pause_self_transition() -> str:
    return (
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: orchestrate\n"
        "    start_state: s1\n"
        "    end_states: [done]\n"
        "    states:\n"
        "      s1:\n"
        "        steps:\n"
        "          - kind: hitl\n"
        "            name: ask\n"
        "            message: 'Proceed?'\n"
        "      done:\n"
        "        steps: []\n"
        "    transitions:\n"
        "      - from: s1\n"
        "        on: pause\n"
        "        to: s1\n"
    )


@pytest.mark.asyncio
@pytest.mark.slow  # HITL/stateful resume (uses SQLite backend, interactive steps)
@pytest.mark.serial  # StateMachine tests have race conditions under xdist
async def test_yaml_pause_transition_self_reentry() -> None:
    pipeline = load_pipeline_blueprint_from_yaml(_yaml_pause_self_transition())
    runner = create_test_flujo(pipeline)
    paused = await gather_result(runner, "start")
    # gather_result returns a PipelineResult at pause with final context
    ctx = paused.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    # Pause metadata
    assert ctx.status == "paused"
    assert isinstance(ctx.pause_message, str)
    # State machine control metadata
    assert ctx.current_state == "s1"
    assert ctx.next_state == "s1"


def _yaml_multi_state_with_when() -> str:
    return (
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: orchestrate\n"
        "    start_state: s1\n"
        "    end_states: [s2, s3]\n"
        "    states:\n"
        "      s1:\n"
        "        steps: []\n"
        "      s2:\n"
        "        steps: []\n"
        "      s3:\n"
        "        steps: []\n"
        "    transitions:\n"
        "      - from: s1\n"
        "        on: success\n"
        "        to: s2\n"
        "        when: \"context.import_artifacts.get('go_s2')\"\n"
        "      - from: s1\n"
        "        on: success\n"
        "        to: s3\n"
    )


@pytest.mark.asyncio
@pytest.mark.serial  # StateMachine tests have race conditions under xdist
async def test_yaml_multi_state_flow_with_when_true() -> None:
    pipeline = load_pipeline_blueprint_from_yaml(_yaml_multi_state_with_when())
    runner = create_test_flujo(pipeline)
    final = None
    async for item in runner.stream_async(
        "start", initial_context_data={"import_artifacts": {"go_s2": True}}
    ):
        final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.current_state == "s2"


@pytest.mark.asyncio
@pytest.mark.serial  # StateMachine tests have race conditions under xdist
async def test_yaml_multi_state_flow_with_when_false() -> None:
    pipeline = load_pipeline_blueprint_from_yaml(_yaml_multi_state_with_when())
    runner = create_test_flujo(pipeline)
    final = None
    async for item in runner.stream_async("start"):
        final = item
    assert final is not None
    ctx = final.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.current_state == "s3"

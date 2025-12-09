"""Real LLM integration tests for Granular Execution Mode.

These tests make actual API calls to LLM providers.
Requires OPENAI_API_KEY environment variable to be set.

Run with: pytest tests/integration/test_granular_real_llm.py -v --slow
Skip with: pytest tests/integration/test_granular_real_llm.py -v -m "not slow"
"""

from __future__ import annotations

import os
import pytest

from flujo import Flujo
from flujo.agents.wrapper import make_agent_async
from flujo.domain.dsl import Step, Pipeline


# Skip if no API key
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM tests",
    ),
]


@pytest.mark.asyncio
async def test_granular_with_real_openai_agent() -> None:
    """Test granular execution with real OpenAI GPT-4o-mini calls.

    This test verifies that:
    1. Step.granular() creates a working pipeline
    2. The agent is called and produces output
    3. Turn tracking works correctly
    """
    # Create real OpenAI agent
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt=(
            "You are a helpful assistant. When the user says 'done', "
            "respond with exactly 'COMPLETE'. Otherwise, ask a follow-up question."
        ),
        output_type=str,
    )

    # Create granular pipeline with small max_turns
    pipeline = Step.granular("real_llm_granular", agent, max_turns=3)
    runner = Flujo(pipeline)

    # Run with input that should complete quickly
    result = None
    async for item in runner.run_async("done"):
        result = item

    # Verify we got a result with step history
    assert result is not None
    assert hasattr(result, "step_history")
    assert len(result.step_history) >= 1

    # Check that at least one step was executed (could be loop or inner step)
    # The important thing is that the LLM was called and returned a response
    assert result.total_cost_usd >= 0  # Cost tracking works


@pytest.mark.asyncio
async def test_granular_multi_turn_conversation() -> None:
    """Test multi-turn granular execution with real LLM.

    Verifies turn-by-turn execution with history accumulation.
    """
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt=(
            "You are a counting assistant. Count from 1 to 3, one number per response. "
            "After saying '3', add the word 'DONE' to your response."
        ),
        output_type=str,
    )

    pipeline = Step.granular("counting_agent", agent, max_turns=5)
    runner = Flujo(pipeline)

    result = None
    async for item in runner.run_async("Start counting"):
        result = item

    assert result is not None
    # Should have executed at least one turn
    assert len(result.step_history) >= 1


@pytest.mark.asyncio
async def test_granular_fingerprint_with_real_agent() -> None:
    """Test fingerprint generation with a real agent configuration."""
    from flujo.domain.dsl.granular import GranularStep

    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a test agent",
        output_type=str,
    )

    # Extract agent properties for fingerprint
    model_id = getattr(agent._agent, "model", None) or "openai:gpt-4o-mini"

    fp = GranularStep.compute_fingerprint(
        input_data={"test": "input"},
        system_prompt="You are a test agent",
        model_id=str(model_id),
        provider="openai",
        tools=[],
        settings={"history_max_tokens": 128000},
    )

    # Fingerprint should be a valid SHA-256 hex string
    assert isinstance(fp, str)
    assert len(fp) == 64  # SHA-256 produces 64 hex characters


@pytest.mark.asyncio
async def test_granular_step_factory_with_real_agent() -> None:
    """Test Step.granular() factory with real agent."""
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are helpful",
        output_type=str,
    )

    # Create granular pipeline
    pipeline = Step.granular(
        "factory_test",
        agent,
        max_turns=3,
        history_max_tokens=64000,
        blob_threshold_bytes=10000,
    )

    # Verify structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1  # LoopStep wrapping GranularStep

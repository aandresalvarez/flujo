"""Tests for exhaustive accounting in step logic."""

from typing import Any
from unittest.mock import Mock

import pytest

from flujo.domain.dsl.step import Step, StepConfig
from flujo.domain.models import StepResult
from flujo.domain.plugins import PluginOutcome
from flujo.application.core.step_logic import _run_step_logic


class StubAgent:
    """A simple stub agent for testing."""

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.call_count = 0

    async def run(self, data: str) -> Any:
        self.call_count += 1
        if self.outputs:
            return self.outputs.pop(0)
        return "default"


class CostlyOutput:
    """An output object that carries cost and token information."""

    def __init__(self, output: str, token_counts: int = 5, cost_usd: float = 0.2) -> None:
        self.output = output
        self.token_counts = token_counts
        self.cost_usd = cost_usd


class DummyPlugin:
    """A dummy plugin for testing."""

    def __init__(self, outcomes: list[PluginOutcome]) -> None:
        self.outcomes = outcomes

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        if self.outcomes:
            return self.outcomes.pop(0)
        return PluginOutcome(success=True)


class MockStepExecutor:
    """A mock step executor for testing fallback scenarios."""

    def __init__(self, fallback_result: StepResult) -> None:
        self.fallback_result = fallback_result
        self.call_count = 0

    async def __call__(
        self, step: Step, data: Any, context: Any, resources: Any, breach_event: Any
    ) -> StepResult:
        self.call_count += 1
        return self.fallback_result


@pytest.mark.asyncio
async def test_failed_primary_step_preserves_metrics() -> None:
    """Test that a failed primary step preserves metrics from the last attempt."""
    # Create a step that will fail after a costly agent run
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="Validation failed")])
    agent = StubAgent([CostlyOutput("expensive output", token_counts=10, cost_usd=0.5)])

    step = Step.model_validate(
        {
            "name": "test_step",
            "agent": agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )

    # Mock step executor that doesn't get called (no fallback)
    mock_executor = Mock()

    # Execute the step
    result = await _run_step_logic(
        step=step,
        data="test input",
        context=None,
        resources=None,
        step_executor=mock_executor,
        context_model_defined=False,
    )

    # Verify the step failed
    assert result.success is False
    assert "Validation failed" in result.feedback

    # Verify metrics are preserved from the failed attempt
    assert result.cost_usd == 0.5
    assert result.token_counts == 10
    assert agent.call_count == 1


@pytest.mark.asyncio
async def test_successful_fallback_preserves_metrics() -> None:
    """Test that a successful fallback correctly aggregates metrics."""
    # Create a primary step that fails
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback="Primary failed")])
    agent_primary = StubAgent([CostlyOutput("primary output", token_counts=8, cost_usd=0.3)])

    primary_step = Step.model_validate(
        {
            "name": "primary",
            "agent": agent_primary,
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )

    # Create a fallback step that succeeds
    fallback_result = StepResult(name="fallback")
    fallback_result.success = True
    fallback_result.output = "fallback output"
    fallback_result.cost_usd = 0.2
    fallback_result.token_counts = 5

    # Mock step executor that returns the fallback result
    mock_executor = MockStepExecutor(fallback_result)

    # Set up fallback
    fallback_step = Step.model_validate({"name": "fallback", "agent": StubAgent([])})
    primary_step.fallback(fallback_step)

    # Execute the step
    result = await _run_step_logic(
        step=primary_step,
        data="test input",
        context=None,
        resources=None,
        step_executor=mock_executor,
        context_model_defined=False,
    )

    # Verify the step succeeded via fallback
    assert result.success is True
    assert result.output == "fallback output"
    assert "fallback_triggered" in result.metadata_

    # Verify metrics are correctly aggregated
    assert result.cost_usd == 0.2  # Only fallback cost
    assert result.token_counts == 13  # Primary (8) + fallback (5)
    assert agent_primary.call_count == 1
    assert mock_executor.call_count == 1


@pytest.mark.asyncio
async def test_failed_fallback_accumulates_metrics() -> None:
    """Test that a failed fallback correctly accumulates metrics from both primary and fallback."""
    # Create a primary step that fails
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback="Primary failed")])
    agent_primary = StubAgent([CostlyOutput("primary output", token_counts=6, cost_usd=0.1)])

    primary_step = Step.model_validate(
        {
            "name": "primary",
            "agent": agent_primary,
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )

    # Create a fallback step that also fails
    fallback_result = StepResult(name="fallback")
    fallback_result.success = False
    fallback_result.feedback = "Fallback failed"
    fallback_result.cost_usd = 0.2
    fallback_result.token_counts = 5

    # Mock step executor that returns the failed fallback result
    mock_executor = MockStepExecutor(fallback_result)

    # Set up fallback
    fallback_step = Step.model_validate({"name": "fallback", "agent": StubAgent([])})
    primary_step.fallback(fallback_step)

    # Execute the step
    result = await _run_step_logic(
        step=primary_step,
        data="test input",
        context=None,
        resources=None,
        step_executor=mock_executor,
        context_model_defined=False,
    )

    # Verify the step failed
    assert result.success is False
    assert "Primary failed" in result.feedback
    assert "Fallback failed" in result.feedback

    # Verify metrics are correctly accumulated
    assert result.cost_usd == 0.2  # Only fallback cost
    assert result.token_counts == 11  # Primary (6) + fallback (5)
    assert agent_primary.call_count == 1
    assert mock_executor.call_count == 1


@pytest.mark.asyncio
async def test_multiple_retries_preserve_last_attempt_metrics() -> None:
    """Test that multiple retries preserve metrics from the last failed attempt."""
    # Create a step that fails twice with different costs
    plugin = DummyPlugin(
        [
            PluginOutcome(success=False, feedback="First attempt failed"),
            PluginOutcome(success=False, feedback="Second attempt failed"),
        ]
    )

    # Agent returns different costly outputs for each attempt
    agent = StubAgent(
        [
            CostlyOutput("first attempt", token_counts=5, cost_usd=0.1),
            CostlyOutput("second attempt", token_counts=10, cost_usd=0.3),
        ]
    )

    step = Step.model_validate(
        {
            "name": "test_step",
            "agent": agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=2),
        }
    )

    # Mock step executor that doesn't get called (no fallback)
    mock_executor = Mock()

    # Execute the step
    result = await _run_step_logic(
        step=step,
        data="test input",
        context=None,
        resources=None,
        step_executor=mock_executor,
        context_model_defined=False,
    )

    # Verify the step failed after all retries
    assert result.success is False
    assert "Second attempt failed" in result.feedback
    assert result.attempts == 2

    # Verify metrics are from the last attempt only
    assert result.cost_usd == 0.3  # Last attempt cost
    assert result.token_counts == 10  # Last attempt tokens
    assert agent.call_count == 2  # Both attempts were made

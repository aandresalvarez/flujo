import pytest
import asyncio
from typing import Any

from flujo.domain.dsl import Step, StepConfig
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from tests.conftest import create_test_flujo


class CostlyOutput:
    def __init__(self, output: str, token_counts: int = 5, cost_usd: float = 0.2) -> None:
        self.output = output
        self.token_counts = token_counts
        self.cost_usd = cost_usd


class ExceptionRaisingAgent:
    """Agent that raises exceptions to test error handling"""

    def __init__(self, exception_type: type = Exception, message: str = "Agent failed"):
        self.exception_type = exception_type
        self.message = message
        self.call_count = 0

    async def run(self, data: Any) -> Any:
        self.call_count += 1
        raise self.exception_type(self.message)


class TimeoutAgent:
    """Agent that takes too long to respond"""

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s
        self.call_count = 0

    async def run(self, data: Any) -> Any:
        self.call_count += 1
        await asyncio.sleep(self.timeout_s)
        return "timeout result"


class ZeroCostAgent:
    """Agent that returns zero cost/tokens to test edge cases"""

    def __init__(self, output: str = "zero cost"):
        self.output = output
        self.call_count = 0

    async def run(self, data: Any) -> Any:
        self.call_count += 1
        result = type(
            "ZeroCostResult", (), {"output": self.output, "token_counts": 0, "cost_usd": 0.0}
        )()
        return result


class HighCostAgent:
    """Agent that returns very high cost to test overflow scenarios"""

    def __init__(self, output: str = "high cost"):
        self.output = output
        self.call_count = 0

    async def run(self, data: Any) -> Any:
        self.call_count += 1
        result = type(
            "HighCostResult",
            (),
            {"output": self.output, "token_counts": 999999, "cost_usd": 999.99},
        )()
        return result


@pytest.mark.asyncio
async def test_fallback_with_zero_cost_agents() -> None:
    """Test fallback behavior when agents return zero cost/tokens"""
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": ZeroCostAgent("primary"),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": ZeroCostAgent("fallback")})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.cost_usd == 0.0
    assert sr.token_counts == 0
    assert sr.latency_s > 0  # Should still have some latency


@pytest.mark.asyncio
async def test_fallback_with_high_cost_agents() -> None:
    """Test fallback behavior with very high cost agents"""
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": HighCostAgent("primary"),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": HighCostAgent("fallback")})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.cost_usd == 999.99  # Should be fallback cost
    assert sr.token_counts == 1999998  # Should be sum of both agents


@pytest.mark.asyncio
async def test_fallback_with_exception_raising_agents() -> None:
    """Test fallback behavior when agents raise exceptions"""
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": ExceptionRaisingAgent(ValueError, "Primary agent failed"),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate(
        {"name": "fb", "agent": ExceptionRaisingAgent(RuntimeError, "Fallback agent failed")}
    )
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is False
    assert "Primary agent failed" in sr.feedback
    assert "Fallback agent failed" in sr.feedback


@pytest.mark.asyncio
async def test_fallback_with_mixed_cost_scenarios() -> None:
    """Test fallback with mixed cost scenarios (zero primary, high fallback)"""
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": ZeroCostAgent("primary"),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": HighCostAgent("fallback")})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.cost_usd == 999.99  # Should be fallback cost
    assert sr.token_counts == 999999  # Should be sum (0 + 999999)


@pytest.mark.asyncio
async def test_fallback_with_negative_metrics() -> None:
    """Test fallback behavior with negative metrics (edge case)"""

    class NegativeCostAgent:
        async def run(self, data: Any) -> Any:
            result = type(
                "NegativeResult", (), {"output": "negative", "token_counts": -5, "cost_usd": -0.1}
            )()
            return result

    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": NegativeCostAgent(),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": NegativeCostAgent()})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    # Should handle negative metrics gracefully
    assert sr.success is True
    assert sr.cost_usd == -0.1  # Should be fallback cost
    assert sr.token_counts == -10  # Should be sum (-5 + -5)


@pytest.mark.asyncio
async def test_fallback_with_missing_metrics() -> None:
    """Test fallback behavior when agents don't provide metrics"""

    class NoMetricsAgent:
        async def run(self, data: Any) -> Any:
            return "no metrics"

    plugin = DummyPlugin([PluginOutcome(success=False, feedback="primary failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": NoMetricsAgent(),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": NoMetricsAgent()})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    # Should handle missing metrics gracefully
    assert sr.success is True
    # When agents don't provide metrics, Flujo defaults to counting string output as tokens
    assert sr.cost_usd == 0.0
    assert sr.token_counts == 2  # 1 token for primary + 1 token for fallback


@pytest.mark.asyncio
async def test_fallback_with_very_long_feedback() -> None:
    """Test fallback behavior with very long feedback messages"""
    long_feedback = "x" * 10000  # Very long feedback

    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback=long_feedback)])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_plugin = DummyPlugin([PluginOutcome(success=False, feedback=long_feedback)])
    fb = Step.model_validate(
        {"name": "fb", "agent": StubAgent([CostlyOutput("oops")]), "plugins": [(fb_plugin, 0)]}
    )
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is False
    assert len(sr.feedback) > 20000  # Should contain both feedbacks
    assert "Original error:" in sr.feedback
    assert "Fallback error:" in sr.feedback


@pytest.mark.asyncio
async def test_fallback_with_none_feedback() -> None:
    """Test fallback behavior when feedback is None"""
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback=None)])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_plugin = DummyPlugin([PluginOutcome(success=False, feedback=None)])
    fb = Step.model_validate(
        {"name": "fb", "agent": StubAgent([CostlyOutput("oops")]), "plugins": [(fb_plugin, 0)]}
    )
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is False
    # Should handle None feedback gracefully - Flujo uses default error messages when feedback is None
    assert sr.feedback is not None
    assert "Original error:" in sr.feedback
    assert "Fallback error:" in sr.feedback
    # The actual error messages are generated by Flujo when feedback is None
    assert "Agent execution failed" in sr.feedback


@pytest.mark.asyncio
async def test_fallback_with_empty_string_feedback() -> None:
    """Test fallback behavior when feedback is empty string"""
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback="")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_plugin = DummyPlugin([PluginOutcome(success=False, feedback="")])
    fb = Step.model_validate(
        {"name": "fb", "agent": StubAgent([CostlyOutput("oops")]), "plugins": [(fb_plugin, 0)]}
    )
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is False
    # Should handle empty feedback gracefully
    assert sr.feedback is not None
    assert "Original error: " in sr.feedback
    assert "Fallback error: " in sr.feedback


@pytest.mark.asyncio
async def test_fallback_with_unicode_feedback() -> None:
    """Test fallback behavior with unicode characters in feedback"""
    unicode_feedback = "🚀 Primary failed with unicode: ñáéíóú"

    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback=unicode_feedback)])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_plugin = DummyPlugin([PluginOutcome(success=False, feedback=unicode_feedback)])
    fb = Step.model_validate(
        {"name": "fb", "agent": StubAgent([CostlyOutput("oops")]), "plugins": [(fb_plugin, 0)]}
    )
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is False
    assert "🚀" in sr.feedback
    assert "ñáéíóú" in sr.feedback


@pytest.mark.asyncio
async def test_fallback_with_very_small_latency() -> None:
    """Test fallback behavior with very small latency values"""

    class FastAgent:
        async def run(self, data: Any) -> Any:
            return "fast"

    plugin = DummyPlugin([PluginOutcome(success=False, feedback="fast failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": FastAgent(),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": FastAgent()})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.latency_s >= 0  # Should be non-negative
    assert sr.latency_s < 1.0  # Should be reasonable


@pytest.mark.asyncio
async def test_fallback_with_retry_scenarios() -> None:
    """Test fallback behavior with retry scenarios"""
    plugin = DummyPlugin(
        [
            PluginOutcome(success=False, feedback="attempt 1"),
            PluginOutcome(success=False, feedback="attempt 2"),
            PluginOutcome(success=False, feedback="attempt 3"),
        ]
    )
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad", "bad", "bad"]),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=3),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": StubAgent([CostlyOutput("success")])})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.attempts == 3  # Should have attempted 3 times before fallback
    assert sr.cost_usd == 0.2  # Should be fallback cost
    assert sr.token_counts >= 3  # Should include all attempts + fallback


@pytest.mark.asyncio
async def test_fallback_with_complex_metadata() -> None:
    """Test fallback behavior with complex metadata scenarios"""
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="complex failed")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": StubAgent([CostlyOutput("success")])})
    primary.fallback(fb)
    runner = create_test_flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]

    assert sr.success is True
    assert sr.metadata_ is not None
    assert sr.metadata_.get("fallback_triggered") is True
    assert "original_error" in sr.metadata_
    assert sr.metadata_["original_error"] == "complex failed"


if __name__ == "__main__":
    pytest.main([__file__])

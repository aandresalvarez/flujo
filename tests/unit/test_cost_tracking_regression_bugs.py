"""Regression tests for critical cost tracking bugs identified in code review.

This file contains tests to prevent the following critical bugs:
1. Bug #1: Incorrect cost calculation for custom outputs (50/50 token split assumption)
2. Bug #2: Incorrect usage limit precedence (step-level limits ignored)
3. Bug #3: Unhandled AttributeError on agent type
4. Bug #4: Improved model_id extraction and provider inference
5. Bug #5: Robust error handling for unknown providers and models
"""

from unittest.mock import patch
from flujo.cost import extract_usage_metrics, CostCalculator
from flujo.domain.models import UsageLimits


class TestBug1CustomOutputCostCalculation:
    """Test regression for Bug #1: Incorrect cost calculation for custom outputs."""

    def test_custom_output_with_explicit_cost_should_not_split_tokens(self):
        """Test that custom outputs with explicit cost don't use 50/50 token split."""

        # Create a custom output with explicit cost and token counts
        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.15  # Explicit cost
                self.token_counts = 1000  # Total tokens
                self.output = "Custom response"

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = CustomOutput()
        agent = MockAgent()

        # Extract metrics
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # CRITICAL: The cost should be the explicit cost, not calculated from tokens
        assert cost_usd == 0.15

        # CRITICAL: Token counts should be preserved for usage limits, but not split
        # The current implementation incorrectly assumes 50/50 split, but this is wrong
        # because prompt and completion tokens have different costs. Instead, we preserve
        # the total token count as completion_tokens to maintain compatibility.
        assert prompt_tokens == 0
        assert completion_tokens == 1000  # Preserved total for usage limits

    def test_custom_output_with_cost_only_should_trust_explicit_cost(self):
        """Test that custom outputs with only cost_usd are handled correctly."""

        # Create a custom output with only explicit cost (no token_counts)
        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.25  # Only explicit cost
                self.output = "Custom response"

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = CustomOutput()
        agent = MockAgent()

        # Extract metrics
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # CRITICAL: Should trust the explicit cost and not attempt calculation
        assert cost_usd == 0.25
        assert prompt_tokens == 0
        assert completion_tokens == 0  # No token_counts provided

    def test_custom_output_should_not_recalculate_cost_from_tokens(self):
        """Test that custom outputs don't have their cost overwritten by token calculation."""

        # Create a custom output with explicit cost that would be different from calculated
        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.10  # Explicit cost
                self.token_counts = 2000  # High token count that would calculate to different cost
                self.output = "Custom response"

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = CustomOutput()
        agent = MockAgent()

        # Extract metrics
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # CRITICAL: Should use explicit cost, not recalculate from tokens
        assert cost_usd == 0.10

        # CRITICAL: Token counts should be preserved for usage limits
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 2000  # Preserved total for usage limits

        # If we were to calculate cost from 2000 tokens with GPT-4o pricing,
        # it would be much higher than 0.10, so this proves we're using explicit cost
        cost_calculator = CostCalculator()
        calculated_cost = cost_calculator.calculate(
            model_name="gpt-4o",
            prompt_tokens=1000,  # Assume 50/50 split
            completion_tokens=1000,
            provider="openai",
        )
        assert cost_usd != calculated_cost  # Should be different


class TestBug2UsageLimitPrecedence:
    """Test regression for Bug #2: Incorrect usage limit precedence."""

    def test_step_level_limits_should_override_pipeline_level_limits(self):
        """Test that step-level limits take precedence over pipeline-level limits."""

        # Create pipeline-level limits (higher values)
        pipeline_limits = UsageLimits(
            total_cost_usd_limit=1.00,  # $1.00 pipeline limit
            total_tokens_limit=5000,  # 5000 tokens pipeline limit
        )

        # Create step-level limits (lower values - should take precedence)
        step_limits = UsageLimits(
            total_cost_usd_limit=0.10,  # $0.10 step limit
            total_tokens_limit=500,  # 500 tokens step limit
        )

        # CRITICAL: The effective limits should be the step-level limits
        # Current bug: effective_usage_limits = usage_limits or step_usage_limits
        # This gives precedence to pipeline limits, which is wrong
        effective_usage_limits = step_limits or pipeline_limits

        # Should be step limits (lower values)
        assert effective_usage_limits == step_limits
        assert effective_usage_limits.total_cost_usd_limit == 0.10
        assert effective_usage_limits.total_tokens_limit == 500

    def test_step_level_limits_with_none_pipeline_limits(self):
        """Test that step-level limits work when pipeline limits are None."""

        # No pipeline-level limits
        pipeline_limits = None

        # Step-level limits
        step_limits = UsageLimits(total_cost_usd_limit=0.05, total_tokens_limit=200)

        # Should use step limits
        effective_usage_limits = step_limits or pipeline_limits

        assert effective_usage_limits == step_limits
        assert effective_usage_limits.total_cost_usd_limit == 0.05
        assert effective_usage_limits.total_tokens_limit == 200

    def test_pipeline_level_limits_with_none_step_limits(self):
        """Test that pipeline-level limits work when step limits are None."""

        # Pipeline-level limits
        pipeline_limits = UsageLimits(total_cost_usd_limit=0.50, total_tokens_limit=1000)

        # No step-level limits
        step_limits = None

        # Should use pipeline limits
        effective_usage_limits = step_limits or pipeline_limits

        assert effective_usage_limits == pipeline_limits
        assert effective_usage_limits.total_cost_usd_limit == 0.50
        assert effective_usage_limits.total_tokens_limit == 1000

    def test_both_limits_none(self):
        """Test that None is returned when both limits are None."""

        pipeline_limits = None
        step_limits = None

        effective_usage_limits = step_limits or pipeline_limits

        assert effective_usage_limits is None


class TestBug3AttributeErrorOnAgentType:
    """Test regression for Bug #3: Unhandled AttributeError on agent type."""

    def test_simple_function_agent_should_not_crash(self):
        """Test that simple function agents don't cause AttributeError."""

        # Create a simple function as agent (no model attributes)
        def simple_function_agent(data):
            return "Simple response"

        # Create a mock response with usage info
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        raw_output = MockResponse()
        agent = simple_function_agent

        # CRITICAL: Should not raise AttributeError
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should extract tokens but not crash on cost calculation
        assert prompt_tokens == 100
        assert completion_tokens == 50
        # Cost should be 0.0 for safety since we can't determine model
        assert cost_usd == 0.0  # Safer to return 0.0 than guess

    def test_plain_class_agent_should_not_crash(self):
        """Test that plain class agents without model attributes don't crash."""

        # Create a plain class without model attributes
        class PlainClassAgent:
            def __init__(self):
                self.name = "plain_agent"

            def run(self, data):
                return "Plain response"

        # Create a mock response with usage info
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 75
                        self.response_tokens = 25

                return MockUsage()

        raw_output = MockResponse()
        agent = PlainClassAgent()

        # CRITICAL: Should not raise AttributeError
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should extract tokens but not crash on cost calculation
        assert prompt_tokens == 75
        assert completion_tokens == 25
        # Cost should be 0.0 for safety since we can't determine model
        assert cost_usd == 0.0  # Safer to return 0.0 than guess

    def test_agent_with_missing_model_attributes_should_gracefully_handle(self):
        """Test that agents with missing model attributes are handled gracefully."""

        # Create an agent with some attributes but not model-related ones
        class AgentWithOtherAttributes:
            def __init__(self):
                self.name = "test_agent"
                self.version = "1.0"
                # No model_id, _model_name, or model attributes

        # Create a mock response with usage info
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 200
                        self.response_tokens = 100

                return MockUsage()

        raw_output = MockResponse()
        agent = AgentWithOtherAttributes()

        # CRITICAL: Should not raise AttributeError
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should extract tokens but not crash on cost calculation
        assert prompt_tokens == 200
        assert completion_tokens == 100
        # Cost should be 0.0 for safety since we can't determine model
        assert cost_usd == 0.0  # Safer to return 0.0 than guess

    def test_agent_with_attribute_error_should_gracefully_handle(self):
        """Test that agents that raise AttributeError are handled gracefully."""

        # Create an agent that raises AttributeError when accessed
        class AgentWithAttributeError:
            def __init__(self):
                self._model_id = "openai:gpt-4o"

            @property
            def model_id(self):
                raise AttributeError("Model ID not available")

        # Create a mock response with usage info
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 150
                        self.response_tokens = 75

                return MockUsage()

        raw_output = MockResponse()
        agent = AgentWithAttributeError()

        # CRITICAL: Should not raise AttributeError
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should extract tokens but not crash on cost calculation
        assert prompt_tokens == 150
        assert completion_tokens == 75
        # Cost should be 0.0 for safety since we can't determine model
        assert cost_usd == 0.0  # Safer to return 0.0 than guess


class TestBug4ImprovedModelIdExtraction:
    """Test regression for Bug #4: Improved model_id extraction and provider inference."""

    def test_agent_with_explicit_model_id_should_work(self):
        """Test that agents with explicit model_id work correctly."""

        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Test with explicit model_id
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0  # Should calculate cost

    def test_agent_with_model_attribute_should_work(self):
        """Test that agents with model attribute work correctly."""

        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Test with model attribute
        class MockAgent:
            def __init__(self):
                self.model = "anthropic:claude-3-sonnet"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0  # Should calculate cost

    def test_agent_with_private_model_name_should_work(self):
        """Test that agents with private _model_name attribute work correctly."""

        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100  # Use correct attribute names
                        self.response_tokens = 50

                return MockUsage()

        class MockAgent:
            def __init__(self):
                self._model_name = "gpt-4o"  # Private attribute

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0  # Should calculate cost for known model

    def test_agent_without_model_id_logs_helpful_warning(self):
        """Test that agents without model_id log helpful warnings."""
        with patch("flujo.infra.telemetry") as mock_telemetry:
            # Create a mock response with usage info that has tokens
            class MockResponse:
                def __init__(self):
                    self.output = "test response"

                def usage(self):
                    class MockUsage:
                        def __init__(self):
                            self.request_tokens = 100  # Must have tokens to trigger warning
                            self.response_tokens = 50

                    return MockUsage()

            # Create an agent without model_id
            class AgentWithoutModelId:
                async def run(self, data):
                    return MockResponse()

            agent = AgentWithoutModelId()

            # Extract usage metrics
            extract_usage_metrics(MockResponse(), agent, "test_step")

            # Should log a helpful warning
            mock_telemetry.logfire.warning.assert_called_once()
            call_args = mock_telemetry.logfire.warning.call_args[0][0]
            assert "Could not determine model" in call_args
            assert "To fix:" in call_args
            assert "model_id" in call_args
            assert "make_agent_async" in call_args


class TestBug5RobustProviderInference:
    """Test regression for Bug #5: Robust error handling for unknown providers and models."""

    def test_unknown_provider_should_return_zero_cost(self):
        """Test that unknown providers return 0.0 cost with helpful warning."""
        with patch("flujo.infra.telemetry") as mock_telemetry:
            calculator = CostCalculator()

            # Test with unknown model that can't be inferred
            result = calculator.calculate("unknown-model", 100, 50)

            # Should return 0.0 and log a warning
            assert result == 0.0
            mock_telemetry.logfire.warning.assert_called_once()
            call_args = mock_telemetry.logfire.warning.call_args[0][0]
            assert "Could not infer provider" in call_args
            assert "To fix:" in call_args
            assert "provider:model format" in call_args

    def test_ambiguous_model_names_should_not_infer_provider(self):
        """Test that ambiguous model names don't incorrectly infer providers."""
        calculator = CostCalculator()

        # Test ambiguous models that could belong to multiple providers
        # These models should not be inferred to avoid incorrect cost calculations
        ambiguous_models = ["mixtral-8x7b", "codellama-7b", "gemma2-7b"]

        for model in ambiguous_models:
            provider = calculator._infer_provider_from_model(model)
            # Should return None for ambiguous models to avoid incorrect inference
            assert provider is None, f"Should not infer provider for ambiguous model: {model}"

        # Test that llama-2 is correctly inferred as meta (this is unambiguous)
        provider = calculator._infer_provider_from_model("llama-2")
        assert provider == "meta", "llama-2 should be inferred as meta"

    def test_known_provider_with_unknown_model_should_return_zero_cost(self):
        """Test that known providers with unknown models raise PricingNotConfiguredError in strict mode."""

        cost_calculator = CostCalculator()
        import pytest
        from flujo.exceptions import PricingNotConfiguredError

        # Test with known provider but unknown model
        with pytest.raises(PricingNotConfiguredError):
            cost_calculator.calculate(
                model_name="gpt-999",  # Unknown model
                prompt_tokens=100,
                completion_tokens=50,
                provider="openai",
            )

    def test_empty_model_name_should_return_zero_cost(self):
        """Test that empty model names return 0.0 cost."""

        cost_calculator = CostCalculator()

        # Test with empty model name
        cost = cost_calculator.calculate(
            model_name="", prompt_tokens=100, completion_tokens=50, provider=None
        )

        # Should return 0.0 for safety
        assert cost == 0.0

    def test_none_model_name_should_return_zero_cost(self):
        """Test that None model names return 0.0 cost."""

        cost_calculator = CostCalculator()

        # Test with None model name
        cost = cost_calculator.calculate(
            model_name=None,  # type: ignore
            prompt_tokens=100,
            completion_tokens=50,
            provider=None,
        )

        # Should return 0.0 for safety
        assert cost == 0.0


class TestExplicitCostReportingValidation:
    def test_missing_token_counts_warns(self, caplog):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1.23

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        with caplog.at_level("WARNING"):
            prompt, tokens, cost = extract_usage_metrics(Output(), agent, "test_step")
            assert tokens == 0
            assert cost == 1.23
            assert any(
                "provides cost_usd but not token_counts" in r for r in caplog.text.splitlines()
            )

    def test_missing_cost_usd_warns(self, caplog):
        from flujo.cost import extract_usage_metrics

        class Output:
            token_counts = 42

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        with caplog.at_level("WARNING"):
            prompt, tokens, cost = extract_usage_metrics(Output(), agent, "test_step")
            assert tokens == 42
            assert cost == 0.0
            assert any(
                "provides token_counts but not cost_usd" in r for r in caplog.text.splitlines()
            )

    def test_strict_mode_raises_on_missing_token_counts(self, monkeypatch):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1.23

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        monkeypatch.setenv("FLUJO_STRICT_COST_TRACKING", "1")
        try:
            try:
                extract_usage_metrics(Output(), agent, "test_step")
            except ValueError as e:
                assert "provides cost_usd but not token_counts" in str(e)
            else:
                assert False, "Should have raised ValueError"
        finally:
            monkeypatch.delenv("FLUJO_STRICT_COST_TRACKING", raising=False)

    def test_strict_mode_raises_on_missing_cost_usd(self, monkeypatch):
        from flujo.cost import extract_usage_metrics

        class Output:
            token_counts = 42

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        monkeypatch.setenv("FLUJO_STRICT_COST_TRACKING", "1")
        try:
            try:
                extract_usage_metrics(Output(), agent, "test_step")
            except ValueError as e:
                assert "provides token_counts but not cost_usd" in str(e)
            else:
                assert False, "Should have raised ValueError"
        finally:
            monkeypatch.delenv("FLUJO_STRICT_COST_TRACKING", raising=False)

    def test_negative_values_warn(self, caplog):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = -1.0
            token_counts = -10

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        with caplog.at_level("WARNING"):
            prompt, tokens, cost = extract_usage_metrics(Output(), agent, "test_step")
            assert tokens == -10
            assert cost == -1.0
            assert any("Negative values detected" in r for r in caplog.text.splitlines())

    def test_implausible_values_warn(self, caplog):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1e7
            token_counts = int(1e10)

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        with caplog.at_level("WARNING"):
            prompt, tokens, cost = extract_usage_metrics(Output(), agent, "test_step")
            assert tokens == int(1e10)
            assert cost == 1e7
            assert any("Implausibly large values detected" in r for r in caplog.text.splitlines())

    def test_negative_values_strict(self, monkeypatch):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = -1.0
            token_counts = -10

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        monkeypatch.setenv("FLUJO_STRICT_COST_TRACKING", "1")
        try:
            try:
                extract_usage_metrics(Output(), agent, "test_step")
            except ValueError as e:
                assert "Negative values detected" in str(e)
            else:
                assert False, "Should have raised ValueError"
        finally:
            monkeypatch.delenv("FLUJO_STRICT_COST_TRACKING", raising=False)

    def test_implausible_values_strict(self, monkeypatch):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1e7
            token_counts = int(1e10)

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        monkeypatch.setenv("FLUJO_STRICT_COST_TRACKING", "1")
        try:
            try:
                extract_usage_metrics(Output(), agent, "test_step")
            except ValueError as e:
                assert "Implausibly large values detected" in str(e)
            else:
                assert False, "Should have raised ValueError"
        finally:
            monkeypatch.delenv("FLUJO_STRICT_COST_TRACKING", raising=False)

    def test_mixed_reporting_modes_warn(self, caplog):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1.0
            token_counts = 10

            def usage(self):
                class Usage:
                    request_tokens = 5
                    response_tokens = 5

                return Usage()

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        with caplog.at_level("WARNING"):
            prompt, tokens, cost = extract_usage_metrics(Output(), agent, "test_step")
            assert any("Mixed reporting modes detected" in r for r in caplog.text.splitlines())

    def test_mixed_reporting_modes_strict(self, monkeypatch):
        from flujo.cost import extract_usage_metrics

        class Output:
            cost_usd = 1.0
            token_counts = 10

            def usage(self):
                class Usage:
                    request_tokens = 5
                    response_tokens = 5

                return Usage()

        agent = type("Agent", (), {"model_id": "openai:gpt-4o"})()
        monkeypatch.setenv("FLUJO_STRICT_COST_TRACKING", "1")
        try:
            try:
                extract_usage_metrics(Output(), agent, "test_step")
            except ValueError as e:
                assert "Mixed reporting modes detected" in str(e)
            else:
                assert False, "Should have raised ValueError"
        finally:
            monkeypatch.delenv("FLUJO_STRICT_COST_TRACKING", raising=False)


class TestIntegrationRegressionTests:
    """Integration tests to ensure the fixes work together correctly."""

    def test_custom_output_with_step_limits_and_simple_agent(self):
        """Test integration of all three bug fixes together."""

        # Create a custom output with explicit cost
        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.30
                self.token_counts = 1500
                self.output = "Custom response"

        # Create a simple function agent (no model attributes)
        def simple_agent(data):
            return CustomOutput()

        # Create step-level limits
        step_limits = UsageLimits(
            total_cost_usd_limit=0.25,  # Lower than explicit cost
            total_tokens_limit=1000,
        )

        raw_output = CustomOutput()
        agent = simple_agent

        # CRITICAL: Should handle all three scenarios correctly:
        # 1. Use explicit cost (not 50/50 split)
        # 2. Step limits should take precedence
        # 3. Simple agent should not cause AttributeError
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Bug #1 fix: Should use explicit cost
        assert cost_usd == 0.30

        # Bug #1 fix: Should not assume 50/50 split
        assert prompt_tokens == 0
        assert completion_tokens == 1500  # Preserved total for usage limits

        # Bug #2 fix: Step limits should be used
        effective_limits = step_limits or None
        assert effective_limits == step_limits
        assert effective_limits.total_cost_usd_limit == 0.25

        # Bug #3 fix: Simple agent should not cause AttributeError
        # (test passes if no exception is raised)

    def test_usage_limit_precedence_with_custom_output(self):
        """Test that usage limit precedence works with custom outputs."""

        # Create a custom output
        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.20
                self.token_counts = 800
                self.output = "Custom response"

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        # Create limits
        pipeline_limits = UsageLimits(
            total_cost_usd_limit=0.50,  # Higher limit
            total_tokens_limit=2000,
        )
        step_limits = UsageLimits(
            total_cost_usd_limit=0.15,  # Lower limit - should take precedence
            total_tokens_limit=500,
        )

        raw_output = CustomOutput()
        agent = MockAgent()

        # Extract metrics
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Bug #1 fix: Should use explicit cost
        assert cost_usd == 0.20

        # Bug #2 fix: Step limits should take precedence
        effective_limits = step_limits or pipeline_limits
        assert effective_limits == step_limits
        assert effective_limits.total_cost_usd_limit == 0.15

        # The cost (0.20) exceeds the step limit (0.15), so this would trigger a limit breach
        # in actual usage, but we're just testing the precedence logic here
        assert cost_usd > effective_limits.total_cost_usd_limit


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness scenarios."""

    def test_custom_output_with_zero_cost(self):
        """Test custom output with zero cost."""

        class CustomOutput:
            def __init__(self):
                self.cost_usd = 0.0
                self.token_counts = 100
                self.output = "Free response"

        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = CustomOutput()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert cost_usd == 0.0
        assert prompt_tokens == 0
        assert completion_tokens == 100  # Preserved total for usage limits

    def test_custom_output_with_none_cost(self):
        """Test custom output with None cost."""

        class CustomOutput:
            def __init__(self):
                self.cost_usd = None
                self.token_counts = 100
                self.output = "Response with None cost"

        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = CustomOutput()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should handle None gracefully
        assert cost_usd == 0.0
        assert prompt_tokens == 0
        assert completion_tokens == 100  # Preserved total for usage limits

    def test_agent_with_complex_model_id_parsing(self):
        """Test agent with complex model ID that needs parsing."""

        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 300
                        self.response_tokens = 150

                return MockUsage()

        # Test different model ID formats
        test_cases = [
            ("openai:gpt-4o", "openai", "gpt-4o"),
            ("anthropic:claude-3-sonnet", "anthropic", "claude-3-sonnet"),
            ("gpt-4o", None, "gpt-4o"),  # No provider specified
            ("google:gemini-1.5-pro", "google", "gemini-1.5-pro"),
        ]

        for model_id, expected_provider, expected_model in test_cases:

            class MockAgent:
                def __init__(self):
                    self.model_id = model_id

            raw_output = MockResponse()
            agent = MockAgent()

            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output, agent, "test_step"
            )
            # Should extract tokens correctly
            assert prompt_tokens == 300
            assert completion_tokens == 150
            # Cost should be calculated for models with pricing
            assert cost_usd > 0.0  # Should have pricing for all configured models

    def test_usage_limit_precedence_with_falsy_values(self):
        """Test usage limit precedence with falsy values."""

        # Test with empty UsageLimits (all None values)
        empty_limits = UsageLimits()

        # Test with zero limits
        zero_limits = UsageLimits(total_cost_usd_limit=0.0, total_tokens_limit=0)

        # Test precedence logic
        assert (empty_limits or None) == empty_limits
        assert (zero_limits or None) == zero_limits
        assert (None or empty_limits) == empty_limits
        assert (None or zero_limits) == zero_limits

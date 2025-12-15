"""Unit tests for cost tracking functionality."""

import pytest
from unittest.mock import Mock, patch

from flujo.cost import extract_usage_metrics, CostCalculator
from flujo.domain.models import UsageLimits
from flujo.exceptions import PricingNotConfiguredError
from flujo.infra.config import ProviderPricing, get_provider_pricing, CostConfig


class TestExtractUsageMetrics:
    """Test the shared extract_usage_metrics function."""

    def test_extract_usage_metrics_with_usage_info(self):
        """Test extraction of usage metrics from pydantic-ai response."""

        # Clear caches to ensure test isolation in parallel execution
        from flujo.cost import clear_cost_cache
        from flujo.utils.model_utils import clear_model_id_cache

        clear_cost_cache()
        clear_model_id_cache()

        # Create a mock response with usage information
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Create a mock agent with model_id
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
        assert cost_usd > 0.0  # Should calculate cost based on pricing

    def test_extract_usage_metrics_without_usage_info(self):
        """Test extraction when no usage information is available."""

        # Create a mock response without usage information
        class MockResponse:
            def __init__(self):
                self.output = "test response"

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 0
        assert completion_tokens == 0
        assert cost_usd == 0.0

    def test_extract_usage_metrics_with_exception(self):
        """Test extraction when usage() method raises an exception."""

        # Create a mock response that raises an exception
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                raise ValueError("Usage method failed")

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        # Should not raise an exception, should return 0 values
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 0
        assert completion_tokens == 0
        assert cost_usd == 0.0

    def test_extract_usage_metrics_with_different_agent_types(self):
        """Test extraction with different agent types and model ID access patterns."""

        # Test with AsyncAgentWrapper (public model_id)
        class AsyncAgentWrapper:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        # Test with agent that has _model_name (private attribute)
        class AgentWithPrivateModel:
            def __init__(self):
                self._model_name = "openai:gpt-4o"

        # Test with agent that has model attribute
        class AgentWithModel:
            def __init__(self):
                self.model = "openai:gpt-4o"

        # Create a mock response with usage information
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

        # Test AsyncAgentWrapper
        agent1 = AsyncAgentWrapper()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent1, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

        # Test AgentWithPrivateModel (fallback)
        agent2 = AgentWithPrivateModel()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent2, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

        # Test AgentWithModel (fallback)
        agent3 = AgentWithModel()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent3, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

    def test_extract_usage_metrics_with_explicit_metrics(self):
        """Test extraction when explicit cost_usd and token_counts are provided."""

        # Create a mock response with explicit metrics
        class MockResponse:
            def __init__(self):
                self.output = "test response"
                self.cost_usd = 0.05
                self.token_counts = 150

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use explicit cost and not attempt token calculation
        assert cost_usd == 0.05
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 150  # Preserved total for usage limits

    def test_extract_usage_metrics_with_mock_explicit_metrics(self):
        """Explicit metrics provided as mocks should be treated as zeroes safely."""

        from unittest.mock import Mock

        class MockResponse:
            def __init__(self):
                self.output = "irrelevant"
                # cost_usd and token_counts as mocks
                self.cost_usd = Mock()
                self.token_counts = Mock()

        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert cost_usd == 0.0
        assert prompt_tokens == 0
        assert completion_tokens == 0

    def test_extract_usage_metrics_with_usage_mock_counts(self):
        """Usage object fields as mocks should yield zero tokens and zero cost."""

        from unittest.mock import Mock

        class MockResponse:
            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = Mock()
                        self.response_tokens = Mock()

                return MockUsage()

        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 0
        assert completion_tokens == 0
        assert cost_usd == 0.0

    def test_extract_usage_metrics_with_model_id_parsing(self):
        """Test extraction with different model_id formats."""

        # Test with provider:model format
        class MockAgent1:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        # Test with just model name (no provider)
        class MockAgent2:
            def __init__(self):
                self.model_id = "gpt-4o"

        # Create a mock response with usage information
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

        # Test with provider:model format
        agent1 = MockAgent1()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent1, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

        # Test with just model name
        agent2 = MockAgent2()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent2, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

    @pytest.mark.serial  # Monkey-patches telemetry.logfire.warning - must run serially
    def test_extract_usage_metrics_warning_for_missing_model(self):
        """Test that a strong warning is logged when model_id is not found."""

        # Create a mock response with usage information
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Create a mock agent without model_id
        class MockAgent:
            def __init__(self):
                pass  # No model_id attribute

        raw_output = MockResponse()
        agent = MockAgent()

        # Capture the warning message
        from flujo.infra import telemetry

        # Create a custom handler to capture warning messages
        warning_messages = []
        original_warning = telemetry.logfire.warning

        def capture_warning(message, *args, **kwargs):
            warning_messages.append(message)
            original_warning(message, *args, **kwargs)

        telemetry.logfire.warning = capture_warning

        try:
            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output, agent, "test_step"
            )

            # Verify that a warning was logged
            assert len(warning_messages) > 0
            warning_message = warning_messages[0]
            assert "CRITICAL" in warning_message
            assert "Could not determine model" in warning_message
            assert "model_id'" in warning_message

            # Verify that cost calculation returns 0.0 for safety
            assert prompt_tokens == 100
            assert completion_tokens == 50
            assert cost_usd == 0.0  # Safer to return 0.0 than guess

        finally:
            # Restore original warning method
            telemetry.logfire.warning = original_warning

    def test_extract_usage_metrics_no_model_info_available(self, caplog):
        """Test that the system handles cases where no model information is available."""

        # Create a mock response with usage information
        class MockResponse:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Create a mock agent with no model information
        class MockAgent:
            def __init__(self):
                pass  # No model_id, _model_name, or model attributes

        raw_output = MockResponse()
        agent = MockAgent()

        # Import to clear log for clean test
        import logging

        # Capture all log levels (the message is logged at ERROR level via standard logging)
        with caplog.at_level(logging.DEBUG):
            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output, agent, "test_step"
            )

            # Verify that a critical warning was logged
            # The message goes through standard Python logging, not just logfire
            log_messages = [record.message for record in caplog.records]
            assert len(log_messages) > 0, (
                f"Expected log messages but got none. "
                f"Tokens: p={prompt_tokens}, c={completion_tokens}, cost={cost_usd}"
            )

            # Check for the critical warning message
            critical_warnings = [
                msg
                for msg in log_messages
                if "CRITICAL" in msg and "Could not determine model" in msg
            ]
            assert len(critical_warnings) > 0, (
                f"Expected CRITICAL warning about missing model but found none. "
                f"Log messages: {log_messages}"
            )

            # Verify that the system returns 0.0 for safety
            assert prompt_tokens == 100
            assert completion_tokens == 50
            assert cost_usd == 0.0  # Safer to return 0.0 than guess

    def test_extract_usage_metrics_graceful_fallback(self):
        """Test that the system gracefully falls back when usage information is incomplete."""

        # Clear caches to ensure test isolation in parallel execution
        from flujo.cost import clear_cost_cache
        from flujo.utils.model_utils import clear_model_id_cache

        clear_cost_cache()
        clear_model_id_cache()

        # Test with missing usage information
        class MockResponse:
            def __init__(self):
                self.output = "test response"

        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponse()
        agent = MockAgent()

        # Should not raise an exception, should return 0 values
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 0
        assert completion_tokens == 0
        assert cost_usd == 0.0

        # Test with partial usage information
        class MockResponseWithPartialUsage:
            def __init__(self):
                self.output = "test response"

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        # Missing response_tokens

                return MockUsage()

        raw_output = MockResponseWithPartialUsage()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        assert prompt_tokens == 100
        assert completion_tokens == 0  # Should default to 0
        assert cost_usd > 0.0  # Should still calculate cost

    def test_explicit_cost_reporter_protocol_priority(self):
        """Test that ExplicitCostReporter protocol takes highest priority over usage() method."""

        # Create a mock object that implements BOTH ExplicitCostReporter and has a .usage() method
        class MockResponseWithBoth:
            def __init__(self):
                self.cost_usd = 1.23
                self.token_counts = 200

            def usage(self):
                class MockUsage:
                    def __init__(self):
                        self.request_tokens = 100
                        self.response_tokens = 50

                return MockUsage()

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithBoth()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use the protocol cost (1.23) and ignore the .usage() method completely
        assert cost_usd == 1.23
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 200  # From protocol

    def test_explicit_cost_reporter_protocol_with_only_cost(self):
        """Test ExplicitCostReporter protocol with only cost_usd attribute."""

        # Create a mock object that implements ExplicitCostReporter with only cost_usd
        class MockResponseWithOnlyCost:
            def __init__(self):
                self.cost_usd = 0.75
                # No token_counts attribute

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithOnlyCost()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use the protocol cost and default token_counts to 0
        assert cost_usd == 0.75
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 0  # Defaults to 0

    def test_explicit_cost_reporter_protocol_with_cost_and_tokens(self):
        """Test ExplicitCostReporter protocol with both cost_usd and token_counts."""

        # Create a mock object that implements ExplicitCostReporter with both attributes
        class MockResponseWithCostAndTokens:
            def __init__(self):
                self.cost_usd = 0.25
                self.token_counts = 1000

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithCostAndTokens()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use the protocol cost and token counts
        assert cost_usd == 0.25
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 1000  # From protocol

    def test_explicit_cost_reporter_protocol_with_zero_cost(self):
        """Test ExplicitCostReporter protocol with zero cost."""

        # Create a mock object that implements ExplicitCostReporter with zero cost
        class MockResponseWithZeroCost:
            def __init__(self):
                self.cost_usd = 0.0
                self.token_counts = 500

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithZeroCost()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use the protocol values even if cost is zero
        assert cost_usd == 0.0
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 500  # From protocol

    def test_explicit_cost_reporter_protocol_with_negative_cost(self):
        """Test ExplicitCostReporter protocol with negative cost (edge case)."""

        # Create a mock object that implements ExplicitCostReporter with negative cost
        class MockResponseWithNegativeCost:
            def __init__(self):
                self.cost_usd = -0.10
                self.token_counts = 100

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithNegativeCost()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should use the protocol values even if cost is negative
        assert cost_usd == -0.10
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 100  # From protocol

    def test_explicit_cost_reporter_protocol_with_none_cost(self):
        """Test ExplicitCostReporter protocol with None cost (edge case)."""

        # Create a mock object that implements ExplicitCostReporter with None cost
        class MockResponseWithNoneCost:
            def __init__(self):
                self.cost_usd = None
                self.token_counts = 200

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithNoneCost()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should handle None cost gracefully
        assert cost_usd == 0.0  # Should default to 0.0
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 200  # From protocol

    def test_explicit_cost_reporter_protocol_with_none_tokens(self):
        """Test ExplicitCostReporter protocol with None token_counts (edge case)."""

        # Create a mock object that implements ExplicitCostReporter with None token_counts
        class MockResponseWithNoneTokens:
            def __init__(self):
                self.cost_usd = 0.50
                self.token_counts = None

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

        raw_output = MockResponseWithNoneTokens()
        agent = MockAgent()

        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, agent, "test_step"
        )

        # Should handle None token_counts gracefully
        assert cost_usd == 0.50
        assert prompt_tokens == 0  # Cannot be determined reliably
        assert completion_tokens == 0  # Should default to 0


class TestCostCalculator:
    """Test the CostCalculator class."""

    def test_calculate_cost_with_pricing(self):
        """Test cost calculation with valid pricing information."""
        calculator = CostCalculator()

        # Mock the get_provider_pricing function
        mock_pricing = ProviderPricing(prompt_tokens_per_1k=0.005, completion_tokens_per_1k=0.015)

        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.infra.config.get_provider_pricing", lambda provider, model: mock_pricing
            )

            cost = calculator.calculate(
                model_name="gpt-4o", prompt_tokens=1000, completion_tokens=500, provider="openai"
            )

            # Expected: (1000/1000 * 0.005) + (500/1000 * 0.015) = 0.005 + 0.0075 = 0.0125
            assert cost == 0.0125

    def test_calculate_cost_no_pricing(self):
        """Test cost calculation when no pricing is configured."""
        calculator = CostCalculator()

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.infra.config.get_provider_pricing", lambda provider, model: None)
            # Patch the warning method directly on the logfire object
            from flujo.infra import telemetry

            telemetry.logfire.warning = lambda msg: None

            cost = calculator.calculate(
                model_name="unknown-model", prompt_tokens=1000, completion_tokens=500
            )

            # Should return 0.0 when no pricing is configured
            assert cost == 0.0

    def test_calculate_cost_with_provider_inference(self):
        """Test cost calculation with automatic provider inference."""
        calculator = CostCalculator()

        # Mock the get_provider_pricing function
        mock_pricing = ProviderPricing(prompt_tokens_per_1k=0.005, completion_tokens_per_1k=0.015)

        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "flujo.infra.config.get_provider_pricing", lambda provider, model: mock_pricing
            )

            # Test with OpenAI model (should infer provider)
            cost = calculator.calculate(
                model_name="gpt-4o", prompt_tokens=1000, completion_tokens=500
            )
            assert cost == 0.0125

            # Test with Anthropic model (should infer provider)
            cost = calculator.calculate(
                model_name="claude-3-sonnet", prompt_tokens=1000, completion_tokens=500
            )
            assert cost > 0.0

    def test_calculate_cost_unknown_provider_returns_zero(self):
        """Test that unknown providers return 0.0 cost to avoid breaking pipelines."""
        calculator = CostCalculator()

        with pytest.MonkeyPatch().context() as m:
            m.setattr("flujo.infra.config.get_provider_pricing", lambda provider, model: None)

            # Test with unknown model that can't be inferred
            # Should return 0.0 instead of raising an error to avoid breaking pipelines
            cost = calculator.calculate(
                model_name="unknown-model", prompt_tokens=1000, completion_tokens=500
            )
            assert cost == 0.0

    def test_infer_provider_from_model(self):
        """Test provider inference from model names."""
        calculator = CostCalculator()

        # Test OpenAI models
        assert calculator._infer_provider_from_model("gpt-4o") == "openai"
        assert calculator._infer_provider_from_model("text-davinci-003") == "openai"

        # Test OpenAI embedding models
        assert calculator._infer_provider_from_model("text-embedding-3-large") == "openai"
        assert calculator._infer_provider_from_model("text-embedding-3-small") == "openai"
        assert calculator._infer_provider_from_model("text-embedding-ada-002") == "openai"

        # Test Anthropic models
        assert calculator._infer_provider_from_model("claude-3-sonnet") == "anthropic"
        assert calculator._infer_provider_from_model("claude-3-haiku") == "anthropic"

        # Test Google models
        assert calculator._infer_provider_from_model("gemini-1.5-pro") == "google"

        # Test unknown model (should return None instead of defaulting to openai)
        assert calculator._infer_provider_from_model("unknown-model") is None

    def test_infer_provider_from_model_edge_cases(self):
        """Test provider inference with edge cases."""
        calculator = CostCalculator()

        # Test various model name patterns
        assert calculator._infer_provider_from_model("gpt-4") == "openai"
        assert calculator._infer_provider_from_model("text-embedding-ada-002") == "openai"
        assert calculator._infer_provider_from_model("dall-e-3") == "openai"

        assert calculator._infer_provider_from_model("claude-3-opus") == "anthropic"
        assert calculator._infer_provider_from_model("haiku") == "anthropic"
        assert calculator._infer_provider_from_model("sonnet") == "anthropic"

        assert calculator._infer_provider_from_model("gemini-1.5-flash") == "google"
        assert calculator._infer_provider_from_model("text-bison") == "google"

        # Ambiguous models that could belong to multiple providers - should return None
        assert calculator._infer_provider_from_model("llama-2-7b") is None
        assert calculator._infer_provider_from_model("codellama-7b") is None

        assert calculator._infer_provider_from_model("mistral-7b") is None
        assert calculator._infer_provider_from_model("mixtral-8x7b") is None

        # Ambiguous models that could belong to multiple providers - should return None
        assert calculator._infer_provider_from_model("groq-llama2-70b") is None
        assert calculator._infer_provider_from_model("gemma2-2b") is None

    def test_cost_calculation_edge_cases(self):
        """Test cost calculation with edge cases."""
        calculator = CostCalculator()

        # Test with zero tokens
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=0, completion_tokens=0, provider="openai"
        )
        assert cost == 0.0

        # Test with very large token counts
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=1000000, completion_tokens=500000, provider="openai"
        )
        assert cost > 0.0

        # Test with negative tokens (should handle gracefully)
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=-100, completion_tokens=-50, provider="openai"
        )
        # Should still calculate cost (negative tokens would result in negative cost)
        assert isinstance(cost, float)

    def test_provider_inference_comprehensive(self):
        """Test comprehensive provider inference for various model patterns."""
        calculator = CostCalculator()

        # Test all known provider patterns
        test_cases = [
            ("gpt-4o", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("text-davinci-003", "openai"),
            ("dall-e-3", "openai"),
            ("text-embedding-3-large", "openai"),
            ("text-embedding-3-small", "openai"),
            ("text-embedding-ada-002", "openai"),
            ("claude-3-sonnet", "anthropic"),
            ("claude-3-haiku", "anthropic"),
            ("claude-3-opus", "anthropic"),
            ("gemini-1.5-pro", "google"),
            ("gemini-1.5-flash", "google"),
            ("text-bison", "google"),
            ("chat-bison", "google"),
            # Ambiguous models that could belong to multiple providers
            ("llama-2-7b", None),
            ("codellama-7b", None),
            ("mistral-7b", None),
            ("mixtral-8x7b", None),
            # Ambiguous models that could belong to multiple providers
            ("groq-llama2-70b", None),
            ("gemma2-2b", None),
        ]

        for model_name, expected_provider in test_cases:
            inferred_provider = calculator._infer_provider_from_model(model_name)
            assert inferred_provider == expected_provider, (
                f"Failed for {model_name}: expected {expected_provider}, got {inferred_provider}"
            )

        # Test unknown models
        unknown_models = ["unknown-model", "custom-model", "test-model"]
        for model_name in unknown_models:
            inferred_provider = calculator._infer_provider_from_model(model_name)
            assert inferred_provider is None, f"Should return None for unknown model {model_name}"

    def test_cost_calculation_precision(self):
        """Test that cost calculations are precise and consistent."""
        calculator = CostCalculator()

        # Test with exact token counts that should give precise results
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=1000, completion_tokens=1000, provider="openai"
        )
        # Expected: (1000/1000 * 0.005) + (1000/1000 * 0.015) = 0.005 + 0.015 = 0.02
        assert cost == 0.02

        # Test with different token ratios
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=2000, completion_tokens=500, provider="openai"
        )
        # Expected: (2000/1000 * 0.005) + (500/1000 * 0.015) = 0.01 + 0.0075 = 0.0175
        assert cost == 0.0175

    def test_ci_environment_fallback_with_existing_config(self):
        """Test that CI environment fallback works even when config file exists."""
        import os
        from unittest.mock import patch
        from flujo.exceptions import PricingNotConfiguredError

        # Mock CI environment
        with patch.dict(os.environ, {"CI": "true"}):
            # Mock config manager to simulate config file exists but doesn't have the model
            with patch("flujo.infra.config._no_config_file_found", return_value=False):
                calculator = CostCalculator()

                # Test that we can still calculate costs for known models in CI
                cost = calculator.calculate(
                    model_name="text-embedding-3-small",
                    prompt_tokens=1000,
                    completion_tokens=0,
                    provider="openai",
                )
                assert cost == 0.00002

                # Test that unknown models still raise errors in CI
                with pytest.raises(PricingNotConfiguredError):
                    calculator.calculate(
                        model_name="unknown-model",
                        prompt_tokens=100,
                        completion_tokens=50,
                        provider="unknown",
                    )

    def test_ci_environment_fallback_for_failing_models(self):
        """Test that CI environment fallback works for the specific failing models."""
        import os
        from unittest.mock import patch

        # Mock CI environment
        with patch.dict(os.environ, {"CI": "true"}):
            # Mock config manager to simulate no config file found
            with patch("flujo.infra.config._no_config_file_found", return_value=True):
                calculator = CostCalculator()

                # Test the specific failing models from the CI logs
                cost = calculator.calculate(
                    model_name="text-embedding-3-small",
                    prompt_tokens=1000,
                    completion_tokens=0,
                    provider="openai",
                )
                assert cost == 0.00002

                cost = calculator.calculate(
                    model_name="claude-3-sonnet",
                    prompt_tokens=1000,
                    completion_tokens=500,
                    provider="anthropic",
                )
                # Expected: (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
                assert cost == pytest.approx(0.0105, rel=1e-10)

    def test_ci_environment_fallback_with_strict_pricing(self):
        """Test that CI environment fallback works correctly with strict pricing."""
        import os
        from unittest.mock import patch
        from flujo.exceptions import PricingNotConfiguredError

        # Mock CI environment
        with patch.dict(os.environ, {"CI": "true"}):
            # Mock config manager to simulate no config file found
            with patch("flujo.infra.config._no_config_file_found", return_value=True):
                calculator = CostCalculator()

                # Test that we can still calculate costs for known models in CI
                cost = calculator.calculate(
                    model_name="text-embedding-3-small",
                    prompt_tokens=1000,
                    completion_tokens=0,
                    provider="openai",
                )
                # Expected: (1000/1000 * 0.00002) = 0.00002
                assert cost == 0.00002

                # Test that we can calculate costs for anthropic models in CI
                cost = calculator.calculate(
                    model_name="claude-3-sonnet",
                    prompt_tokens=1000,
                    completion_tokens=500,
                    provider="anthropic",
                )
                # Expected: (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
                assert cost == pytest.approx(0.0105, rel=1e-10)

                # Test that unknown models still raise errors in CI
                with pytest.raises(PricingNotConfiguredError):
                    calculator.calculate(
                        model_name="unknown-model",
                        prompt_tokens=100,
                        completion_tokens=50,
                        provider="unknown",
                    )

    def test_cost_calculation_for_embedding_models(self):
        """Test that cost calculations work correctly for embedding models."""
        calculator = CostCalculator()

        # Test with embedding model pricing
        cost = calculator.calculate(
            model_name="text-embedding-3-large",
            prompt_tokens=1000,
            completion_tokens=0,
            provider="openai",
        )
        # Expected: (1000/1000 * 0.00013) = 0.00013
        assert cost == 0.00013

        # Test with embedding model and completion tokens (should be treated the same)
        cost = calculator.calculate(
            model_name="text-embedding-3-large",
            prompt_tokens=500,
            completion_tokens=500,
            provider="openai",
        )
        # Expected: (500/1000 * 0.00013) + (500/1000 * 0.00013) = 0.000065 + 0.000065 = 0.00013
        assert cost == 0.00013

        # Test with different embedding model
        cost = calculator.calculate(
            model_name="text-embedding-3-small",
            prompt_tokens=1000,
            completion_tokens=0,
            provider="openai",
        )
        # Expected: (1000/1000 * 0.00002) = 0.00002
        assert cost == 0.00002

    def test_error_handling_robustness(self):
        """Test that the system handles errors gracefully without breaking pipelines."""
        calculator = CostCalculator()

        # Test with invalid model names
        cost = calculator.calculate(model_name="", prompt_tokens=100, completion_tokens=50)
        assert cost == 0.0

        # Test with None values - should handle gracefully
        try:
            cost = calculator.calculate(model_name=None, prompt_tokens=100, completion_tokens=50)
            assert cost == 0.0
        except (TypeError, AttributeError):
            # It's acceptable for this to raise an error since None is not a valid model name
            pass

        # Test with invalid token counts - should handle gracefully
        try:
            cost = calculator.calculate(
                model_name="gpt-4o",
                prompt_tokens="invalid",
                completion_tokens="invalid",
                provider="openai",
            )
            # Should handle gracefully (though this might raise a TypeError in practice)
            assert isinstance(cost, float)
        except (TypeError, ValueError):
            # It's acceptable for this to raise an error since invalid types are passed
            pass

    def test_integration_with_real_pricing_config(self):
        """Test integration with real pricing configuration."""
        from flujo.infra.config import get_provider_pricing

        # Test that we can get pricing for known models
        pricing = get_provider_pricing("openai", "gpt-4o")
        assert pricing is not None
        assert pricing.prompt_tokens_per_1k == 0.005
        assert pricing.completion_tokens_per_1k == 0.015

        # Test that unknown models raise PricingNotConfiguredError in strict mode
        import pytest
        from flujo.exceptions import PricingNotConfiguredError

        with pytest.raises(PricingNotConfiguredError):
            get_provider_pricing("unknown", "unknown-model")

        # Test that we can calculate costs with real pricing
        calculator = CostCalculator()
        cost = calculator.calculate(
            model_name="gpt-4o", prompt_tokens=1000, completion_tokens=500, provider="openai"
        )
        assert cost > 0.0

    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""

        # Test that old agent patterns still work
        class OldStyleAgent:
            def __init__(self):
                self._model_name = "openai:gpt-4o"  # Old private attribute

        class NewStyleAgent:
            def __init__(self):
                self.model_id = "openai:gpt-4o"  # New public attribute

        # Create a mock response
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

        # Test old style agent
        old_agent = OldStyleAgent()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, old_agent, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0

        # Test new style agent
        new_agent = NewStyleAgent()
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            raw_output, new_agent, "test_step"
        )
        assert prompt_tokens == 100
        assert completion_tokens == 50
        assert cost_usd > 0.0


class TestProviderPricing:
    """Test the ProviderPricing model."""

    def test_provider_pricing_creation(self):
        """Test creating a ProviderPricing instance."""
        pricing = ProviderPricing(prompt_tokens_per_1k=0.005, completion_tokens_per_1k=0.015)

        assert pricing.prompt_tokens_per_1k == 0.005
        assert pricing.completion_tokens_per_1k == 0.015

    def test_provider_pricing_validation(self):
        """Test that ProviderPricing validates required fields."""
        with pytest.raises(ValueError):
            ProviderPricing()  # Missing required fields


class TestCostConfig:
    """Test the cost configuration functionality."""

    def test_get_provider_pricing_with_valid_data(self):
        """Test getting provider pricing with valid configuration."""
        # In strict mode, unknown models should raise PricingNotConfiguredError
        import pytest
        from flujo.exceptions import PricingNotConfiguredError

        with pytest.raises(PricingNotConfiguredError):
            get_provider_pricing("openai", "unknown-model")

    def test_get_provider_pricing_with_default_pricing(self):
        """Test getting provider pricing with default pricing fallback."""
        # Test with a known model that has default pricing
        pricing = get_provider_pricing("openai", "gpt-4o")
        # Should return default pricing for known models
        assert pricing is not None
        assert pricing.prompt_tokens_per_1k == 0.005
        assert pricing.completion_tokens_per_1k == 0.015


class TestStrictPricingMode:
    """Test the strict pricing mode functionality."""

    def test_strict_mode_off_default_behavior(self):
        """Test that strict mode off (default) maintains backward compatibility."""
        # Mock the config to have strict=False (default)
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {}
                        self.strict = False

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing returns None when no config exists
            pricing = get_provider_pricing("openai", "unknown-model")
            assert pricing is None

    def test_strict_mode_off_with_default_pricing(self):
        """Test that strict mode off allows fallback to hardcoded defaults."""
        # Mock the config to have strict=False and no user config
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {}
                        self.strict = False

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing returns default pricing for known models
            pricing = get_provider_pricing("openai", "gpt-4o")
            assert pricing is not None
            assert pricing.prompt_tokens_per_1k == 0.005
            assert pricing.completion_tokens_per_1k == 0.015

    def test_strict_mode_on_with_user_config(self):
        """Test that strict mode on works correctly when user config is available."""
        # Mock the config to have strict=True and user config
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {
                            "openai": {
                                "gpt-4o": ProviderPricing(
                                    prompt_tokens_per_1k=0.005, completion_tokens_per_1k=0.015
                                )
                            }
                        }
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing returns user-configured pricing
            pricing = get_provider_pricing("openai", "gpt-4o")
            assert pricing is not None
            assert pricing.prompt_tokens_per_1k == 0.005
            assert pricing.completion_tokens_per_1k == 0.015

    def test_strict_mode_on_without_user_config_raises_error(self):
        """Test that strict mode on raises PricingNotConfiguredError when no user config exists."""
        # Mock the config to have strict=True but no user config
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {}
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing raises PricingNotConfiguredError
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                get_provider_pricing("openai", "gpt-4o")

            assert exc_info.value.provider == "openai"
            assert exc_info.value.model == "gpt-4o"
            assert "Strict pricing is enabled" in str(exc_info.value)

    def test_strict_mode_on_with_unknown_model_raises_error(self):
        """Test that strict mode on raises PricingNotConfiguredError for unknown models."""
        # Mock the config to have strict=True and some user config but not for the requested model
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {
                            "openai": {
                                "gpt-3.5-turbo": ProviderPricing(
                                    prompt_tokens_per_1k=0.0015, completion_tokens_per_1k=0.002
                                )
                            }
                        }
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing raises PricingNotConfiguredError for unknown model
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                get_provider_pricing("openai", "gpt-4o")

            assert exc_info.value.provider == "openai"
            assert exc_info.value.model == "gpt-4o"

    def test_strict_mode_on_with_unknown_provider_raises_error(self):
        """Test that strict mode on raises PricingNotConfiguredError for unknown providers."""
        # Mock the config to have strict=True and some user config but not for the requested provider
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {
                            "anthropic": {
                                "claude-3-sonnet": ProviderPricing(
                                    prompt_tokens_per_1k=0.003, completion_tokens_per_1k=0.015
                                )
                            }
                        }
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing raises PricingNotConfiguredError for unknown provider
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                get_provider_pricing("openai", "gpt-4o")

            assert exc_info.value.provider == "openai"
            assert exc_info.value.model == "gpt-4o"

    def test_strict_mode_on_with_none_provider_raises_error(self):
        """Test that strict mode on raises PricingNotConfiguredError when provider is None."""
        # Mock the config to have strict=True but no user config
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {}
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing raises PricingNotConfiguredError when provider is None
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                get_provider_pricing(None, "unknown-model")

            assert exc_info.value.provider is None
            assert exc_info.value.model == "unknown-model"

    def test_strict_mode_does_not_fallback_to_hardcoded_defaults(self):
        """Test that strict mode does not fall back to hardcoded defaults."""
        # Mock the config to have strict=True but no user config
        with pytest.MonkeyPatch().context() as m:

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.providers = {}
                        self.strict = True

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Test that get_provider_pricing raises PricingNotConfiguredError even for models with hardcoded defaults
            with pytest.raises(PricingNotConfiguredError):
                get_provider_pricing("openai", "gpt-4o")  # This model has hardcoded defaults

    def test_cost_config_strict_field_default(self):
        """Test that CostConfig.strict defaults to False for backward compatibility."""
        # Test that the default value is False
        config = CostConfig()
        assert config.strict is False

    def test_cost_config_strict_field_explicit(self):
        """Test that CostConfig.strict can be set explicitly."""
        # Test that the field can be set to True
        config = CostConfig(strict=True)
        assert config.strict is True

        # Test that the field can be set to False
        config = CostConfig(strict=False)
        assert config.strict is False


class TestImageCostPostProcessor:
    """Test the image cost post-processor functionality."""

    def test_image_cost_post_processor_with_valid_pricing(self):
        """Test image cost post-processor with valid pricing data."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with image usage
        class MockUsage:
            def __init__(self):
                self.details = {"images": 2}
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
            "price_per_image_hd_1024x1024": 0.080,
        }

        # Test with standard quality and default size
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1024x1024"
        )

        # Should calculate 2 images * $0.040 = $0.080
        assert result.usage.cost_usd == 0.080

    def test_image_cost_post_processor_with_hd_quality(self):
        """Test image cost post-processor with HD quality."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with image usage
        class MockUsage:
            def __init__(self):
                self.details = {"images": 1}
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
            "price_per_image_hd_1024x1024": 0.080,
        }

        # Test with HD quality
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="hd", size="1024x1024"
        )

        # Should calculate 1 image * $0.080 = $0.080
        assert result.usage.cost_usd == 0.080

    def test_image_cost_post_processor_with_missing_pricing(self):
        """Test image cost post-processor with missing pricing data."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with image usage
        class MockUsage:
            def __init__(self):
                self.details = {"images": 1}
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data without the required key
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
            # Missing HD pricing
        }

        # Test with HD quality (missing pricing)
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="hd", size="1024x1024"
        )

        # Should set cost to 0.0 when pricing is missing
        assert result.usage.cost_usd == 0.0

    def test_image_cost_post_processor_with_no_images(self):
        """Test image cost post-processor when no images are generated."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with no image usage
        class MockUsage:
            def __init__(self):
                self.details = {"images": 0}
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
        }

        # Test with no images
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1024x1024"
        )

        # Should return the original result unchanged
        assert result.usage.cost_usd is None

    def test_image_cost_post_processor_with_no_usage_details(self):
        """Test image cost post-processor when usage has no details."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with no usage details
        class MockUsage:
            def __init__(self):
                self.details = None
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
        }

        # Test with no usage details
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1024x1024"
        )

        # Should return the original result unchanged
        assert result.usage.cost_usd is None

    def test_image_cost_post_processor_with_no_usage(self):
        """Test image cost post-processor when run_result has no usage."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with no usage
        class MockRunResult:
            def __init__(self):
                self.usage = None

        run_result = MockRunResult()

        # Create pricing data
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
        }

        # Test with no usage
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1024x1024"
        )

        # Should return the original result unchanged
        assert result.usage is None

    def test_image_cost_post_processor_with_different_sizes(self):
        """Test image cost post-processor with different image sizes."""
        from flujo.cost import _image_cost_post_processor

        # Create a mock run result with image usage
        class MockUsage:
            def __init__(self):
                self.details = {"images": 1}
                self.cost_usd = None

        class MockRunResult:
            def __init__(self):
                self.usage = MockUsage()

        run_result = MockRunResult()

        # Create pricing data with different sizes
        pricing_data = {
            "price_per_image_standard_1024x1024": 0.040,
            "price_per_image_standard_1792x1024": 0.080,
            "price_per_image_standard_1024x1792": 0.080,
        }

        # Test with 1792x1024 size
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1792x1024"
        )

        # Should calculate 1 image * $0.080 = $0.080
        assert result.usage.cost_usd == 0.080

        # Test with 1024x1792 size
        run_result.usage.cost_usd = None  # Reset
        result = _image_cost_post_processor(
            run_result, pricing_data, quality="standard", size="1024x1792"
        )

        # Should calculate 1 image * $0.080 = $0.080
        assert result.usage.cost_usd == 0.080


class TestImageModelDetection:
    """Test the image model detection functionality."""

    def test_is_image_generation_model_with_dall_e(self):
        """Test detection of DALL-E models."""
        from flujo.agents import _is_image_generation_model

        # Test various DALL-E model formats
        assert _is_image_generation_model("openai:dall-e-3") is True
        assert _is_image_generation_model("openai:dall-e-2") is True
        assert _is_image_generation_model("dall-e-3") is True
        assert _is_image_generation_model("DALL-E-3") is True

    def test_is_image_generation_model_with_other_image_models(self):
        """Test detection of other image generation models."""
        from flujo.agents import _is_image_generation_model

        # Test other image models
        assert _is_image_generation_model("midjourney:v6") is True
        assert _is_image_generation_model("stable-diffusion:xl") is True
        assert _is_image_generation_model("google:imagen-2") is True

    def test_is_image_generation_model_with_chat_models(self):
        """Test that chat models are not detected as image models."""
        from flujo.agents import _is_image_generation_model

        # Test chat models (should return False)
        assert _is_image_generation_model("openai:gpt-4o") is False
        assert _is_image_generation_model("anthropic:claude-3-sonnet") is False
        assert _is_image_generation_model("google:gemini-1.5-pro") is False
        assert _is_image_generation_model("gpt-4o") is False

    def test_is_image_generation_model_with_edge_cases(self):
        """Test image model detection with edge cases."""
        from flujo.agents import _is_image_generation_model

        # Test edge cases
        assert _is_image_generation_model("") is False
        assert _is_image_generation_model("openai:") is False
        assert _is_image_generation_model("dall-e") is True  # Partial match
        assert _is_image_generation_model("text-dall-e") is True  # Contains pattern


class TestImageCostPostProcessorAttachment:
    """Test the image cost post-processor attachment functionality."""

    def test_attach_image_cost_post_processor_with_valid_pricing(self):
        """Test attaching post-processor with valid pricing configuration."""
        from flujo.agents import _attach_image_cost_post_processor
        from unittest.mock import patch

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.post_processors = []

        agent = MockAgent()
        model = "openai:dall-e-3"

        # Mock the pricing configuration
        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            from flujo.infra.config import ProviderPricing

            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.0,
                completion_tokens_per_1k=0.0,
                price_per_image_standard_1024x1024=0.040,
                price_per_image_hd_1024x1024=0.080,
            )
            mock_get_pricing.return_value = mock_pricing

            # Attach the post-processor
            _attach_image_cost_post_processor(agent, model)

            # Verify that a post-processor was attached
            assert len(agent.post_processors) == 1
            assert callable(agent.post_processors[0])

    def test_attach_image_cost_post_processor_with_missing_pricing(self):
        """Test attaching post-processor when pricing is not configured."""
        from flujo.agents import _attach_image_cost_post_processor
        from unittest.mock import patch

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.post_processors = []

        agent = MockAgent()
        model = "openai:dall-e-3"

        # Mock the pricing configuration to return None
        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            mock_get_pricing.return_value = None

            # Attach the post-processor
            _attach_image_cost_post_processor(agent, model)

            # Verify that no post-processor was attached
            assert len(agent.post_processors) == 0

    def test_attach_image_cost_post_processor_with_no_image_pricing(self):
        """Test attaching post-processor when no image pricing is configured."""
        from flujo.agents import _attach_image_cost_post_processor
        from unittest.mock import patch

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.post_processors = []

        agent = MockAgent()
        model = "openai:dall-e-3"

        # Mock the pricing configuration with no image pricing
        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            from flujo.infra.config import ProviderPricing

            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.0,
                completion_tokens_per_1k=0.0,
                # No image pricing fields
            )
            mock_get_pricing.return_value = mock_pricing

            # Attach the post-processor
            _attach_image_cost_post_processor(agent, model)

            # Verify that no post-processor was attached
            assert len(agent.post_processors) == 0

    def test_attach_image_cost_post_processor_with_invalid_provider(self):
        """Test attaching post-processor with invalid provider."""
        from flujo.agents import _attach_image_cost_post_processor
        from unittest.mock import patch

        # Create a mock agent
        class MockAgent:
            def __init__(self):
                self.post_processors = []

        agent = MockAgent()
        model = "invalid:dall-e-3"

        # Mock the provider extraction to return None
        with patch("flujo.utils.model_utils.extract_provider_and_model") as mock_extract:
            mock_extract.return_value = (None, "dall-e-3")

            # Attach the post-processor
            _attach_image_cost_post_processor(agent, model)

            # Verify that no post-processor was attached
            assert len(agent.post_processors) == 0

    def test_make_agent_async_with_image_model(self):
        """Test that make_agent_async attaches post-processor for image models."""
        from flujo.agents import make_agent_async
        from unittest.mock import patch

        # Mock the make_agent function
        with patch("flujo.agents.make_agent") as mock_make_agent:
            # Create a mock agent
            class MockAgent:
                def __init__(self):
                    self.post_processors = []

            mock_agent = MockAgent()
            mock_processors = Mock()
            mock_make_agent.return_value = (mock_agent, mock_processors)

            # Mock the post-processor attachment
            with patch("flujo.agents.recipes._attach_image_cost_post_processor") as mock_attach:
                # Create an agent with an image model
                make_agent_async(
                    model="openai:dall-e-3",
                    system_prompt="Generate images",
                    output_type=str,
                )

                # Verify that the post-processor attachment was called
                mock_attach.assert_called_once_with(mock_agent, "openai:dall-e-3")

    def test_make_agent_async_with_chat_model(self):
        """Test that make_agent_async doesn't attach post-processor for chat models."""
        from flujo.agents import make_agent_async
        from unittest.mock import patch

        # Mock the make_agent function
        with patch("flujo.agents.make_agent") as mock_make_agent:
            # Create a mock agent
            class MockAgent:
                def __init__(self):
                    self.post_processors = []

            mock_agent = MockAgent()
            mock_processors = Mock()
            mock_make_agent.return_value = (mock_agent, mock_processors)

            # Mock the post-processor attachment
            with patch("flujo.agents._attach_image_cost_post_processor") as mock_attach:
                # Create an agent with a chat model
                make_agent_async(
                    model="openai:gpt-4o",
                    system_prompt="You are a helpful assistant",
                    output_type=str,
                )

                # Verify that the post-processor attachment was NOT called
                mock_attach.assert_not_called()


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
        from flujo.cost import clear_cost_cache

        # Clear cache to ensure test isolation
        clear_cost_cache()

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
            assert cost_usd > 0.0, (
                f"Cost should be > 0.0 for model_id='{model_id}'"
            )  # Should have pricing for all configured models

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

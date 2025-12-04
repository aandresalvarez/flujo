"""Unit tests for embedding cost tracking functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from flujo.embeddings.models import EmbeddingResult
from flujo.embeddings.clients.openai_client import OpenAIEmbeddingClient
from flujo.embeddings import get_embedding_client
from flujo.cost import CostCalculator
from flujo.exceptions import PricingNotConfiguredError

try:
    from pydantic_ai.usage import RunUsage
except ImportError:
    from typing import Any

    try:
        from pydantic_ai.usage import Usage as RunUsage
    except ImportError:
        RunUsage = Any


def create_run_usage(input_tokens=0, output_tokens=0):
    """Helper to create RunUsage compatible with different pydantic-ai versions."""
    try:
        return RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    except TypeError:
        return RunUsage(
            request_tokens=input_tokens,
            response_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


class TestEmbeddingResult:
    """Test the EmbeddingResult dataclass."""

    def test_embedding_result_creation(self):
        """Test creating an EmbeddingResult instance."""
        usage_info = create_run_usage(input_tokens=100, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        assert result.embeddings == embeddings
        assert result.usage_info == usage_info

    def test_embedding_result_usage_method(self):
        """Test that EmbeddingResult implements the UsageReportingProtocol."""
        usage_info = create_run_usage(input_tokens=100, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3]]

        result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        # Test that it has a usage() method that returns the usage_info
        assert hasattr(result, "usage")
        assert callable(result.usage)
        assert result.usage() == usage_info

    def test_embedding_result_with_zero_tokens(self):
        """Test EmbeddingResult with zero token usage."""
        usage_info = create_run_usage(input_tokens=0, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3]]

        result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        usage = result.usage()
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        total_tokens = getattr(usage, "total_tokens", 0)

        assert input_tokens == 0
        assert total_tokens == 0


class TestOpenAIEmbeddingClient:
    """Test the OpenAIEmbeddingClient class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = AsyncMock()

        # Mock the embeddings.create method
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, total_tokens=100)

        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        return mock_client

    def test_openai_embedding_client_initialization(self):
        """Test OpenAIEmbeddingClient initialization."""
        client = OpenAIEmbeddingClient("text-embedding-3-large")

        assert client.model_name == "text-embedding-3-large"
        assert client.model_id == "openai:text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_openai_embedding_client_embed_method(self, mock_openai_client):
        """Test the embed method of OpenAIEmbeddingClient."""
        with patch(
            "flujo.embeddings.clients.openai_client.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            client = OpenAIEmbeddingClient("text-embedding-3-large")

            texts = ["Hello world", "Test embedding"]
            result = await client.embed(texts)

            # Verify the API was called correctly
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-large", input=texts
            )

            # Verify the result structure
            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 2
            assert result.embeddings[0] == [0.1, 0.2, 0.3]
            assert result.embeddings[1] == [0.4, 0.5, 0.6]

            # Verify usage information
            usage = result.usage()
            input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
            total_tokens = getattr(usage, "total_tokens", 0)

            assert input_tokens == 100
            assert total_tokens == 100

    @pytest.mark.asyncio
    async def test_openai_embedding_client_embed_single_text(self, mock_openai_client):
        """Test embedding a single text."""
        with patch(
            "flujo.embeddings.clients.openai_client.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            client = OpenAIEmbeddingClient("text-embedding-3-large")

            text = "Single text"
            result = await client.embed([text])

            # Verify the API was called with single text
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-large", input=[text]
            )

            assert len(result.embeddings) == 2  # Mock returns 2 embeddings
            assert isinstance(result, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_openai_embedding_client_embed_empty_list(self, mock_openai_client):
        """Test embedding an empty list of texts."""
        with patch(
            "flujo.embeddings.clients.openai_client.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            client = OpenAIEmbeddingClient("text-embedding-3-large")

            result = await client.embed([])

            # Verify the API was called with empty list
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-large", input=[]
            )

            assert isinstance(result, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_openai_embedding_client_api_error_handling(self, mock_openai_client):
        """Test error handling when the API call fails."""
        # Mock API error
        mock_openai_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        with patch(
            "flujo.embeddings.clients.openai_client.openai.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            client = OpenAIEmbeddingClient("text-embedding-3-large")

            with pytest.raises(Exception, match="API Error"):
                await client.embed(["test"])

    def test_openai_embedding_client_model_id_format(self):
        """Test that model_id is correctly formatted."""
        client = OpenAIEmbeddingClient("text-embedding-3-large")
        assert client.model_id == "openai:text-embedding-3-large"

        client = OpenAIEmbeddingClient("text-embedding-ada-002")
        assert client.model_id == "openai:text-embedding-ada-002"


class TestGetEmbeddingClient:
    """Test the get_embedding_client factory function."""

    def test_get_embedding_client_openai(self):
        """Test getting an OpenAI embedding client."""
        client = get_embedding_client("openai:text-embedding-3-large")

        assert isinstance(client, OpenAIEmbeddingClient)
        assert client.model_name == "text-embedding-3-large"
        assert client.model_id == "openai:text-embedding-3-large"

    def test_get_embedding_client_unknown_provider(self):
        """Test that unknown providers raise an error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_embedding_client("unknown:text-embedding-3-large")

    def test_get_embedding_client_invalid_format(self):
        """Test that invalid format raises an error."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            get_embedding_client("invalid-format")

    def test_get_embedding_client_missing_model(self):
        """Test that missing model raises an error."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            get_embedding_client("openai:")

    def test_get_embedding_client_empty_provider(self):
        """Test that empty provider raises an error."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            get_embedding_client(":text-embedding-3-large")


class TestCostCalculatorForEmbeddings:
    """Test the CostCalculator for embedding models."""

    def test_calculate_cost_for_embedding_model(self):
        """Test cost calculation for embedding models with tokens_per_1k pricing."""
        calculator = CostCalculator()

        # Mock pricing for embedding model
        from flujo.infra.config import ProviderPricing

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            # Mock embedding model pricing (only tokens_per_1k, no prompt/completion distinction)
            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.00013,  # For embeddings, this is tokens_per_1k
                completion_tokens_per_1k=0.00013,  # Same for embeddings
            )
            mock_get_pricing.return_value = mock_pricing

            # Calculate cost for embedding model
            cost = calculator.calculate(
                model_name="text-embedding-3-large",
                prompt_tokens=1000,  # For embeddings, this is total_tokens
                completion_tokens=0,  # Not used for embeddings
                provider="openai",
            )

            # Expected: (1000/1000 * 0.00013) = 0.00013
            assert cost == 0.00013

    def test_calculate_cost_for_embedding_model_with_completion_tokens(self):
        """Test cost calculation for embedding models when completion_tokens is provided."""
        calculator = CostCalculator()

        # Mock pricing for embedding model
        from flujo.infra.config import ProviderPricing

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            # Mock embedding model pricing
            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00013
            )
            mock_get_pricing.return_value = mock_pricing

            # Calculate cost for embedding model with completion_tokens
            cost = calculator.calculate(
                model_name="text-embedding-3-large",
                prompt_tokens=500,
                completion_tokens=500,  # Should be treated the same as prompt_tokens for embeddings
                provider="openai",
            )

            # Expected: (500/1000 * 0.00013) + (500/1000 * 0.00013) = 0.000065 + 0.000065 = 0.00013
            assert cost == 0.00013

    def test_calculate_cost_for_embedding_model_no_pricing(self):
        """Test cost calculation for embedding models when no pricing is configured."""
        calculator = CostCalculator()

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            mock_get_pricing.return_value = None

            # Should return 0.0 when no pricing is configured
            cost = calculator.calculate(
                model_name="text-embedding-3-large",
                prompt_tokens=1000,
                completion_tokens=0,
                provider="openai",
            )

            assert cost == 0.0

    def test_calculate_cost_for_embedding_model_strict_mode(self):
        """Test cost calculation for embedding models in strict mode."""
        calculator = CostCalculator()

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            # Mock strict mode error
            mock_get_pricing.side_effect = PricingNotConfiguredError(
                "openai", "text-embedding-3-large"
            )

            # Should raise PricingNotConfiguredError in strict mode
            with pytest.raises(PricingNotConfiguredError):
                calculator.calculate(
                    model_name="text-embedding-3-large",
                    prompt_tokens=1000,
                    completion_tokens=0,
                    provider="openai",
                )

    def test_calculate_cost_for_embedding_model_provider_inference(self):
        """Test cost calculation for embedding models with automatic provider inference."""
        calculator = CostCalculator()

        # Mock pricing for embedding model
        from flujo.infra.config import ProviderPricing

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00013
            )
            mock_get_pricing.return_value = mock_pricing

            # Test with OpenAI embedding model (should infer provider)
            cost = calculator.calculate(
                model_name="text-embedding-3-large", prompt_tokens=1000, completion_tokens=0
            )

            assert cost == 0.00013

    def test_calculate_cost_for_embedding_model_edge_cases(self):
        """Test cost calculation for embedding models with edge cases."""
        calculator = CostCalculator()

        # Mock pricing for embedding model
        from flujo.infra.config import ProviderPricing

        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00013
            )
            mock_get_pricing.return_value = mock_pricing

            # Test with zero tokens
            cost = calculator.calculate(
                model_name="text-embedding-3-large",
                prompt_tokens=0,
                completion_tokens=0,
                provider="openai",
            )
            assert cost == 0.0

            # Test with very large token counts
            cost = calculator.calculate(
                model_name="text-embedding-3-large",
                prompt_tokens=1000000,
                completion_tokens=0,
                provider="openai",
            )
            assert abs(cost - 0.13) < 0.000001  # 1000000/1000 * 0.00013


class TestEmbeddingIntegrationWithCostTracking:
    """Test integration between embedding clients and cost tracking."""

    @pytest.mark.asyncio
    async def test_embedding_result_with_cost_tracking(self):
        """Test that EmbeddingResult works with the existing cost tracking system."""
        from flujo.cost import extract_usage_metrics

        # Create an embedding result
        usage_info = create_run_usage(input_tokens=100, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedding_result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        # Create a mock agent with model_id
        class MockEmbeddingAgent:
            def __init__(self):
                self.model_id = "openai:text-embedding-3-large"

        agent = MockEmbeddingAgent()

        # Test that extract_usage_metrics can handle EmbeddingResult
        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            from flujo.infra.config import ProviderPricing

            mock_pricing = ProviderPricing(
                prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00013
            )
            mock_get_pricing.return_value = mock_pricing

            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                embedding_result, agent, "test_embedding_step"
            )

            assert prompt_tokens == 100
            assert completion_tokens == 0  # Embeddings don't have completion tokens
            assert cost_usd > 0.0  # Should calculate cost based on pricing

    @pytest.mark.asyncio
    async def test_embedding_result_without_model_id(self):
        """Test that EmbeddingResult works when agent has no model_id."""
        from flujo.cost import extract_usage_metrics, clear_cost_cache
        from flujo.utils.model_utils import clear_model_id_cache

        # Clear caches to ensure test isolation
        clear_cost_cache()
        clear_model_id_cache()

        # Create an embedding result
        usage_info = create_run_usage(input_tokens=100, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3]]
        embedding_result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        # Create a mock agent without model_id
        class MockEmbeddingAgent:
            def __init__(self):
                pass  # No model_id

        agent = MockEmbeddingAgent()

        # Should handle gracefully and return 0.0 cost
        prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
            embedding_result, agent, "test_embedding_step"
        )

        assert prompt_tokens == 100
        assert completion_tokens == 0
        assert cost_usd == 0.0  # Should return 0.0 for safety

    @pytest.mark.asyncio
    async def test_embedding_result_with_strict_mode(self):
        """Test that EmbeddingResult works correctly in strict mode."""
        from flujo.cost import extract_usage_metrics

        # Create an embedding result
        usage_info = create_run_usage(input_tokens=100, output_tokens=0)
        embeddings = [[0.1, 0.2, 0.3]]
        embedding_result = EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

        # Create a mock agent with model_id
        class MockEmbeddingAgent:
            def __init__(self):
                self.model_id = "openai:text-embedding-3-large"

        agent = MockEmbeddingAgent()

        # Test that PricingNotConfiguredError is raised in strict mode
        with patch("flujo.infra.config.get_provider_pricing") as mock_get_pricing:
            mock_get_pricing.side_effect = PricingNotConfiguredError(
                "openai", "text-embedding-3-large"
            )

            with pytest.raises(PricingNotConfiguredError):
                extract_usage_metrics(embedding_result, agent, "test_embedding_step")


class TestEmbeddingModelPricing:
    """Test pricing configuration for embedding models."""

    def test_embedding_model_pricing_configuration(self):
        """Test that embedding models can be configured with tokens_per_1k pricing."""
        from flujo.infra.config import ProviderPricing

        # Test that ProviderPricing can handle embedding model pricing
        # For embeddings, prompt_tokens_per_1k and completion_tokens_per_1k should be the same
        pricing = ProviderPricing(prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00013)

        assert pricing.prompt_tokens_per_1k == 0.00013
        assert pricing.completion_tokens_per_1k == 0.00013

    def test_embedding_model_pricing_validation(self):
        """Test that ProviderPricing validates required fields for embedding models."""
        from flujo.infra.config import ProviderPricing

        with pytest.raises(ValueError):
            # Should raise error when required fields are missing
            ProviderPricing()

    def test_embedding_model_pricing_different_values(self):
        """Test that embedding models can have different prompt and completion pricing."""
        from flujo.infra.config import ProviderPricing

        # Some embedding models might have different pricing for input vs output
        pricing = ProviderPricing(prompt_tokens_per_1k=0.00013, completion_tokens_per_1k=0.00015)

        assert pricing.prompt_tokens_per_1k == 0.00013
        assert pricing.completion_tokens_per_1k == 0.00015

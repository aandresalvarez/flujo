"""Integration tests for embedding cost tracking functionality."""

import pytest
import tempfile

from pathlib import Path
from unittest.mock import AsyncMock

from flujo import Step, Pipeline
from flujo.embeddings import get_embedding_client, EmbeddingResult
from flujo.exceptions import PricingNotConfiguredError
from tests.conftest import create_test_flujo


class DocumentEmbeddingAgent:
    """Example agent that uses embedding functionality."""

    def __init__(self):
        # Agent declares the client it needs
        self.client = get_embedding_client("openai:text-embedding-3-large")
        self.model_id = self.client.model_id  # Expose model_id for metric extraction

    async def run(self, document: str) -> EmbeddingResult:
        # The agent's job is to call the client and return the result
        return await self.client.embed(texts=[document])


class TestEmbeddingCostIntegration:
    """Test integration of embedding cost tracking with pipelines."""

    @pytest.fixture
    def flujo_toml_with_embedding_pricing(self):
        """Create a flujo.toml file with embedding model pricing."""
        content = """
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015

[cost.providers.openai.text-embedding-3-large]
prompt_tokens_per_1k = 0.00013
completion_tokens_per_1k = 0.00013
"""
        return content

    @pytest.fixture
    def flujo_toml_without_embedding_pricing(self):
        """Create a flujo.toml file without embedding model pricing."""
        content = """
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        return content

    @pytest.fixture
    def flujo_toml_non_strict(self):
        """Create a flujo.toml file with non-strict mode."""
        content = """
[cost]
strict = false

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        return content

    def create_temp_config(self, content: str) -> Path:
        """Create a temporary flujo.toml file."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "flujo.toml"
        config_path.write_text(content)
        return config_path

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_success(self, flujo_toml_with_embedding_pricing):
        """Test successful embedding cost tracking with configured pricing."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_embedding_pricing)

        # Mock the OpenAI client to avoid actual API calls
        with pytest.MonkeyPatch().context() as m:
            # Mock the OpenAI client
            mock_response = type(
                "MockResponse",
                (),
                {
                    "data": [type("MockEmbedding", (), {"embedding": [0.1, 0.2, 0.3]})()],
                    "usage": type("MockUsage", (), {"prompt_tokens": 100, "total_tokens": 100})(),
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "embeddings": type(
                        "MockEmbeddings", (), {"create": AsyncMock(return_value=mock_response)}
                    )()
                },
            )()

            m.setattr("openai.AsyncOpenAI", lambda: mock_client)

            # Set the config path
            m.setenv("FLUJO_CONFIG_PATH", str(config_path))

            # Create the pipeline
            embedding_step = Step(name="EmbedDocument", agent=DocumentEmbeddingAgent())

            pipeline = Pipeline(name="test_embedding_pipeline", steps=[embedding_step])

            # Create runner and run the pipeline
            runner = create_test_flujo(pipeline)
            result = None
            async for r in runner.run_async({"document": "This is a test document for embedding."}):
                result = r
            assert result is not None, "No result returned from runner.run_async()"

            # Assertions
            assert len(result.step_history) == 1
            assert result.step_history[0].success

            step_result = result.step_history[0]
            assert step_result.name == "EmbedDocument"
            assert step_result.cost_usd > 0.0  # Should have calculated cost
            assert step_result.token_counts > 0  # Should have token counts

            # Verify the cost calculation is reasonable
            # 100 tokens * 0.00013 per 1k tokens = 0.000013 USD
            expected_cost = 100 / 1000.0 * 0.00013
            assert abs(step_result.cost_usd - expected_cost) < 0.000001

            # Verify the embeddings were returned
            assert isinstance(step_result.output, EmbeddingResult)
            assert len(step_result.output.embeddings) == 1
            assert len(step_result.output.embeddings[0]) == 3  # Mock embedding vector

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_strict_mode_failure(
        self, flujo_toml_without_embedding_pricing
    ):
        """Test that strict mode fails when embedding pricing is not configured."""
        # Create temporary config without embedding pricing
        config_path = self.create_temp_config(flujo_toml_without_embedding_pricing)

        with pytest.MonkeyPatch().context() as m:
            # Mock the OpenAI client to avoid actual API calls
            mock_response = type(
                "MockResponse",
                (),
                {
                    "data": [type("MockEmbedding", (), {"embedding": [0.1, 0.2, 0.3]})()],
                    "usage": type("MockUsage", (), {"prompt_tokens": 100, "total_tokens": 100})(),
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "embeddings": type(
                        "MockEmbeddings", (), {"create": AsyncMock(return_value=mock_response)}
                    )()
                },
            )()

            m.setattr("openai.AsyncOpenAI", lambda: mock_client)

            # Mock the config manager to use our temporary file
            def mock_get_config_manager():
                class MockConfigManager:
                    def __init__(self):
                        self.config_path = config_path

                    def load_config(self):
                        class MockConfig:
                            def __init__(self):
                                self.cost = {
                                    "strict": True,
                                    "providers": {
                                        "openai": {
                                            "gpt-4o": {
                                                "prompt_tokens_per_1k": 0.005,
                                                "completion_tokens_per_1k": 0.015,
                                            }
                                        }
                                    },
                                }

                        return MockConfig()

                    def get_settings(self):
                        from flujo.infra.settings import settings as _settings

                        return _settings

                return MockConfigManager()

            m.setattr("flujo.infra.config_manager.get_config_manager", mock_get_config_manager)

            # Create the pipeline
            embedding_step = Step(name="EmbedDocument", agent=DocumentEmbeddingAgent())

            pipeline = Pipeline(name="test_embedding_pipeline", steps=[embedding_step])

            # Create runner and run the pipeline - should fail with PricingNotConfiguredError
            runner = create_test_flujo(pipeline)
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                async for r in runner.run_async(
                    {"document": "This is a test document for embedding."}
                ):
                    pass

            # Verify the error message
            assert "text-embedding-3-large" in str(exc_info.value)
            assert "openai" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_non_strict_mode(self, flujo_toml_non_strict):
        """Test that non-strict mode works with default pricing."""
        # Create temporary config with non-strict mode
        config_path = self.create_temp_config(flujo_toml_non_strict)

        # Mock the OpenAI client to avoid actual API calls
        with pytest.MonkeyPatch().context() as m:
            # Mock the OpenAI client
            mock_response = type(
                "MockResponse",
                (),
                {
                    "data": [type("MockEmbedding", (), {"embedding": [0.1, 0.2, 0.3]})()],
                    "usage": type("MockUsage", (), {"prompt_tokens": 100, "total_tokens": 100})(),
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "embeddings": type(
                        "MockEmbeddings", (), {"create": AsyncMock(return_value=mock_response)}
                    )()
                },
            )()

            m.setattr("openai.AsyncOpenAI", lambda: mock_client)

            # Set the config path
            m.setenv("FLUJO_CONFIG_PATH", str(config_path))

            # Create the pipeline
            embedding_step = Step(name="EmbedDocument", agent=DocumentEmbeddingAgent())

            pipeline = Pipeline(name="test_embedding_pipeline", steps=[embedding_step])

            # Create runner and run the pipeline - should work with default pricing
            runner = create_test_flujo(pipeline)
            result = None
            async for r in runner.run_async({"document": "This is a test document for embedding."}):
                result = r
            assert result is not None, "No result returned from runner.run_async()"

            # Assertions
            assert len(result.step_history) == 1
            assert result.step_history[0].success

            step_result = result.step_history[0]
            assert step_result.name == "EmbedDocument"
            assert step_result.cost_usd > 0.0  # Should have calculated cost using default pricing
            assert step_result.token_counts > 0  # Should have token counts

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_multiple_texts(self, flujo_toml_with_embedding_pricing):
        """Test embedding cost tracking with multiple texts."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_embedding_pricing)

        # Mock the OpenAI client to avoid actual API calls
        with pytest.MonkeyPatch().context() as m:
            # Mock the OpenAI client with multiple embeddings
            mock_response = type(
                "MockResponse",
                (),
                {
                    "data": [
                        type("MockEmbedding", (), {"embedding": [0.1, 0.2, 0.3]})(),
                        type("MockEmbedding", (), {"embedding": [0.4, 0.5, 0.6]})(),
                    ],
                    "usage": type("MockUsage", (), {"prompt_tokens": 200, "total_tokens": 200})(),
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "embeddings": type(
                        "MockEmbeddings", (), {"create": AsyncMock(return_value=mock_response)}
                    )()
                },
            )()

            m.setattr("openai.AsyncOpenAI", lambda: mock_client)

            # Set the config path
            m.setenv("FLUJO_CONFIG_PATH", str(config_path))

            # Create an agent that embeds multiple texts
            class MultiTextEmbeddingAgent:
                def __init__(self):
                    self.client = get_embedding_client("openai:text-embedding-3-large")
                    self.model_id = self.client.model_id

                async def run(self, documents: list[str]) -> EmbeddingResult:
                    return await self.client.embed(texts=documents)

            # Create the pipeline
            embedding_step = Step(name="EmbedDocuments", agent=MultiTextEmbeddingAgent())

            pipeline = Pipeline(name="test_multi_embedding_pipeline", steps=[embedding_step])

            # Create runner and run the pipeline
            runner = create_test_flujo(pipeline)
            result = None
            async for r in runner.run_async(
                {"documents": ["This is the first document.", "This is the second document."]}
            ):
                result = r
            assert result is not None, "No result returned from runner.run_async()"

            # Assertions
            assert len(result.step_history) == 1
            assert result.step_history[0].success

            step_result = result.step_history[0]
            assert step_result.name == "EmbedDocuments"
            assert step_result.cost_usd > 0.0  # Should have calculated cost
            assert step_result.token_counts > 0  # Should have token counts

            # Verify the cost calculation is reasonable for multiple texts
            # 200 tokens * 0.00013 per 1k tokens = 0.000026 USD
            expected_cost = 200 / 1000.0 * 0.00013
            assert abs(step_result.cost_usd - expected_cost) < 0.000001

            # Verify the embeddings were returned
            assert isinstance(step_result.output, EmbeddingResult)
            assert len(step_result.output.embeddings) == 2  # Two documents
            assert len(step_result.output.embeddings[0]) == 3  # Mock embedding vector
            assert len(step_result.output.embeddings[1]) == 3  # Mock embedding vector

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_with_chat_agent(self, flujo_toml_with_embedding_pricing):
        """Test that embedding cost tracking doesn't break existing chat agent cost tracking."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_embedding_pricing)

        # Mock the OpenAI client to avoid actual API calls
        with pytest.MonkeyPatch().context() as m:
            # Mock the OpenAI client for embeddings
            mock_embedding_response = type(
                "MockResponse",
                (),
                {
                    "data": [type("MockEmbedding", (), {"embedding": [0.1, 0.2, 0.3]})()],
                    "usage": type("MockUsage", (), {"prompt_tokens": 100, "total_tokens": 100})(),
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "embeddings": type(
                        "MockEmbeddings",
                        (),
                        {"create": AsyncMock(return_value=mock_embedding_response)},
                    )()
                },
            )()

            m.setattr("openai.AsyncOpenAI", lambda: mock_client)

            # Set the config path
            m.setenv("FLUJO_CONFIG_PATH", str(config_path))

            # Create a chat agent
            class MockChatAgent:
                def __init__(self):
                    self.model_id = "openai:gpt-4o"

                async def run(self, message: str):
                    # Mock chat response
                    class MockChatResponse:
                        def __init__(self):
                            self.output = "This is a test response."

                        def usage(self):
                            class MockUsage:
                                def __init__(self):
                                    self.request_tokens = 50
                                    self.response_tokens = 25

                            return MockUsage()

                    return MockChatResponse()

            # Create the pipeline with both embedding and chat steps
            embedding_step = Step(name="EmbedDocument", agent=DocumentEmbeddingAgent())

            chat_step = Step(name="ChatResponse", agent=MockChatAgent())

            pipeline = Pipeline(name="test_mixed_pipeline", steps=[embedding_step, chat_step])

            # Create runner and run the pipeline
            runner = create_test_flujo(pipeline)
            result = None
            async for r in runner.run_async(
                {
                    "document": "This is a test document for embedding.",
                    "message": "Please analyze this document.",
                }
            ):
                result = r
            assert result is not None, "No result returned from runner.run_async()"

            # Assertions
            assert len(result.step_history) == 2
            assert result.step_history[0].success
            assert result.step_history[1].success

            # Check embedding step
            embedding_result = result.step_history[0]
            assert embedding_result.name == "EmbedDocument"
            assert embedding_result.cost_usd > 0.0

            # Check chat step
            chat_result = result.step_history[1]
            assert chat_result.name == "ChatResponse"
            assert chat_result.cost_usd > 0.0

            # Verify total cost is sum of both steps
            total_cost = sum(step.cost_usd for step in result.step_history)
            assert result.total_cost_usd == total_cost

    def test_embedding_model_pricing_configuration(self, flujo_toml_with_embedding_pricing):
        """Test that embedding model pricing can be configured in flujo.toml."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_embedding_pricing)

        # Verify the config file was created correctly
        config_content = config_path.read_text()
        assert "text-embedding-3-large" in config_content
        assert "0.00013" in config_content
        assert "strict = true" in config_content

    def test_embedding_model_pricing_validation(self, flujo_toml_without_embedding_pricing):
        """Test that embedding model pricing validation works correctly."""
        # Create temporary config without embedding pricing
        config_path = self.create_temp_config(flujo_toml_without_embedding_pricing)

        # Verify the config file was created correctly
        config_content = config_path.read_text()
        assert "text-embedding-3-large" not in config_content
        assert "strict = true" in config_content

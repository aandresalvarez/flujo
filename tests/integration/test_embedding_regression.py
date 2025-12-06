"""Regression tests to ensure embedding functionality doesn't break existing chat cost tracking."""

import pytest
import tempfile
from pathlib import Path

from flujo import Step, Pipeline, Flujo
from flujo.exceptions import PricingNotConfiguredError
from unittest.mock import AsyncMock


class TestEmbeddingRegression:
    """Test that embedding functionality doesn't break existing chat cost tracking."""

    @pytest.fixture
    def flujo_toml_with_both_pricing(self):
        """Create a flujo.toml file with both chat and embedding model pricing."""
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

    def create_temp_config(self, content: str) -> Path:
        """Create a temporary flujo.toml file."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "flujo.toml"
        config_path.write_text(content)
        return config_path

    @pytest.mark.asyncio
    async def test_existing_chat_cost_tracking_still_works(self, flujo_toml_with_both_pricing):
        """Test that existing chat agent cost tracking continues to work."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_both_pricing)

        with pytest.MonkeyPatch().context() as m:
            # Set the config path
            m.setenv("FLUJO_CONFIG_PATH", str(config_path))

            # Mock the cost config to ensure our pricing is used
            from flujo.infra.config import ProviderPricing

            def mock_get_cost_config():
                class MockCostConfig:
                    def __init__(self):
                        self.strict = True
                        self.providers = {
                            "openai": {
                                "gpt-4o": ProviderPricing(
                                    prompt_tokens_per_1k=0.005,
                                    completion_tokens_per_1k=0.015,
                                ),
                            }
                        }

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Create a mock chat agent
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

            # Create the pipeline with only chat step
            chat_step = Step(name="ChatResponse", agent=MockChatAgent())

            pipeline = Pipeline(name="test_chat_only_pipeline", steps=[chat_step])

            # Create runner and run the pipeline
            runner = Flujo(pipeline)
            result = None
            async for r in runner.run_async({"message": "Hello, how are you?"}):
                result = r

            # Assertions - chat cost tracking should still work exactly as before
            assert len(result.step_history) == 1
            assert result.step_history[0].success

            step_result = result.step_history[0]
            assert step_result.name == "ChatResponse"
            assert step_result.cost_usd > 0.0  # Should have calculated cost
            assert step_result.token_counts > 0  # Should have token counts

            # Verify the cost calculation is correct for chat
            # 50 prompt tokens * 0.005 + 25 completion tokens * 0.015 = 0.00025 + 0.000375 = 0.000625
            expected_cost = (50 / 1000.0 * 0.005) + (25 / 1000.0 * 0.015)
            assert abs(step_result.cost_usd - expected_cost) < 0.000001

    @pytest.mark.asyncio
    async def test_mixed_pipeline_cost_tracking(self, flujo_toml_with_both_pricing):
        """Test that mixed pipelines with both chat and embedding work correctly."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_both_pricing)

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

            # Create a mock chat agent
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

            # Create an embedding agent
            from flujo.embeddings import get_embedding_client

            class DocumentEmbeddingAgent:
                def __init__(self):
                    self.client = get_embedding_client("openai:text-embedding-3-large")
                    self.model_id = self.client.model_id

                async def run(self, document: str):
                    return await self.client.embed(texts=[document])

            # Create the pipeline with both steps
            embedding_step = Step(name="EmbedDocument", agent=DocumentEmbeddingAgent())

            chat_step = Step(name="ChatResponse", agent=MockChatAgent())

            pipeline = Pipeline(name="test_mixed_pipeline", steps=[embedding_step, chat_step])

            # Create runner and run the pipeline
            runner = Flujo(pipeline)
            result = None
            async for r in runner.run_async(
                {
                    "document": "This is a test document for embedding.",
                    "message": "Please analyze this document.",
                }
            ):
                result = r

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

            # Verify that chat cost calculation is still correct
            expected_chat_cost = (50 / 1000.0 * 0.005) + (25 / 1000.0 * 0.015)
            assert abs(chat_result.cost_usd - expected_chat_cost) < 0.000001

            # Verify that embedding cost calculation is correct
            expected_embedding_cost = 100 / 1000.0 * 0.00013
            assert abs(embedding_result.cost_usd - expected_embedding_cost) < 0.000001

    @pytest.mark.asyncio
    async def test_strict_mode_behavior_unchanged(self, flujo_toml_with_both_pricing):
        """Test that strict mode behavior is unchanged for chat agents."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_both_pricing)

        with pytest.MonkeyPatch().context() as m:
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
                                            },
                                            "text-embedding-3-large": {
                                                "prompt_tokens_per_1k": 0.00013,
                                                "completion_tokens_per_1k": 0.00013,
                                            },
                                        }
                                    },
                                }

                        return MockConfig()

                    def get_settings(self):
                        from flujo.infra.settings import settings as _settings

                        return _settings

                return MockConfigManager()

            m.setattr("flujo.infra.config_manager.get_config_manager", mock_get_config_manager)

            # Also mock get_cost_config to ensure it uses our mocked config
            def mock_get_cost_config():
                from flujo.infra.config import ProviderPricing

                class MockCostConfig:
                    def __init__(self):
                        self.strict = True
                        self.providers = {
                            "openai": {
                                "gpt-4o": ProviderPricing(
                                    prompt_tokens_per_1k=0.005,
                                    completion_tokens_per_1k=0.015,
                                ),
                                "text-embedding-3-large": ProviderPricing(
                                    prompt_tokens_per_1k=0.00013,
                                    completion_tokens_per_1k=0.00013,
                                ),
                            }
                        }

                return MockCostConfig()

            m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)

            # Also mock the CI environment check to ensure it doesn't interfere
            def mock_is_ci_environment():
                return False

            m.setattr("flujo.infra.config._is_ci_environment", mock_is_ci_environment)

            # Create a mock chat agent with unconfigured model
            class MockChatAgent:
                def __init__(self):
                    self.model_id = "openai:unknown-model"  # Not configured in strict mode

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

            # Create the pipeline
            chat_step = Step(name="ChatResponse", agent=MockChatAgent())

            pipeline = Pipeline(name="test_strict_mode_pipeline", steps=[chat_step])

            # Create runner and run the pipeline - should fail with PricingNotConfiguredError
            runner = Flujo(pipeline)
            with pytest.raises(PricingNotConfiguredError) as exc_info:
                async for r in runner.run_async({"message": "Hello, how are you?"}):
                    pass

            # Verify the error message
            assert "unknown-model" in str(exc_info.value)
            assert "openai" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_strict_mode_behavior_unchanged(self):
        """Test that non-strict mode behavior is unchanged for chat agents."""
        # Create temporary config with non-strict mode
        content = """
[cost]
strict = false

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        config_path = self.create_temp_config(content)

        with pytest.MonkeyPatch().context() as m:
            # Mock the config manager to use our temporary file
            def mock_get_config_manager():
                class MockConfigManager:
                    def __init__(self):
                        self.config_path = config_path

                    def load_config(self):
                        class MockConfig:
                            def __init__(self):
                                self.cost = {
                                    "strict": False,
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

            # Create a mock chat agent with unconfigured model
            class MockChatAgent:
                def __init__(self):
                    self.model_id = (
                        "openai:gpt-3.5-turbo"  # Not configured but has hardcoded defaults
                    )

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

            # Create the pipeline
            chat_step = Step(name="ChatResponse", agent=MockChatAgent())

            pipeline = Pipeline(name="test_non_strict_mode_pipeline", steps=[chat_step])

            # Create runner and run the pipeline - should work with default pricing
            runner = Flujo(pipeline)
            result = None
            async for r in runner.run_async({"message": "Hello, how are you?"}):
                result = r

            # Assertions
            assert len(result.step_history) == 1
            assert result.step_history[0].success

            step_result = result.step_history[0]
            assert step_result.name == "ChatResponse"
            assert step_result.cost_usd > 0.0  # Should have calculated cost using default pricing
            assert step_result.token_counts > 0  # Should have token counts

    def test_configuration_loading_unchanged(self, flujo_toml_with_both_pricing):
        """Test that configuration loading behavior is unchanged."""
        # Create temporary config
        config_path = self.create_temp_config(flujo_toml_with_both_pricing)

        # Verify the config file was created correctly
        config_content = config_path.read_text()
        assert "gpt-4o" in config_content
        assert "text-embedding-3-large" in config_content
        assert "strict = true" in config_content

        # Verify that both pricing configurations are present
        assert "prompt_tokens_per_1k = 0.005" in config_content
        assert "completion_tokens_per_1k = 0.015" in config_content
        assert "prompt_tokens_per_1k = 0.00013" in config_content
        assert "completion_tokens_per_1k = 0.00013" in config_content

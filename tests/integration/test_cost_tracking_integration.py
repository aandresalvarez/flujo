"""Integration tests for cost tracking functionality."""

import pytest
import tempfile
import os
from flujo import Flujo, Step, Pipeline
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError, PricingNotConfiguredError


class MockAgentWithUsage:
    """A mock agent that simulates pydantic-ai usage tracking."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # CRITICAL: Add model_id for proper cost tracking
        self.model_id = "openai:gpt-4o"

    async def run(self, data: str):
        """Simulate a pydantic-ai agent response with usage information."""

        # Create a response object that mimics pydantic-ai's AgentRunResult
        class AgentResponse:
            def __init__(self, output, prompt_tokens, completion_tokens):
                self.output = output
                self._prompt_tokens = prompt_tokens
                self._completion_tokens = completion_tokens

            def usage(self):
                class UsageInfo:
                    def __init__(self, prompt_tokens, completion_tokens):
                        self.request_tokens = prompt_tokens
                        self.response_tokens = completion_tokens

                return UsageInfo(self._prompt_tokens, self._completion_tokens)

        return AgentResponse(f"Response to: {data}", self.prompt_tokens, self.completion_tokens)


@pytest.mark.asyncio
async def test_cost_tracking_in_pipeline():
    """Test that cost tracking works in a real pipeline execution."""
    # Create a mock agent with usage information
    mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

    # Create a simple pipeline
    step = Step(name="test_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create runner with cost limits
    limits = UsageLimits(total_cost_usd_limit=0.1)  # Allow up to $0.10
    runner = Flujo(pipeline, usage_limits=limits)

    # Run the pipeline
    result = None
    async for item in runner.run_async("test input"):
        result = item

    # Verify that the result contains cost information
    assert result is not None
    assert len(result.step_history) == 1

    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.token_counts == 150  # 100 + 50
    assert step_result.cost_usd > 0.0  # Should have calculated cost

    # Verify total cost is tracked
    assert result.total_cost_usd > 0.0


@pytest.mark.asyncio
async def test_cost_limit_enforcement():
    """Test that cost limits are properly enforced."""
    # Create a mock agent that would exceed the cost limit
    mock_agent = MockAgentWithUsage(prompt_tokens=10000, completion_tokens=5000)

    # Create a simple pipeline
    step = Step(name="expensive_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create runner with very low cost limit
    limits = UsageLimits(total_cost_usd_limit=0.001)  # Very low limit
    runner = Flujo(pipeline, usage_limits=limits)

    # Run the pipeline - should raise UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError):
        async for item in runner.run_async("test input"):
            pass


@pytest.mark.asyncio
async def test_token_limit_enforcement():
    """Test that token limits are properly enforced."""
    # Create a mock agent that would exceed the token limit
    mock_agent = MockAgentWithUsage(prompt_tokens=1000, completion_tokens=500)

    # Create a simple pipeline
    step = Step(name="token_heavy_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create runner with token limit
    limits = UsageLimits(total_tokens_limit=100)  # Low token limit
    runner = Flujo(pipeline, usage_limits=limits)

    # Run the pipeline - should raise UsageLimitExceededError
    with pytest.raises(UsageLimitExceededError):
        async for item in runner.run_async("test input"):
            pass


@pytest.mark.asyncio
async def test_cost_tracking_without_config(monkeypatch):
    """Test that cost tracking works even without pricing configuration."""
    # Mock the get_provider_pricing to return None to simulate no config

    def mock_get_provider_pricing(provider, model_name):
        return None

    monkeypatch.setattr("flujo.infra.config.get_provider_pricing", mock_get_provider_pricing)

    # Create a mock agent
    mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

    # Create a simple pipeline
    step = Step(name="test_step", agent=mock_agent)
    pipeline = Pipeline.from_step(step)

    # Create runner without cost limits
    runner = Flujo(pipeline)

    # Run the pipeline
    result = None
    async for item in runner.run_async("test input"):
        result = item

    # Verify that the result contains token information even if cost is 0
    assert result is not None
    assert len(result.step_history) == 1

    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.token_counts == 150  # 100 + 50
    # Cost should be 0.0 when no pricing is configured
    assert step_result.cost_usd == 0.0


class TestStrictPricingModeIntegration:
    """Integration tests for strict pricing mode functionality."""

    @pytest.fixture
    def temp_flujo_toml(self):
        """Create a temporary flujo.toml file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            yield f.name
        # Clean up the file after the test
        try:
            os.unlink(f.name)
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_strict_mode_on_success_case(self, temp_flujo_toml, monkeypatch):
        """Test strict mode on with successful configuration."""
        # Create a flujo.toml with strict mode enabled and pricing configured
        flujo_toml_content = """
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        with open(temp_flujo_toml, "w") as f:
            f.write(flujo_toml_content)

        # Mock the config manager to use our temporary file

        def mock_get_config_manager():
            class MockConfigManager:
                def __init__(self):
                    self.config_path = temp_flujo_toml

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

            mock = MockConfigManager()
            if not hasattr(mock, "get_settings"):
                from flujo.infra.settings import settings as _settings

                mock.get_settings = lambda: _settings  # type: ignore[attr-defined]
            return mock

        monkeypatch.setattr(
            "flujo.infra.config_manager.get_config_manager", mock_get_config_manager
        )

        # Create a mock agent
        mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

        # Create a simple pipeline
        step = Step(name="test_step", agent=mock_agent)
        pipeline = Pipeline.from_step(step)

        # Create runner without cost limits
        runner = Flujo(pipeline)

        # Run the pipeline - should succeed
        result = None
        async for item in runner.run_async("test input"):
            result = item

        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1

        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.token_counts == 150  # 100 + 50
        assert step_result.cost_usd > 0.0  # Should have calculated cost based on TOML file

    @pytest.mark.asyncio
    async def test_strict_mode_on_failure_case(self, temp_flujo_toml, monkeypatch):
        """Test strict mode on with missing pricing configuration."""
        # Create a flujo.toml with strict mode enabled but no pricing for gpt-4o
        flujo_toml_content = """
[cost]
strict = true

[cost.providers.openai.gpt-3.5-turbo]
prompt_tokens_per_1k = 0.0015
completion_tokens_per_1k = 0.002
"""
        with open(temp_flujo_toml, "w") as f:
            f.write(flujo_toml_content)

        # Mock the config manager to use our temporary file

        def mock_get_config_manager():
            class MockConfigManager:
                def __init__(self):
                    self.config_path = temp_flujo_toml

                def load_config(self):
                    class MockConfig:
                        def __init__(self):
                            self.cost = {
                                "strict": True,
                                "providers": {
                                    "openai": {
                                        "gpt-3.5-turbo": {
                                            "prompt_tokens_per_1k": 0.0015,
                                            "completion_tokens_per_1k": 0.002,
                                        }
                                    }
                                },
                            }

                    return MockConfig()

            mock = MockConfigManager()
            if not hasattr(mock, "get_settings"):
                from flujo.infra.settings import settings as _settings

                mock.get_settings = lambda: _settings  # type: ignore[attr-defined]
            return mock

        monkeypatch.setattr(
            "flujo.infra.config_manager.get_config_manager", mock_get_config_manager
        )

        # Create a mock agent that uses gpt-4o (not configured)
        class MockAgentWithGpt4o:
            def __init__(self):
                self.model_id = "openai:gpt-4o"

            async def run(self, data: str):
                class AgentResponse:
                    def __init__(self):
                        self.output = f"Response to: {data}"

                    def usage(self):
                        class UsageInfo:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50

                        return UsageInfo()

                return AgentResponse()

        mock_agent = MockAgentWithGpt4o()

        # Create a simple pipeline
        step = Step(name="test_step", agent=mock_agent)
        pipeline = Pipeline.from_step(step)

        # Create runner without cost limits
        runner = Flujo(pipeline)

        # Run the pipeline - should raise PricingNotConfiguredError
        with pytest.raises(PricingNotConfiguredError) as exc_info:
            async for item in runner.run_async("test input"):
                pass

        # Verify the error message
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4o"
        assert "Strict pricing is enabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_strict_mode_off_fallback_case(self, temp_flujo_toml, monkeypatch):
        """Test strict mode off with fallback to hardcoded defaults."""
        # Create an empty flujo.toml (no strict mode, no pricing)
        flujo_toml_content = """
# Empty configuration - no strict mode, no pricing
"""
        with open(temp_flujo_toml, "w") as f:
            f.write(flujo_toml_content)

        # Mock the config manager to use our temporary file

        def mock_get_config_manager():
            class MockConfigManager:
                def __init__(self):
                    self.config_path = temp_flujo_toml

                def load_config(self):
                    class MockConfig:
                        def __init__(self):
                            self.cost = {}  # Empty cost config

                    return MockConfig()

            mock = MockConfigManager()
            if not hasattr(mock, "get_settings"):
                from flujo.infra.settings import settings as _settings

                mock.get_settings = lambda: _settings  # type: ignore[attr-defined]
            return mock

        monkeypatch.setattr(
            "flujo.infra.config_manager.get_config_manager", mock_get_config_manager
        )

        # Create a mock agent
        mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

        # Create a simple pipeline
        step = Step(name="test_step", agent=mock_agent)
        pipeline = Pipeline.from_step(step)

        # Create runner without cost limits
        runner = Flujo(pipeline)

        # Run the pipeline - should succeed with hardcoded default pricing
        result = None
        async for item in runner.run_async("test input"):
            result = item

        # Verify that the result contains cost information
        assert result is not None
        assert len(result.step_history) == 1

        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.token_counts == 150  # 100 + 50
        assert step_result.cost_usd > 0.0  # Should have calculated cost based on hardcoded defaults

    @pytest.mark.asyncio
    async def test_strict_mode_on_with_unknown_provider(self, temp_flujo_toml, monkeypatch):
        """Test strict mode on with unknown provider."""
        # Create a flujo.toml with strict mode enabled but no pricing for unknown provider
        flujo_toml_content = """
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
"""
        with open(temp_flujo_toml, "w") as f:
            f.write(flujo_toml_content)

        # Mock the config manager to use our temporary file

        def mock_get_config_manager():
            class MockConfigManager:
                def __init__(self):
                    self.config_path = temp_flujo_toml

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

            mock = MockConfigManager()
            if not hasattr(mock, "get_settings"):
                from flujo.infra.settings import settings as _settings

                mock.get_settings = lambda: _settings  # type: ignore[attr-defined]
            return mock

        monkeypatch.setattr(
            "flujo.infra.config_manager.get_config_manager", mock_get_config_manager
        )

        # Create a mock agent that uses an unknown provider
        class MockAgentWithUnknownProvider:
            def __init__(self):
                self.model_id = "unknown:unknown-model"

            async def run(self, data: str):
                class AgentResponse:
                    def __init__(self):
                        self.output = f"Response to: {data}"

                    def usage(self):
                        class UsageInfo:
                            def __init__(self):
                                self.request_tokens = 100
                                self.response_tokens = 50

                        return UsageInfo()

                return AgentResponse()

        mock_agent = MockAgentWithUnknownProvider()

        # Create a simple pipeline
        step = Step(name="test_step", agent=mock_agent)
        pipeline = Pipeline.from_step(step)

        # Create runner without cost limits
        runner = Flujo(pipeline)

        # Run the pipeline - should raise PricingNotConfiguredError
        with pytest.raises(PricingNotConfiguredError) as exc_info:
            async for item in runner.run_async("test input"):
                pass

        # Verify the error message
        assert exc_info.value.provider == "unknown"
        assert exc_info.value.model == "unknown-model"
        assert "Strict pricing is enabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_regression_no_flujo_toml_present(self, monkeypatch):
        """Test regression: pipeline should not crash when no flujo.toml is present."""
        # Mock the config manager to return None config path (no flujo.toml)

        def mock_get_config_manager():
            class MockConfigManager:
                def __init__(self):
                    self.config_path = None  # No flujo.toml present

                def load_config(self):
                    class MockConfig:
                        def __init__(self):
                            pass  # No cost config

                    return MockConfig()

                def get_settings(self):
                    from flujo.infra.settings import settings as _settings

                    return _settings

            mock = MockConfigManager()
            if not hasattr(mock, "get_settings"):
                from flujo.infra.settings import settings as _settings

                mock.get_settings = lambda: _settings  # type: ignore[attr-defined]
            return mock

        monkeypatch.setattr(
            "flujo.infra.config_manager.get_config_manager", mock_get_config_manager
        )

        # Create a mock agent
        mock_agent = MockAgentWithUsage(prompt_tokens=100, completion_tokens=50)

        # Create a simple pipeline
        step = Step(name="test_step", agent=mock_agent)
        pipeline = Pipeline.from_step(step)

        # Create runner without cost limits
        runner = Flujo(pipeline)

        # Run the pipeline - should complete successfully
        result = None
        async for item in runner.run_async("test input"):
            result = item

        # Verify that the pipeline completed successfully
        assert result is not None
        assert len(result.step_history) == 1

        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.token_counts == 150  # 100 + 50
        # Cost should be based on hardcoded defaults (if applicable) or 0.0
        # The key is that the pipeline must not crash
        assert isinstance(step_result.cost_usd, float)

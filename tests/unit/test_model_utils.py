"""Test centralized model ID extraction utilities."""

from unittest.mock import Mock
from flujo.utils.model_utils import extract_model_id, validate_model_id, extract_provider_and_model


class TestModelIDExtraction:
    """Test the extract_model_id function with various agent configurations."""

    def test_extract_model_id_from_model_id_attribute(self):
        """Test extraction from explicit model_id attribute."""
        agent = Mock()
        agent.model_id = "openai:gpt-4o"

        result = extract_model_id(agent, "test_step")
        assert result == "openai:gpt-4o"

    def test_extract_model_id_from_private_model_name(self):
        """Test extraction from private _model_name attribute (backward compatibility)."""
        agent = Mock()
        # Configure the mock to return None for model_id and the actual value for _model_name
        agent.model_id = None
        agent._model_name = "gpt-4o"

        result = extract_model_id(agent, "test_step")
        assert result == "gpt-4o"

    def test_extract_model_id_from_model_attribute(self):
        """Test extraction from model attribute."""
        agent = Mock()
        # Configure the mock to return None for higher priority attributes
        agent.model_id = None
        agent._model_name = None
        agent.model = "claude-3-sonnet"

        result = extract_model_id(agent, "test_step")
        assert result == "claude-3-sonnet"

    def test_extract_model_id_from_model_name_attribute(self):
        """Test extraction from model_name attribute."""
        agent = Mock()
        # Configure the mock to return None for higher priority attributes
        agent.model_id = None
        agent._model_name = None
        agent.model = None
        agent.model_name = "gemini-pro"

        result = extract_model_id(agent, "test_step")
        assert result == "gemini-pro"

    def test_extract_model_id_from_llm_model_attribute(self):
        """Test extraction from llm_model attribute."""
        agent = Mock()
        # Configure the mock to return None for higher priority attributes
        agent.model_id = None
        agent._model_name = None
        agent.model = None
        agent.model_name = None
        agent.llm_model = "llama-3-8b"

        result = extract_model_id(agent, "test_step")
        assert result == "llama-3-8b"

    def test_extract_model_id_priority_order(self):
        """Test that the search order is respected (model_id > _model_name > model > model_name > llm_model)."""
        agent = Mock()
        # Set multiple attributes to test priority
        agent.model_id = "openai:gpt-4o"  # Should be selected
        agent._model_name = "gpt-4o"
        agent.model = "gpt-4o"
        agent.model_name = "gpt-4o"
        agent.llm_model = "gpt-4o"

        result = extract_model_id(agent, "test_step")
        assert result == "openai:gpt-4o"

    def test_extract_model_id_none_value(self):
        """Test handling of None values in attributes."""
        agent = Mock()
        agent.model_id = None
        agent._model_name = None
        agent.model = "gpt-4o"  # Should fall back to this

        result = extract_model_id(agent, "test_step")
        assert result == "gpt-4o"

    def test_extract_model_id_no_attributes(self):
        """Test handling when no model attributes are found."""
        agent = Mock()
        # Don't set any model-related attributes
        # Mock objects by default have all attributes, so we need to explicitly remove them
        del agent.model_id
        del agent._model_name
        del agent.model
        del agent.model_name
        del agent.llm_model

        result = extract_model_id(agent, "test_step")
        assert result is None

    def test_extract_model_id_none_agent(self):
        """Test handling of None agent."""
        result = extract_model_id(None, "test_step")
        assert result is None

    def test_extract_model_id_converts_to_string(self):
        """Test that non-string values are converted to strings."""
        agent = Mock()
        agent.model_id = 12345  # Non-string value

        result = extract_model_id(agent, "test_step")
        assert result == "12345"


class TestModelIDValidation:
    """Test the validate_model_id function."""

    def test_validate_model_id_valid(self):
        """Test validation of valid model IDs."""
        assert validate_model_id("gpt-4o", "test_step") is True
        assert validate_model_id("openai:gpt-4o", "test_step") is True
        assert validate_model_id("claude-3-sonnet", "test_step") is True

    def test_validate_model_id_none(self):
        """Test validation of None model ID."""
        assert validate_model_id(None, "test_step") is False

    def test_validate_model_id_not_string(self):
        """Test validation of non-string model ID."""
        assert validate_model_id(123, "test_step") is False
        assert validate_model_id([], "test_step") is False
        assert validate_model_id({}, "test_step") is False

    def test_validate_model_id_empty_string(self):
        """Test validation of empty or whitespace-only strings."""
        assert validate_model_id("", "test_step") is False
        assert validate_model_id("   ", "test_step") is False
        assert validate_model_id("\t\n", "test_step") is False

    def test_validate_model_id_whitespace_padded(self):
        """Test validation of strings with leading/trailing whitespace."""
        assert validate_model_id("  gpt-4o  ", "test_step") is True  # Should be valid


class TestProviderAndModelExtraction:
    """Test the extract_provider_and_model function."""

    def test_extract_provider_and_model_with_provider(self):
        """Test extraction when provider is specified."""
        provider, model = extract_provider_and_model("openai:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_extract_provider_and_model_without_provider(self):
        """Test extraction when no provider is specified."""
        provider, model = extract_provider_and_model("gpt-4o")
        assert provider is None
        assert model == "gpt-4o"

    def test_extract_provider_and_model_multiple_colons(self):
        """Test extraction with multiple colons (should split on first colon)."""
        provider, model = extract_provider_and_model("openai:gpt-4o:latest")
        assert provider == "openai"
        assert model == "gpt-4o:latest"

    def test_extract_provider_and_model_whitespace_handling(self):
        """Test extraction with whitespace around provider and model."""
        provider, model = extract_provider_and_model("  openai  :  gpt-4o  ")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_extract_provider_and_model_empty_parts(self):
        """Test extraction with empty provider or model parts."""
        provider, model = extract_provider_and_model(":gpt-4o")
        assert provider == ""
        assert model == "gpt-4o"

        provider, model = extract_provider_and_model("openai:")
        assert provider == "openai"
        assert model == ""

    def test_extract_provider_and_model_various_formats(self):
        """Test extraction with various common model formats."""
        test_cases = [
            ("anthropic:claude-3-sonnet", ("anthropic", "claude-3-sonnet")),
            ("google:gemini-pro", ("google", "gemini-pro")),
            ("meta:llama-3-8b", ("meta", "llama-3-8b")),
            ("mistral:mistral-7b", ("mistral", "mistral-7b")),
            ("cohere:command", ("cohere", "command")),
        ]

        for model_id, expected in test_cases:
            provider, model = extract_provider_and_model(model_id)
            assert (provider, model) == expected


class TestIntegrationWithCostTracking:
    """Test integration with cost tracking scenarios."""

    def test_model_id_extraction_for_cost_calculation(self):
        """Test that extracted model IDs work correctly for cost calculation."""

        # Test with provider:model format
        agent = Mock()
        agent.model_id = "openai:gpt-4o"

        model_id = extract_model_id(agent, "test_step")
        assert model_id == "openai:gpt-4o"

        provider, model_name = extract_provider_and_model(model_id)
        assert provider == "openai"
        assert model_name == "gpt-4o"

        # Test cost calculation (this would require proper pricing configuration)
        # Note: This test assumes pricing is configured or uses defaults

    def test_model_id_extraction_for_unknown_models(self):
        """Test handling of unknown model formats."""
        agent = Mock()
        agent.model_id = "unknown:custom-model"

        model_id = extract_model_id(agent, "test_step")
        assert model_id == "unknown:custom-model"

        provider, model_name = extract_provider_and_model(model_id)
        assert provider == "unknown"
        assert model_name == "custom-model"

    def test_model_id_extraction_consistency_across_modules(self):
        """Test that model ID extraction is consistent between cost.py and agents.py."""
        agent = Mock()
        agent.model_id = "anthropic:claude-3-sonnet"

        # Test extraction using our utility
        model_id_utility = extract_model_id(agent, "test_step")

        # Test extraction using the old method (for comparison)
        model_id_old = None
        if hasattr(agent, "model_id"):
            model_id_old = agent.model_id
        elif hasattr(agent, "_model_name"):
            model_id_old = agent._model_name
        elif hasattr(agent, "model"):
            model_id_old = agent.model

        # Both should return the same result
        assert model_id_utility == model_id_old
        assert model_id_utility == "anthropic:claude-3-sonnet"

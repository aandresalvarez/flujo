import pytest
from unittest.mock import patch
from flujo.images.clients.openai_client import OpenAIImageClient
from flujo.exceptions import PricingNotConfiguredError
from flujo.images.models import ImageGenerationResult
from flujo.images import get_image_client


@pytest.fixture
def mock_pricing():
    return {
        "price_per_image_standard_1024x1024": 0.040,
        "price_per_image_standard_1024x1792": 0.080,
        "price_per_image_hd_1024x1024": 0.080,
    }


def test_cost_calculation_logic(mock_pricing):
    client = OpenAIImageClient(pricing_data=mock_pricing, model="dall-e-3")
    with patch("openai.images.generate") as mock_generate:
        mock_generate.return_value = {"data": [{"url": "http://img1"}]}
        result = client.generate("a cat", size="1024x1024", quality="standard")
        assert isinstance(result, ImageGenerationResult)
        assert result.cost_usd == 0.040
        assert result.token_counts == 0
        assert result.image_urls == ["http://img1"]


def test_api_call_formatting(mock_pricing):
    client = OpenAIImageClient(pricing_data=mock_pricing, model="dall-e-3")
    with patch("openai.images.generate") as mock_generate:
        mock_generate.return_value = {"data": [{"url": "http://img2"}]}
        client.generate("a dog", size="1024x1792", quality="standard", foo="bar")
        mock_generate.assert_called_once_with(
            model="dall-e-3", prompt="a dog", size="1024x1792", quality="standard", foo="bar"
        )


def test_strict_mode_price_missing(mock_pricing):
    client = OpenAIImageClient(pricing_data=mock_pricing, model="dall-e-3", strict=True)
    with patch("openai.images.generate"):
        with pytest.raises(PricingNotConfiguredError):
            client.generate("a horse", size="512x512", quality="standard")


def test_get_image_client_invalid_model_id():
    with pytest.raises(ValueError, match="model_id must be in 'provider:model' format"):
        get_image_client("dall-e-3")


def test_pricing_not_configured_error_signature():
    """Test that PricingNotConfiguredError uses the canonical signature."""
    client = OpenAIImageClient(pricing_data={}, model="dall-e-3", strict=True)

    with patch("openai.images.generate"):
        with pytest.raises(PricingNotConfiguredError) as exc_info:
            client.generate("a horse", size="512x512", quality="standard")

        # Verify the exception has the expected attributes from the canonical version
        assert hasattr(exc_info.value, "provider")
        assert hasattr(exc_info.value, "model")
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "dall-e-3"

        # Verify the message format matches the canonical version
        expected_message = "Strict pricing is enabled, but no configuration was found for provider='openai', model='dall-e-3' in flujo.toml."
        assert str(exc_info.value) == expected_message

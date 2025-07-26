import pytest
from unittest.mock import patch
from flujo.images.clients.openai_client import OpenAIImageClient, PricingNotConfiguredError
from flujo.images.models import ImageGenerationResult


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

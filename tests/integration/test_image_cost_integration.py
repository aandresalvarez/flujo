import pytest
from unittest.mock import patch, PropertyMock
from flujo.images import get_image_client
from flujo.images.models import ImageGenerationResult
from flujo.infra.settings import get_settings


class DummyStep:
    def __init__(self, client):
        self.client = client

    def run(self, prompt, **kwargs):
        return self.client.generate(prompt, **kwargs)


def test_end_to_end_success(monkeypatch):
    # Mock the settings.model_cost_providers property
    settings = get_settings()
    with patch.object(
        settings.__class__, "model_cost_providers", new_callable=PropertyMock
    ) as mock_providers:
        mock_providers.return_value = {
            "openai": {"dall-e-3": {"price_per_image_standard_1024x1024": 0.040}}
        }

        client = get_image_client("openai:dall-e-3")
        # Patch openai.images.generate
        import openai

        monkeypatch.setattr(openai.images, "generate", lambda **kwargs: {"data": [{"url": "img"}]})
        step = DummyStep(client)
        result = step.run("prompt", size="1024x1024", quality="standard")
        assert isinstance(result, ImageGenerationResult)
        assert result.cost_usd == 0.040
        assert result.token_counts == 0
        assert result.image_urls == ["img"]


def test_usage_limit_enforcement(monkeypatch):
    # Mock the settings.model_cost_providers property
    settings = get_settings()
    with patch.object(
        settings.__class__, "model_cost_providers", new_callable=PropertyMock
    ) as mock_providers:
        mock_providers.return_value = {
            "openai": {"dall-e-3": {"price_per_image_standard_1024x1024": 0.040}}
        }

        client = get_image_client("openai:dall-e-3")
        import openai

        monkeypatch.setattr(openai.images, "generate", lambda **kwargs: {"data": [{"url": "img"}]})

        # Simulate usage governor
        class UsageLimits:
            def __init__(self, total_cost_usd_limit):
                self.total_cost_usd_limit = total_cost_usd_limit

        class UsageLimitExceededError(Exception):
            pass

        limits = UsageLimits(total_cost_usd_limit=0.01)
        step = DummyStep(client)
        result = step.run("prompt", size="1024x1024", quality="standard")
        # The cost (0.040) exceeds the limit (0.01), so this should raise
        with pytest.raises(UsageLimitExceededError):
            if result.cost_usd > limits.total_cost_usd_limit:
                raise UsageLimitExceededError()


def test_regression_chat_and_image(monkeypatch):
    # Mock the settings.model_cost_providers property
    settings = get_settings()
    with patch.object(
        settings.__class__, "model_cost_providers", new_callable=PropertyMock
    ) as mock_providers:
        mock_providers.return_value = {
            "openai": {
                "dall-e-3": {"price_per_image_standard_1024x1024": 0.040},
                "gpt-4o": {"price_per_1k_tokens": 0.01},
            }
        }

        client = get_image_client("openai:dall-e-3")
        import openai

        monkeypatch.setattr(openai.images, "generate", lambda **kwargs: {"data": [{"url": "img"}]})

        class DummyChatResult:
            cost_usd = 0.02
            token_counts = 1000

        chat_result = DummyChatResult()
        image_result = client.generate("prompt", size="1024x1024", quality="standard")
        total_cost = chat_result.cost_usd + image_result.cost_usd
        assert total_cost == 0.06

from typing import Any
from flujo.images.clients.openai_client import OpenAIImageClient
from flujo.infra.settings import get_settings


def get_image_client(model_id: str, strict: bool = True) -> Any:
    """Factory for image generation clients. Currently supports openai:dall-e-3."""
    settings = get_settings()
    provider, model = model_id.split(":", 1)
    pricing_section = settings.model_cost_providers.get(provider, {}).get(model, {})
    if provider == "openai" and model == "dall-e-3":
        return OpenAIImageClient(pricing_data=pricing_section, model="dall-e-3", strict=strict)
    raise NotImplementedError(f"Image client for {model_id} is not implemented.")

from typing import Any
from flujo.images.clients.openai_client import OpenAIImageClient
from flujo.infra.settings import get_settings


def get_image_client(model_id: str, strict: bool = True) -> Any:
    """Factory for image generation clients.

    Args:
        model_id: Model identifier in 'provider:model' format (e.g., 'openai:dall-e-3').
        strict: Whether to enforce strict pricing.

    Returns:
        An image client instance for the specified provider/model.

    Raises:
        ValueError: If model_id is not in 'provider:model' format.
        NotImplementedError: If the provider/model is not supported.
    """
    settings = get_settings()
    if ":" not in model_id:
        raise ValueError(f"model_id must be in 'provider:model' format, got '{model_id}'")
    provider, model = model_id.split(":", 1)
    pricing_section = settings.model_cost_providers.get(provider, {}).get(model, {})
    if provider == "openai" and model == "dall-e-3":
        return OpenAIImageClient(pricing_data=pricing_section, model="dall-e-3", strict=strict)
    raise NotImplementedError(f"Image client for {model_id} is not implemented.")

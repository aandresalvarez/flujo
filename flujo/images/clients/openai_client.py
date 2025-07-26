from typing import Any, Dict
from flujo.images.models import ImageGenerationResult
from flujo.exceptions import PricingNotConfiguredError


class OpenAIImageClient:
    def __init__(self, pricing_data: Dict[str, float], model: str, strict: bool = True):
        self.pricing_data = pricing_data
        self.model = model
        self.strict = strict

    def _get_price_key(self, size: str, quality: str) -> str:
        # Example: price_per_image_standard_1024x1024
        return f"price_per_image_{quality}_{size}"

    def _get_pricing_for_request(self, size: str, quality: str) -> float:
        price_key = self._get_price_key(size, quality)
        price = self.pricing_data.get(price_key)
        if price is None:
            if self.strict:
                # For image models, we use the model name as both provider and model
                # since the pricing is specific to the image model
                raise PricingNotConfiguredError(
                    provider="openai",  # Assuming OpenAI for image models
                    model=self.model,
                )
            price = 0.0
        return price

    def generate(
        self, prompt: str, size: str = "1024x1024", quality: str = "standard", **kwargs: Any
    ) -> ImageGenerationResult:
        price = self._get_pricing_for_request(size, quality)
        # --- Call OpenAI API ---
        import openai

        response = openai.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            **kwargs,  # type: ignore
        )
        # Handle both dict responses (from mocks) and ImagesResponse objects (from real API)
        if hasattr(response, "data"):
            # Real API response
            image_urls = [item.url for item in response.data]
        else:
            # Mock response (dict)
            image_urls = [item["url"] for item in response["data"]]
        return ImageGenerationResult(image_urls=image_urls, cost_usd=price)

from dataclasses import dataclass
from flujo.cost import ExplicitCostReporter


@dataclass
class ImageGenerationResult(ExplicitCostReporter):
    """Standardized result from an image generation client."""

    image_urls: list[str]
    # --- ExplicitCostReporter Protocol Implementation ---
    cost_usd: float
    token_counts: int = 0  # Image generation has no token cost

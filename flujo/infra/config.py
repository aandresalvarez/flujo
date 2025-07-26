"""Configuration management for cost tracking in flujo."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ..exceptions import PricingNotConfiguredError


class ProviderPricing(BaseModel):
    """Pricing information for a specific provider and model."""

    prompt_tokens_per_1k: float = Field(..., description="Cost per 1K prompt tokens in USD")
    completion_tokens_per_1k: float = Field(..., description="Cost per 1K completion tokens in USD")
    # Image generation pricing (optional)
    price_per_image_standard_1024x1024: Optional[float] = Field(
        None, description="Cost per image for standard quality 1024x1024"
    )
    price_per_image_hd_1024x1024: Optional[float] = Field(
        None, description="Cost per image for HD quality 1024x1024"
    )
    price_per_image_standard_1792x1024: Optional[float] = Field(
        None, description="Cost per image for standard quality 1792x1024"
    )
    price_per_image_hd_1792x1024: Optional[float] = Field(
        None, description="Cost per image for HD quality 1792x1024"
    )
    price_per_image_standard_1024x1792: Optional[float] = Field(
        None, description="Cost per image for standard quality 1024x1792"
    )
    price_per_image_hd_1024x1792: Optional[float] = Field(
        None, description="Cost per image for HD quality 1024x1792"
    )


class CostConfig(BaseModel):
    """Configuration for cost tracking and provider pricing."""

    providers: Dict[str, Dict[str, ProviderPricing]] = Field(
        default_factory=dict,
        description="Provider pricing information organized by provider and model",
    )
    strict: bool = Field(
        default=False,
        description="When enabled, raises PricingNotConfiguredError if pricing is not explicitly configured",
    )


def get_cost_config() -> CostConfig:
    """Get the cost configuration from the current flujo.toml file."""
    from .config_manager import get_config_manager

    config_manager = get_config_manager()
    config = config_manager.load_config()

    # Extract cost configuration from the config
    cost_data = {}
    if hasattr(config, "cost") and config.cost:
        cost_data = config.cost

    return CostConfig(**cost_data)


def get_provider_pricing(provider: Optional[str], model: str) -> Optional[ProviderPricing]:
    """Get pricing information for a specific provider and model."""
    cost_config = get_cost_config()

    # 1. Check for explicit user configuration first.
    if provider in cost_config.providers and model in cost_config.providers[provider]:
        return cost_config.providers[provider][model]

    # 2. If not found, check if strict mode is enabled.
    if cost_config.strict:
        # In CI environments, allow fallback to defaults ONLY when no config file exists
        from .config_manager import get_config_manager

        config_manager = get_config_manager()
        if _is_ci_environment() and _no_config_file_found(config_manager):
            default_pricing = _get_default_pricing(provider, model)
            if default_pricing:
                import logging

                logging.warning(
                    f"Strict pricing enabled in CI but model '{provider}:{model}' not found in config. "
                    f"Using default pricing."
                )
                return default_pricing

        raise PricingNotConfiguredError(provider, model)

    # 3. If not strict, proceed with the existing fallback logic (hardcoded defaults).
    default_pricing = _get_default_pricing(provider, model)
    if default_pricing:
        # Log a critical error when using hardcoded defaults - but only once per model
        from . import telemetry

        # Use a critical error for hardcoded prices to emphasize the risk
        telemetry.logfire.error(
            f"CRITICAL WARNING: Using INACCURATE hardcoded default price for '{provider}:{model}' "
            f"(${default_pricing.prompt_tokens_per_1k}/1K prompt, ${default_pricing.completion_tokens_per_1k}/1K completion). "
            f"These prices may be stale and INACCURATE. Configure explicit pricing in flujo.toml for production use."
        )
        return default_pricing

    # 4. If no explicit or default pricing is found, return None.
    return None


def _get_default_pricing(provider: Optional[str], model: str) -> Optional[ProviderPricing]:
    """Get default pricing for common models when not configured."""

    # If provider is None, we can't provide default pricing
    if provider is None:
        return None

    # OpenAI pricing (as of 2024)
    if provider == "openai":
        if model == "gpt-4o":
            return ProviderPricing(
                prompt_tokens_per_1k=0.005,
                completion_tokens_per_1k=0.015,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "gpt-4o-mini":
            return ProviderPricing(
                prompt_tokens_per_1k=0.00015,
                completion_tokens_per_1k=0.0006,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "gpt-4":
            return ProviderPricing(
                prompt_tokens_per_1k=0.03,
                completion_tokens_per_1k=0.06,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "gpt-3.5-turbo":
            return ProviderPricing(
                prompt_tokens_per_1k=0.0015,
                completion_tokens_per_1k=0.002,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        # OpenAI embedding models
        elif model == "text-embedding-3-large":
            return ProviderPricing(
                prompt_tokens_per_1k=0.00013,
                completion_tokens_per_1k=0.00013,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "text-embedding-3-small":
            return ProviderPricing(
                prompt_tokens_per_1k=0.00002,
                completion_tokens_per_1k=0.00002,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "text-embedding-ada-002":
            return ProviderPricing(
                prompt_tokens_per_1k=0.0001,
                completion_tokens_per_1k=0.0001,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )

    # Anthropic pricing (as of 2024)
    elif provider == "anthropic":
        if model == "claude-3-opus":
            return ProviderPricing(
                prompt_tokens_per_1k=0.015,
                completion_tokens_per_1k=0.075,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "claude-3-sonnet":
            return ProviderPricing(
                prompt_tokens_per_1k=0.003,
                completion_tokens_per_1k=0.015,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )
        elif model == "claude-3-haiku":
            return ProviderPricing(
                prompt_tokens_per_1k=0.00025,
                completion_tokens_per_1k=0.00125,
                price_per_image_standard_1024x1024=None,
                price_per_image_hd_1024x1024=None,
                price_per_image_standard_1792x1024=None,
                price_per_image_hd_1792x1024=None,
                price_per_image_standard_1024x1792=None,
                price_per_image_hd_1024x1792=None,
            )

    return None


def _is_ci_environment() -> bool:
    """Check if we're running in a CI environment."""
    import os

    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _no_config_file_found(config_manager: Optional[Any] = None) -> bool:
    """Check if no configuration file was found."""
    if config_manager is None:
        from .config_manager import get_config_manager

        config_manager = get_config_manager()

    try:
        # If the config manager has no config path, it means no file was found
        if config_manager.config_path is None:
            return True

        # Also check if the config was loaded but is empty (no cost section)
        config = config_manager.load_config()
        if not config.cost:
            return True

        return False
    except Exception:
        # If there's any error getting the config manager, assume no config file
        return True

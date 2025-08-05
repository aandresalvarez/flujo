"""Cost calculation utilities for LLM usage tracking."""

from __future__ import annotations

from typing import (
    Optional,
    Tuple,
    Any,
    Protocol,
    runtime_checkable,
    Dict,
    TypeVar,
    Callable,
)
import flujo.infra.config
from flujo.exceptions import PricingNotConfiguredError

# Cache for model information to reduce repeated extraction overhead
_model_cache: dict[str, tuple[Optional[str], str]] = {}

# Type variable for generic callable resolution
T = TypeVar("T")


def resolve_callable(value: T | Callable[[], T]) -> T:
    """Resolve a value that might be a callable or the value itself.

    This utility function handles the common pattern where an attribute
    might be either a callable that returns a value, or the value itself.
    This reduces code duplication and improves maintainability.

    Args:
        value: Either a callable that returns T, or T directly

    Returns:
        The resolved value of type T
    """
    return value() if callable(value) else value


def clear_cost_cache() -> None:
    """Clear the cost calculation cache. Useful for testing to ensure isolation."""
    global _model_cache
    _model_cache.clear()


@runtime_checkable
class ExplicitCostReporter(Protocol):
    """A protocol for objects that can report their own pre-calculated cost.

    Attributes
    ----------
    cost_usd : float
        The explicit cost in USD for the operation.
    token_counts : int, optional
        The total token count for the operation, if applicable. If not present, will be treated as 0 by extraction logic.
    """

    cost_usd: float
    token_counts: int  # Optional; if missing, treated as 0


def extract_usage_metrics(raw_output: Any, agent: Any, step_name: str) -> Tuple[int, int, float]:
    """
    Extract usage metrics from a pydantic-ai agent response.

    This is a shared helper function to eliminate code duplication in ultra_executor.py.

    Parameters
    ----------
    raw_output : Any
        The raw output from the agent
    agent : Any
        The agent that produced the output
    step_name : str
        Name of the step for logging purposes

    Returns
    -------
    Tuple[int, int, float]
        (prompt_tokens, completion_tokens, cost_usd)
    """
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0

    from .infra import telemetry

    # 1. HIGHEST PRIORITY: Check if the output object reports its own cost.
    # We check for the protocol attributes manually since token_counts is optional
    if hasattr(raw_output, "cost_usd"):
        cost_usd = getattr(raw_output, "cost_usd", 0.0) or 0.0
        # For explicit costs, we don't try to split tokens.
        # We take the total token count if provided, otherwise it's 0.
        total_tokens = getattr(raw_output, "token_counts", 0) or 0
        
        # Handle Mock objects in cost extraction
        if hasattr(cost_usd, '_mock_name'):
            cost_usd = 0.0
        if hasattr(total_tokens, '_mock_name'):
            total_tokens = 0

        telemetry.logfire.info(
            f"Using explicit cost from '{type(raw_output).__name__}' for step '{step_name}': cost=${cost_usd}, tokens={total_tokens}"
        )

        # Return prompt_tokens as 0 since it cannot be determined reliably here.
        return 0, total_tokens, cost_usd

    # 2. Handle string outputs as 1 token (matches original fallback logic behavior)
    if isinstance(raw_output, str):
        telemetry.logfire.info(
            f"Counting string output as 1 token for step '{step_name}': '{raw_output[:50]}{'...' if len(raw_output) > 50 else ''}'"
        )
        return 0, 1, 0.0

    # 3. If explicit metrics are not fully present, proceed with usage() extraction
    if hasattr(raw_output, "usage"):
        try:
            usage_info = raw_output.usage()
            prompt_tokens = getattr(usage_info, "request_tokens", 0) or 0
            completion_tokens = getattr(usage_info, "response_tokens", 0) or 0

            # Check if cost was set by a post-processor (e.g., image cost post-processor)
            usage_cost = getattr(usage_info, "cost_usd", None)
            if usage_cost is not None:
                cost_usd = usage_cost
                telemetry.logfire.info(
                    f"Using cost from usage object for step '{step_name}': cost=${cost_usd}"
                )
                return prompt_tokens, completion_tokens, cost_usd

            # Only log if we have meaningful token counts
            if prompt_tokens > 0 or completion_tokens > 0:
                telemetry.logfire.info(
                    f"Extracted tokens for step '{step_name}': prompt={prompt_tokens}, completion={completion_tokens}"
                )

            # Calculate cost if we have token information
            if prompt_tokens > 0 or completion_tokens > 0:
                # Get the model information from the agent using centralized extraction
                from .utils.model_utils import (
                    extract_model_id,
                    extract_provider_and_model,
                )

                model_id = extract_model_id(agent, step_name)

                if model_id:
                    # Use cached model information to reduce repeated parsing
                    cache_key = f"{agent.__class__.__name__}:{model_id}"
                    if cache_key not in _model_cache:
                        _model_cache[cache_key] = extract_provider_and_model(model_id)

                    provider, model_name = _model_cache[cache_key]

                    cost_calculator = CostCalculator()
                    cost_usd = cost_calculator.calculate(
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        provider=provider,
                    )

                    # Only log if cost is significant
                    if cost_usd > 0.0:
                        telemetry.logfire.info(
                            f"Calculated cost for step '{step_name}': {cost_usd} USD for model {model_name}"
                        )
                else:
                    # FIXED: Return 0.0 cost for agents without model_id instead of guessing OpenAI pricing
                    telemetry.logfire.warning(
                        f"CRITICAL: Could not determine model for step '{step_name}'. "
                        f"Cost will be reported as 0.0. "
                        f"To fix: ensure your agent has a 'model_id' attribute (e.g., 'openai:gpt-4o') "
                        f"or use make_agent_async() with explicit model parameter."
                    )
                    cost_usd = 0.0  # Return 0, which is safer than an incorrect guess.

        except Exception as e:
            # Check if this is a PricingNotConfiguredError that should be re-raised
            if isinstance(e, PricingNotConfiguredError):
                # Re-raise the exception for strict mode failures
                raise
            else:
                # For other exceptions, log a warning and return 0.0
                telemetry.logfire.warning(
                    f"Failed to extract usage metrics for step '{step_name}': {e}"
                )
                cost_usd = 0.0

    return prompt_tokens, completion_tokens, cost_usd


def _validate_usage_object(run_result: Any, telemetry: Any) -> Optional[Any]:
    """Validate and extract the usage object from run_result."""
    if not hasattr(run_result, "usage") or not run_result.usage:
        telemetry.logfire.warning("Image cost post-processor: No usage information found")
        return None

    usage_obj = resolve_callable(run_result.usage)

    if not hasattr(usage_obj, "details") or not usage_obj.details:
        return None

    return usage_obj


def _calculate_image_cost(
    image_count: int,
    pricing_data: Dict[str, Optional[float]],
    price_key: str,
    quality: str,
    size: str,
    telemetry: Any,
) -> float:
    """Calculate the total cost for image generation."""
    price_per_image = pricing_data.get(price_key)

    if price_per_image is None:
        telemetry.logfire.warning(
            f"Image cost post-processor: No pricing found for key '{price_key}'. "
            f"Setting cost to 0.0. Available keys: {list(pricing_data.keys())}"
        )
        return 0.0

    total_cost = image_count * price_per_image
    telemetry.logfire.info(
        f"Image cost post-processor: Calculated cost ${total_cost} "
        f"for {image_count} image(s) at ${price_per_image} each "
        f"(quality: {quality}, size: {size})"
    )
    return total_cost


def _image_cost_post_processor(
    run_result: Any, pricing_data: Dict[str, Optional[float]], **kwargs: Any
) -> Any:
    """
    A pydantic-ai post-processor that calculates and injects image generation cost.

    This function is designed to be attached to a pydantic-ai Agent's post_processors list.
    It receives the AgentRunResult after an API call and calculates the cost based on
    the number of images generated and the pricing configuration.

    Parameters
    ----------
    run_result : Any
        The AgentRunResult from the pydantic-ai agent
    pricing_data : dict
        Dictionary containing pricing information for different image configurations
    **kwargs : Any
        Additional keyword arguments that may contain size and quality information

    Returns
    -------
    Any
        The modified run_result with cost_usd added to the usage object
    """
    from .infra import telemetry

    # Validate and extract the usage object
    usage_obj = _validate_usage_object(run_result, telemetry)
    if not usage_obj:
        return run_result

    # Extract image count
    image_count = usage_obj.details.get("images", 0)
    if image_count == 0:
        return run_result

    # Determine price key from agent call parameters (e.g., size, quality)
    size = kwargs.get("size", "1024x1024")
    quality = kwargs.get("quality", "standard")
    price_key = f"price_per_image_{quality}_{size}"

    # Calculate the cost
    usage_obj.cost_usd = _calculate_image_cost(
        image_count, pricing_data, price_key, quality, size, telemetry
    )

    return run_result


class CostCalculator:
    """Calculates costs for LLM usage based on token counts and model pricing."""

    def __init__(self) -> None:
        """Initialize the cost calculator."""
        pass

    def calculate(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: Optional[str] = None,
    ) -> float:
        """
        Calculate the cost in USD for a given token usage.

        Parameters
        ----------
        model_name : str
            The model name (e.g., "gpt-4o", "claude-3-sonnet")
        prompt_tokens : int
            Number of prompt tokens used
        completion_tokens : int
            Number of completion tokens used
        provider : Optional[str]
            The provider name (e.g., "openai", "anthropic"). If None, will be inferred from model_name.

        Returns
        -------
        float
            The calculated cost in USD

        Raises
        ------
        PricingNotConfiguredError
            When strict pricing mode is enabled but no pricing configuration is found
        """
        # Import telemetry at the start to ensure it's available throughout the method
        from .infra import telemetry

        # If provider is not specified, try to infer it from model_name
        if provider is None:
            provider = self._infer_provider_from_model(model_name)
            if provider is None:
                # CRITICAL: If we cannot infer the provider, log a warning and return 0.0
                # This is safer than guessing and potentially providing incorrect billing estimates
                telemetry.logfire.warning(
                    f"Could not infer provider for '{model_name}'. "
                    f"Cost will be reported as 0.0. "
                    f"To fix: use explicit provider:model format (e.g., 'openai:gpt-4o') "
                    f"or configure pricing in flujo.toml for '{model_name}'."
                )
                return 0.0

        # Get pricing information for this provider and model
        # This may raise PricingNotConfiguredError if strict mode is enabled
        pricing = flujo.infra.config.get_provider_pricing(provider, model_name)

        # Debug logging
        telemetry.logfire.info(
            f"CostCalculator: provider={provider}, model={model_name}, pricing={pricing}"
        )

        if pricing is None:
            # If no pricing is configured, return 0.0 to avoid breaking pipelines
            # This allows pipelines to run even without cost configuration
            telemetry.logfire.warning(
                f"No pricing found for provider={provider}, model={model_name}. "
                f"Cost will be reported as 0.0. "
                f"Configure pricing in flujo.toml for accurate cost tracking."
            )
            return 0.0

        # Calculate costs
        prompt_cost = (prompt_tokens / 1000.0) * pricing.prompt_tokens_per_1k
        completion_cost = (completion_tokens / 1000.0) * pricing.completion_tokens_per_1k

        total_cost = prompt_cost + completion_cost

        telemetry.logfire.info(
            f"Cost calculation: prompt_cost={prompt_cost}, completion_cost={completion_cost}, total={total_cost}"
        )

        return total_cost

    def _infer_provider_from_model(self, model_name: str) -> Optional[str]:
        """
        Infer the provider from the model name.

        Parameters
        ----------
        model_name : str
            The model name (e.g., "gpt-4o", "claude-3-sonnet")

        Returns
        -------
        Optional[str]
            The inferred provider name, or None if cannot be determined
        """
        # Handle None or empty model names
        if not model_name:
            return None

        # Common model name patterns - infer for known models
        if model_name.startswith(("gemini-", "text-bison", "chat-bison")):
            return "google"
        elif model_name.startswith(("gpt-", "dall-e", "text-", "embedding-")):
            return "openai"
        elif model_name.startswith(("claude-", "haiku", "sonnet")):
            return "anthropic"
        elif model_name.startswith("cohere-"):
            return "cohere"
        elif model_name == "llama-2":  # Only the base model name is unambiguous
            return "meta"

        # For ambiguous or unknown models, return None to avoid incorrect inference
        # This includes models like llama-2-7b, mistral-*, etc. that could be hosted by multiple providers
        return None

"""Cost calculation utilities for LLM usage tracking."""

from __future__ import annotations

from typing import Optional, Tuple, Any, Protocol, runtime_checkable
import flujo.infra.config
import os

# Cache for model information to reduce repeated extraction overhead
_model_cache: dict[str, tuple[Optional[str], str]] = {}


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

    This is a shared helper function to eliminate code duplication between
    _run_step_logic.py and ultra_executor.py.

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
    # Strict mode must be checked at runtime to support test monkeypatching
    STRICT_COST_TRACKING = os.environ.get("FLUJO_STRICT_COST_TRACKING", "0") == "1"
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0

    from .infra import telemetry

    # Detect mixed reporting modes: both explicit and usage() present
    has_cost = hasattr(raw_output, "cost_usd")
    has_tokens = hasattr(raw_output, "token_counts")
    has_usage = hasattr(raw_output, "usage")
    if (has_cost or has_tokens) and has_usage:
        msg = f"Mixed reporting modes detected for step '{step_name}': both explicit cost/token attributes and usage() method present. This may cause inconsistent reporting."
        if STRICT_COST_TRACKING:
            telemetry.logfire.error(msg)
            raise ValueError(msg)
        else:
            telemetry.logfire.warning(msg)

    # 1. HIGHEST PRIORITY: Check if the output object reports its own cost.
    if has_cost or has_tokens:
        # Do not fetch values until after strict mode check
        if has_cost and not has_tokens:
            msg = f"Output for step '{step_name}' provides cost_usd but not token_counts. This may cause token usage to be under-reported."
            if STRICT_COST_TRACKING:
                telemetry.logfire.error(msg)
                raise ValueError(msg)
            else:
                telemetry.logfire.warning(msg)
            cost_usd = getattr(raw_output, "cost_usd", 0.0) or 0.0
            total_tokens = 0
        elif has_tokens and not has_cost:
            msg = f"Output for step '{step_name}' provides token_counts but not cost_usd. This may cause cost to be under-reported."
            if STRICT_COST_TRACKING:
                telemetry.logfire.error(msg)
                raise ValueError(msg)
            else:
                telemetry.logfire.warning(msg)
            cost_usd = 0.0
            total_tokens = getattr(raw_output, "token_counts", 0) or 0
        elif not has_cost and not has_tokens:
            # Should not reach here, but fallback
            cost_usd = 0.0
            total_tokens = 0
        else:
            # Both present
            cost_usd = getattr(raw_output, "cost_usd", 0.0) or 0.0
            total_tokens = getattr(raw_output, "token_counts", 0) or 0
        # Validate for negative or implausible values
        if cost_usd < 0 or total_tokens < 0:
            msg = f"Negative values detected in output for step '{step_name}': cost_usd={cost_usd}, token_counts={total_tokens}."
            if STRICT_COST_TRACKING:
                telemetry.logfire.error(msg)
                raise ValueError(msg)
            else:
                telemetry.logfire.warning(msg)
        if cost_usd > 1e6 or total_tokens > 1e9:
            msg = f"Implausibly large values detected in output for step '{step_name}': cost_usd={cost_usd}, token_counts={total_tokens}."
            if STRICT_COST_TRACKING:
                telemetry.logfire.error(msg)
                raise ValueError(msg)
            else:
                telemetry.logfire.warning(msg)
        telemetry.logfire.info(
            f"Using explicit cost from '{type(raw_output).__name__}' for step '{step_name}': cost=${cost_usd}, tokens={total_tokens}"
        )
        return 0, total_tokens, cost_usd
    # 2. If explicit metrics are not fully present, proceed with usage() extraction
    if has_usage:
        try:
            usage_info = raw_output.usage()
            prompt_tokens = getattr(usage_info, "request_tokens", 0) or 0
            completion_tokens = getattr(usage_info, "response_tokens", 0) or 0
            # Mixed reporting mode detection: if both explicit and usage() are present
            if has_cost or has_tokens:
                msg = f"Mixed reporting modes detected for step '{step_name}': both explicit cost/token attributes and usage() method present. This may cause inconsistent reporting."
                if STRICT_COST_TRACKING:
                    telemetry.logfire.error(msg)
                    raise ValueError(msg)
                else:
                    telemetry.logfire.warning(msg)
            # Only log if we have meaningful token counts
            if prompt_tokens > 0 or completion_tokens > 0:
                telemetry.logfire.info(
                    f"Extracted tokens for step '{step_name}': prompt={prompt_tokens}, completion={completion_tokens}"
                )
            # Calculate cost if we have token information
            if prompt_tokens > 0 or completion_tokens > 0:
                from .utils.model_utils import extract_model_id, extract_provider_and_model

                model_id = extract_model_id(agent, step_name)
                if model_id:
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
                    if cost_usd > 0.0:
                        telemetry.logfire.info(
                            f"Calculated cost for step '{step_name}': {cost_usd} USD for model {model_name}"
                        )
                else:
                    msg = (
                        f"CRITICAL: Could not determine model for step '{step_name}'. "
                        f"Cost will be reported as 0.0. "
                        f"To fix: ensure your agent has a 'model_id' attribute (e.g., 'openai:gpt-4o') "
                        f"or use make_agent_async() with explicit model parameter."
                    )
                    if STRICT_COST_TRACKING:
                        telemetry.logfire.error(msg)
                        raise ValueError(msg)
                    else:
                        telemetry.logfire.warning(msg)
                    cost_usd = 0.0
        except Exception as e:
            from .exceptions import PricingNotConfiguredError

            if isinstance(e, PricingNotConfiguredError):
                raise
            msg = f"Could not extract usage metrics for step '{step_name}': {e}"
            if STRICT_COST_TRACKING:
                telemetry.logfire.error(msg)
                raise
            else:
                telemetry.logfire.warning(msg)
    return prompt_tokens, completion_tokens, cost_usd


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

        # Check if this is a text model or image model
        if pricing.is_text_model():
            # Calculate costs for text models using token-based pricing
            # We already checked is_text_model() which ensures both values are not None
            assert pricing.prompt_tokens_per_1k is not None
            assert pricing.completion_tokens_per_1k is not None

            prompt_cost = (prompt_tokens / 1000.0) * pricing.prompt_tokens_per_1k
            completion_cost = (completion_tokens / 1000.0) * pricing.completion_tokens_per_1k
            total_cost = prompt_cost + completion_cost

            telemetry.logfire.info(
                f"Cost calculation (text model): prompt_cost={prompt_cost}, completion_cost={completion_cost}, total={total_cost}"
            )
            return total_cost
        elif pricing.is_image_model():
            # For image models, we can't calculate cost from tokens alone
            # Image cost should be calculated by the image client based on size/quality
            telemetry.logfire.warning(
                f"Attempting to calculate cost for image model '{model_name}' using token-based method. "
                f"This is not supported. Image costs should be calculated by the image client. "
                f"Cost will be reported as 0.0."
            )
            return 0.0
        else:
            # Neither text nor image model pricing found
            telemetry.logfire.warning(
                f"No valid pricing found for provider={provider}, model={model_name}. "
                f"Cost will be reported as 0.0."
            )
            return 0.0

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

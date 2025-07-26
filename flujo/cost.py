"""Cost calculation utilities for LLM usage tracking."""

from __future__ import annotations

from typing import Optional, Tuple, Any, Protocol, runtime_checkable
import flujo.infra.config

# Cache for model information to reduce repeated extraction overhead
_model_cache: dict[str, tuple[Optional[str], str]] = {}


@runtime_checkable
class ExplicitCostReporter(Protocol):
    """A protocol for objects that can report their own pre-calculated cost."""

    cost_usd: float
    token_counts: int = 0  # Defaults to 0 for non-token operations


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
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0

    from .infra import telemetry

    # 1. HIGHEST PRIORITY: Check if the output object reports its own cost.
    if isinstance(raw_output, ExplicitCostReporter):
        cost_usd = getattr(raw_output, "cost_usd", 0.0) or 0.0
        # For explicit costs, we don't try to split tokens.
        # We take the total token count if provided, otherwise it's 0.
        total_tokens = getattr(raw_output, "token_counts", 0) or 0

        telemetry.logfire.info(
            f"Using explicit cost from '{type(raw_output).__name__}' for step '{step_name}': cost=${cost_usd}, tokens={total_tokens}"
        )

        # Return prompt_tokens as 0 since it cannot be determined reliably here.
        return 0, total_tokens, cost_usd

    # 2. Check for explicit cost first - if cost is provided, trust it and don't attempt token calculation
    if hasattr(raw_output, "cost_usd"):
        cost_usd = raw_output.cost_usd or 0.0
        # If token_counts is also present, extract it. Otherwise, default to 0.
        # We cannot reliably split total_tokens, so we return the total as completion_tokens
        # to preserve the token count for usage limits and reporting.
        if hasattr(raw_output, "token_counts"):
            total_tokens = raw_output.token_counts or 0
            # For custom outputs with explicit cost, we cannot reliably split the tokens
            # because prompt and completion tokens have different costs. So we return the
            # total as completion_tokens to preserve the count for usage limits.
            prompt_tokens = 0  # Cannot be determined reliably
            completion_tokens = total_tokens  # Preserve total for usage limits
        else:
            prompt_tokens, completion_tokens = 0, 0

        # Reduced logging frequency for performance
        if cost_usd > 0.0:
            telemetry.logfire.info(
                f"Using explicit cost from custom output for step '{step_name}': cost={cost_usd} USD, total_tokens={completion_tokens}"
            )
        # Return immediately, bypassing all other calculation logic.
        return prompt_tokens, completion_tokens, cost_usd

    # 3. If explicit metrics are not fully present, proceed with usage() extraction
    if hasattr(raw_output, "usage"):
        try:
            usage_info = raw_output.usage()
            prompt_tokens = getattr(usage_info, "request_tokens", 0) or 0
            completion_tokens = getattr(usage_info, "response_tokens", 0) or 0

            # Only log if we have meaningful token counts
            if prompt_tokens > 0 or completion_tokens > 0:
                telemetry.logfire.info(
                    f"Extracted tokens for step '{step_name}': prompt={prompt_tokens}, completion={completion_tokens}"
                )

            # Calculate cost if we have token information
            if prompt_tokens > 0 or completion_tokens > 0:
                # Get the model information from the agent using centralized extraction
                from .utils.model_utils import extract_model_id, extract_provider_and_model

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
            # For PricingNotConfiguredError in strict mode, re-raise it
            from .exceptions import PricingNotConfiguredError

            if isinstance(e, PricingNotConfiguredError):
                raise

            # For other exceptions, log warning but don't crash the pipeline
            telemetry.logfire.warning(
                f"Could not extract usage metrics for step '{step_name}': {e}"
            )

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

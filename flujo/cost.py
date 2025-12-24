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
    overload,
)
from types import FunctionType, BuiltinFunctionType, MethodType
import flujo.infra.config
from flujo.exceptions import PricingNotConfiguredError

# Cache for model information to reduce repeated extraction overhead
_model_cache: dict[str, tuple[Optional[str], str]] = {}

# Type variable for generic callable resolution
T = TypeVar("T")
# Fast-path callable types for micro-optimized resolution in tight loops
_FAST_CALLABLE_TYPES: tuple[type[Any], ...] = (FunctionType, BuiltinFunctionType, MethodType)


@overload
def resolve_callable(value: Callable[[], T]) -> T: ...


@overload
def resolve_callable(value: T) -> T: ...


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
    vt = type(value)
    if vt is FunctionType or vt is BuiltinFunctionType or vt is MethodType:
        callable_value: Callable[[], T] = value  # type: ignore[assignment]
        return callable_value()

    if callable(value):
        callable_value = value
        return callable_value()

    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast to float, returning ``default`` for mocks or invalid values."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely cast to int, returning ``default`` for mocks or invalid values."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


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


@runtime_checkable
class _TiktokenEncoding(Protocol):
    def encode(self, text: str) -> list[int]: ...


@runtime_checkable
class _TiktokenModule(Protocol):
    def get_encoding(self, name: str) -> _TiktokenEncoding: ...


def extract_usage_metrics(raw_output: Any, agent: Any, step_name: str) -> Tuple[int, int, float]:
    """
    Extract usage metrics from an agent response.

    This function supports both FlujoAgentResult (vendor-agnostic) and legacy
    pydantic-ai responses for backward compatibility.

    Parameters
    ----------
    raw_output : Any
        The raw output from the agent (FlujoAgentResult or legacy format)
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
    provider: Optional[str] = None
    model_name: Optional[str] = None

    from .infra import telemetry
    from .domain.agent_result import FlujoAgentResult

    missing_model_warned = False

    # 0. PRIORITY: Check if this is a FlujoAgentResult (vendor-agnostic interface)
    if isinstance(raw_output, FlujoAgentResult):
        # Extract usage from FlujoAgentResult
        usage = raw_output.usage()
        if usage is not None:
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            # Prefer cost from usage, then from result
            cost_usd = getattr(usage, "cost_usd", None) or raw_output.cost_usd or 0.0

            # If we have tokens, we might need to calculate cost
            if prompt_tokens > 0 or completion_tokens > 0:
                # Try to calculate cost if not already set
                if cost_usd == 0.0:
                    # Get model information for cost calculation
                    try:
                        from .utils.model_utils import (
                            extract_model_id,
                            extract_provider_and_model,
                        )

                        model_id = extract_model_id(agent, step_name)
                        if model_id:
                            cache_key = f"{agent.__class__.__name__}:{model_id}"
                            if cache_key not in _model_cache:
                                _model_cache[cache_key] = extract_provider_and_model(model_id)
                            provider, model_name = _model_cache[cache_key]
                            if provider is not None:
                                try:
                                    _ = flujo.infra.config.get_provider_pricing(
                                        provider, model_name
                                    )
                                except PricingNotConfiguredError:
                                    raise
                                cost_calculator = CostCalculator()
                                cost_usd = cost_calculator.calculate(
                                    model_name=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    provider=provider,
                                )
                    except PricingNotConfiguredError:
                        raise
                    except Exception:
                        pass  # Non-fatal, continue with cost_usd = 0.0

                telemetry.logfire.info(
                    f"Extracted usage from FlujoAgentResult for step '{step_name}': "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, cost=${cost_usd}"
                )
                return prompt_tokens, completion_tokens, cost_usd

        # If no usage but explicit cost/tokens, use those
        if raw_output.cost_usd is not None:
            cost_usd = _safe_float(raw_output.cost_usd, default=0.0)
            total_tokens = (
                _safe_int(raw_output.token_counts, default=0) if raw_output.token_counts else 0
            )
            telemetry.logfire.info(
                f"Using explicit cost from FlujoAgentResult for step '{step_name}': "
                f"cost=${cost_usd}, tokens={total_tokens}"
            )
            return 0, total_tokens, cost_usd

        # If we have FlujoAgentResult but no usage/cost info, return zeros
        # (don't fall through to legacy extraction - FlujoAgentResult.output is the actual output, not a response object)
        telemetry.logfire.info(
            f"FlujoAgentResult for step '{step_name}' has no usage/cost info; returning zeros"
        )
        return 0, 0, 0.0

    # 1. HIGHEST PRIORITY: Check if the output object reports its own cost.
    # We check for the protocol attributes manually since token_counts is optional
    if hasattr(raw_output, "cost_usd"):
        # For explicit costs, we trust object's own reporting but guard against mocks
        cost_usd = _safe_float(getattr(raw_output, "cost_usd", 0.0), default=0.0)
        # We take the total token count if provided, otherwise it's 0.
        total_tokens = _safe_int(getattr(raw_output, "token_counts", 0), default=0)

        telemetry.logfire.info(
            f"Using explicit cost from '{type(raw_output).__name__}' for step '{step_name}': cost=${cost_usd}, tokens={total_tokens}"
        )

        # Return prompt_tokens as 0 since it cannot be determined reliably here.
        return 0, total_tokens, cost_usd

    # 2. Handle string outputs: use a stable count of 1 token
    if isinstance(raw_output, str):
        telemetry.logfire.info(f"Counting string output as 1 tokens for step '{step_name}'")
        return 0, 1, 0.0

    # 3. If explicit metrics are not fully present, proceed with usage() extraction
    if hasattr(raw_output, "usage"):
        try:
            # Proactively resolve provider/model from the agent when available and
            # surface strict pricing errors deterministically before doing any
            # cost work. This also ensures tests that patch
            # flujo.infra.config.get_provider_pricing to raise will observe the
            # error from extract_usage_metrics directly.
            try:
                from .utils.model_utils import (
                    extract_model_id as _extract_model_id,
                    extract_provider_and_model as _extract_pm,
                )

                mid = _extract_model_id(agent, step_name)
                # Fallback: if extraction fails but the agent exposes model_id/model, use it directly
                if not mid:
                    direct_mid = getattr(agent, "model_id", None) or getattr(agent, "model", None)
                    if isinstance(direct_mid, str) and direct_mid.strip():
                        mid = str(direct_mid).strip()

                if mid:
                    prov, mname = _extract_pm(mid)
                    provider, model_name = prov, mname
                    if prov is None:
                        # Attempt provider inference for bare model ids to ensure strict pricing surfaces
                        try:
                            prov = CostCalculator()._infer_provider_from_model(mname)
                            provider = prov
                        except Exception:
                            prov = None
                    if prov is not None:
                        # Will raise in strict mode when unconfigured
                        _ = flujo.infra.config.get_provider_pricing(prov, mname)
            except PricingNotConfiguredError:
                # Re-raise to match strict-mode expectations
                raise
            except Exception as e:
                # Non-fatal: continue to usage extraction, but record why
                telemetry.logfire.debug(
                    f"Pricing pre-check skipped for step '{step_name}': {type(e).__name__}: {e}"
                )

            usage_info = raw_output.usage()
            # Guard against mocks and invalid values; prefer the modern fields and
            # fall back only when absent (avoid touching deprecated attributes when possible).
            if hasattr(usage_info, "input_tokens"):
                prompt_tokens = _safe_int(getattr(usage_info, "input_tokens"), default=0)
            elif hasattr(usage_info, "request_tokens"):
                prompt_tokens = _safe_int(getattr(usage_info, "request_tokens"), default=0)
            else:
                prompt_tokens = 0

            if hasattr(usage_info, "output_tokens"):
                completion_tokens = _safe_int(getattr(usage_info, "output_tokens"), default=0)
            elif hasattr(usage_info, "response_tokens"):
                completion_tokens = _safe_int(getattr(usage_info, "response_tokens"), default=0)
            else:
                completion_tokens = 0

            # Check if cost was set by a post-processor (e.g., image cost post-processor)
            usage_cost = getattr(usage_info, "cost_usd", None)
            if usage_cost is not None:
                cost_usd = _safe_float(usage_cost, default=0.0)
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
                    if provider is None:
                        try:
                            inferred_provider = CostCalculator()._infer_provider_from_model(
                                model_name
                            )
                        except Exception:
                            inferred_provider = None
                        if inferred_provider is not None:
                            provider = inferred_provider
                            _model_cache[cache_key] = (provider, model_name)

                    # Pre-check pricing availability only when provider is known
                    # to surface strict-mode errors deterministically while
                    # preserving provider inference when model_id lacks a
                    # provider prefix (e.g., "gpt-4o").
                    if provider is not None:
                        try:
                            _ = flujo.infra.config.get_provider_pricing(provider, model_name)
                        except PricingNotConfiguredError:
                            # Surface strict-mode failure immediately
                            raise

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
                    msg = (
                        f"CRITICAL: Could not determine model for step '{step_name}'. "
                        f"Cost will be reported as 0.0. "
                        f"To fix: ensure your agent has a 'model_id' attribute (e.g., 'openai:gpt-4o') "
                        f"or use make_agent_async() with explicit model parameter."
                    )
                    telemetry.logfire.warning(msg)
                    missing_model_warned = True
                    # Also emit via standard logging to ensure capture in parallel/CI environments
                    try:
                        import logging as _logging

                        _logging.getLogger("flujo").warning(msg)
                    except Exception:
                        pass
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

    # Final safety net: if we had meaningful tokens but could not compute cost and
    # no earlier missing-model warning was emitted, check model_id and emit a
    # critical warning to satisfy strict test expectations and aid diagnostics.
    try:
        if (
            (prompt_tokens > 0 or completion_tokens > 0)
            and cost_usd == 0.0
            and not missing_model_warned
        ):
            from .utils.model_utils import extract_model_id as _extract_model_id

            _mid = _extract_model_id(agent, step_name)
            if not _mid:
                telemetry.logfire.warning(
                    f"CRITICAL: Could not determine model for step '{step_name}'. Cost will be reported as 0.0. "
                    f"To fix: ensure your agent has a 'model_id' attribute (e.g., 'openai:gpt-4o') or use make_agent_async() with explicit model parameter."
                )
    except Exception:
        # Never fail extraction due to diagnostics
        pass

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

        # Honor strict mode deterministically even when callers provide a mocked
        # cost config object. If strict is enabled and there is no explicit
        # pricing for this provider+model, raise before consulting defaults.
        try:
            cfg = flujo.infra.config.get_cost_config()
            is_strict = bool(getattr(cfg, "strict", False))
            providers_map = getattr(cfg, "providers", {}) or {}
            if is_strict:
                has_provider = provider in providers_map
                has_model = (
                    has_provider
                    and isinstance(providers_map.get(provider), dict)
                    and (model_name in providers_map[provider])
                )
                if not has_model:
                    raise PricingNotConfiguredError(provider, model_name)
        except PricingNotConfiguredError:
            # Propagate strict-mode error unmodified
            raise
        except Exception as e:
            from .infra import telemetry

            telemetry.logfire.debug(
                f"Strict-mode pre-check skipped for {provider}:{model_name}: {type(e).__name__}: {e}"
            )

        # Get pricing information for this provider and model
        # This may raise PricingNotConfiguredError if strict mode is enabled
        pricing = flujo.infra.config.get_provider_pricing(provider, model_name)

        # Debug logging
        telemetry.logfire.info(
            f"CostCalculator: provider={provider}, model={model_name}, pricing={pricing}"
        )

        if pricing is None:
            # Non-strict path: treat missing pricing as zero cost and warn.
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

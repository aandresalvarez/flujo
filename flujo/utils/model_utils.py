"""Model-related utility functions for consistent model ID extraction and validation."""

from __future__ import annotations

from typing import Optional
from weakref import WeakKeyDictionary
from ..infra import telemetry

# Cache for model ID extraction to reduce repeated overhead.
_model_id_cache: WeakKeyDictionary[object, Optional[str]] = WeakKeyDictionary()

# Cache for warning flags to avoid duplicate warnings
_warning_cache: dict[str, bool] = {}


def clear_model_id_cache() -> None:
    """Clear the model ID cache. Useful for testing to ensure isolation."""
    global _model_id_cache, _warning_cache
    _model_id_cache.clear()
    _warning_cache.clear()


def extract_model_id(agent: object, step_name: str = "unknown") -> Optional[str]:
    """
    Extract model ID from an agent using a comprehensive search strategy.

    This function implements a robust model ID extraction strategy that searches
    for the model identifier in multiple possible locations on the agent object.
    The search order is optimized for common patterns and provides detailed logging
    for debugging purposes.

    Parameters
    ----------
    agent : object
        The agent object to extract the model ID from
    step_name : str
        Name of the step for logging purposes (default: "unknown")

    Returns
    -------
    Optional[str]
        The extracted model ID, or None if not found

    Examples
    --------
    >>> agent = SomeAgent(model="gpt-4o")
    >>> extract_model_id(agent, "my_step")
    'gpt-4o'

    >>> agent = SomeAgent(model_id="openai:gpt-4o")
    >>> extract_model_id(agent, "my_step")
    'openai:gpt-4o'
    """
    if agent is None:
        telemetry.logfire.warning(f"Agent is None for step '{step_name}'")
        return None

    # Use caching to reduce repeated extraction overhead
    try:
        cached = _model_id_cache.get(agent)
        if cached is not None or agent in _model_id_cache:
            return cached
    except TypeError:
        # Some objects cannot be weak-referenced (and thus cannot be cached safely).
        pass

    # Search order: most specific to least specific
    search_attributes = [
        "model_id",  # Most specific - explicit model ID
        "_model_name",  # Private attribute (backward compatibility)
        "model",  # Common attribute name
        "model_name",  # Alternative common name
        "llm_model",  # Some frameworks use this
    ]

    for attr_name in search_attributes:
        if hasattr(agent, attr_name):
            model_id = getattr(agent, attr_name)
            if model_id is not None:
                # Cache the result (best-effort; skip for non-weakrefable objects)
                try:
                    _model_id_cache[agent] = str(model_id)
                except TypeError:
                    pass

                # Only log for significant model IDs to reduce noise
                if str(model_id).strip():
                    telemetry.logfire.info(
                        f"Extracted model ID for step '{step_name}' from '{attr_name}': {model_id}"
                    )
                return str(model_id)

    # Cache None result to avoid repeated searches
    try:
        _model_id_cache[agent] = None
    except TypeError:
        pass

    # If no model ID found, log a detailed warning with suggestions.
    # In test mode, always emit the warning (tests assert on this). In non-test
    # environments, dedupe by agent class to reduce noise.
    try:
        from ..infra.settings import get_settings as _get_settings

        test_mode = bool(getattr(_get_settings(), "test_mode", False))
    except Exception:
        test_mode = False

    # Suppress warnings for non-LLM/pure-callable agents and core builtins
    try:
        # Heuristic: builtins register as callables or wrappers without model fields
        aid = getattr(agent, "id", None) or getattr(getattr(agent, "agent", None), "id", None)
        if isinstance(aid, str) and aid.startswith("flujo.builtins."):
            return None
        # If the object only exposes a 'run' attribute and no model-ish attributes, treat as non-LLM
        attrs = set(dir(agent))
        modelish = {"model_id", "model", "_model_name", "usage"}
        if (attrs & modelish) == set() and ("run" in attrs or callable(agent)):
            return None
    except Exception:
        pass

    if test_mode:
        telemetry.logfire.warning(
            f"CRITICAL: Could not determine model for step '{step_name}'. "
            f"Agent type: {type(agent).__name__}. "
            f"Available attributes: {[attr for attr in dir(agent) if not attr.startswith('_') or attr in ['_model_name']]}. "
            f"To fix: ensure the agent has a 'model_id', 'model', or '_model_name' attribute, "
            f"or use explicit provider:model format (e.g., 'openai:gpt-4o')."
        )
    else:
        agent_type_key = f"warning:{agent.__class__.__name__}"
        if agent_type_key not in _warning_cache:
            _warning_cache[agent_type_key] = True  # Mark as warned
            telemetry.logfire.warning(
                f"CRITICAL: Could not determine model for step '{step_name}'. "
                f"Agent type: {type(agent).__name__}. "
                f"Available attributes: {[attr for attr in dir(agent) if not attr.startswith('_') or attr in ['_model_name']]}. "
                f"To fix: ensure the agent has a 'model_id', 'model', or '_model_name' attribute, "
                f"or use explicit provider:model format (e.g., 'openai:gpt-4o')."
            )
    return None


def validate_model_id(model_id: Optional[str], step_name: str = "unknown") -> bool:
    """
    Validate that a model ID is properly formatted.

    Parameters
    ----------
    model_id : Optional[str]
        The model ID to validate
    step_name : str
        Name of the step for logging purposes (default: "unknown")

    Returns
    -------
    bool
        True if the model ID is valid, False otherwise
    """
    if model_id is None:
        telemetry.logfire.warning(f"Model ID is None for step '{step_name}'")
        return False

    if not isinstance(model_id, str):
        telemetry.logfire.warning(
            f"Model ID for step '{step_name}' is not a string: {type(model_id)}"
        )
        return False

    if not model_id.strip():
        telemetry.logfire.warning(f"Model ID for step '{step_name}' is empty or whitespace")
        return False

    return True


def extract_provider_and_model(model_id: str) -> tuple[Optional[str], str]:
    """
    Extract provider and model name from a model ID.

    Parameters
    ----------
    model_id : str
        The model ID (e.g., "openai:gpt-4o", "gpt-4o")

    Returns
    -------
    tuple[Optional[str], str]
        (provider, model_name) - provider may be None if not specified

    Examples
    --------
    >>> extract_provider_and_model("openai:gpt-4o")
    ('openai', 'gpt-4o')

    >>> extract_provider_and_model("gpt-4o")
    (None, 'gpt-4o')
    """
    if ":" in model_id:
        provider, model_name = model_id.split(":", 1)
        return provider.strip(), model_name.strip()
    else:
        return None, model_id.strip()

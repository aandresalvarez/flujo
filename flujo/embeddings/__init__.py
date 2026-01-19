"""Embedding functionality for Flujo."""

from __future__ import annotations


from .models import EmbeddingResult
from .clients.openai_client import OpenAIEmbeddingClient


EMBEDDING_DIMENSIONS: dict[str, int] = {
    "openai:text-embedding-3-small": 1536,
    "openai:text-embedding-3-large": 3072,
    "openai:text-embedding-ada-002": 1536,
}


def get_embedding_dimensions(model_id: str) -> int | None:
    """Return the embedding vector length for a known model id."""
    return EMBEDDING_DIMENSIONS.get(model_id)


def get_embedding_client(model_id: str) -> OpenAIEmbeddingClient:
    """
    Get an embedding client for the specified model.

    Parameters
    ----------
    model_id : str
        The model ID in the format "provider:model_name" (e.g., "openai:text-embedding-3-large")

    Returns
    -------
    OpenAIEmbeddingClient
        An embedding client for the specified model.

    Raises
    ------
    ValueError
        If the model_id format is invalid or the provider is not supported.
    """
    if ":" not in model_id:
        raise ValueError("Invalid model_id format. Expected 'provider:model_name'")

    provider, model_name = model_id.split(":", 1)

    if not provider or not model_name:
        raise ValueError("Invalid model_id format. Expected 'provider:model_name'")

    if provider == "openai":
        return OpenAIEmbeddingClient(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    "EmbeddingResult",
    "OpenAIEmbeddingClient",
    "get_embedding_client",
    "get_embedding_dimensions",
]

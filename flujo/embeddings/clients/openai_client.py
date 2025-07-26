"""OpenAI embedding client for Flujo."""

from __future__ import annotations

from typing import List
import openai
from pydantic_ai.usage import Usage

from ..models import EmbeddingResult


class OpenAIEmbeddingClient:
    """
    OpenAI embedding client for generating text embeddings.

    This client handles embedding operations using OpenAI's embedding models
    and returns EmbeddingResult objects that are compatible with Flujo's
    cost tracking system.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the OpenAI embedding client.

        Parameters
        ----------
        model_name : str
            The name of the embedding model (e.g., "text-embedding-3-large")
        """
        self.model_name = model_name
        self.model_id = f"openai:{model_name}"
        self.client = openai.AsyncOpenAI()

    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for the given texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed

        Returns
        -------
        EmbeddingResult
            The embedding results with vectors and usage information

        Raises
        ------
        Exception
            If the embedding API call fails
        """
        # Call the OpenAI embeddings API
        response = await self.client.embeddings.create(model=self.model_name, input=texts)

        # Extract embeddings from the response
        embeddings = [item.embedding for item in response.data]

        # Create usage information from the response
        usage_info = Usage(
            request_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens
        )

        # Return the embedding result
        return EmbeddingResult(embeddings=embeddings, usage_info=usage_info)

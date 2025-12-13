"""OpenAI embedding client for Flujo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import inspect

import openai

from ..models import EmbeddingResult, UsageLike, UsageType, resolve_usage_constructor


@dataclass(slots=True)
class _UsageFallback:
    input_tokens: int | None
    output_tokens: int | None
    request_tokens: int | None
    response_tokens: int | None
    total_tokens: int | None
    requests: int | None


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
        # Lazily construct the OpenAI client to avoid requiring API key at import/initialization time
        # This allows unit tests that only validate formatting to run without network secrets.
        self._client: openai.AsyncOpenAI | None = None
        self._usage_ctor: type[object] = resolve_usage_constructor()

    @staticmethod
    def _get_param_names(usage_ctor: object) -> Mapping[str, inspect.Parameter]:
        try:
            if not callable(usage_ctor):
                return {}
            return inspect.signature(usage_ctor).parameters
        except Exception:
            return {}

    def _build_usage(self, prompt_tokens: int, total_tokens: int) -> UsageType:
        """
        Construct a RunUsage-compatible object across pydantic-ai versions.

        Handles both the modern input/output_tokens signature and the older
        request/response_tokens shape without raising when fields are absent.
        """
        usage_ctor: object = self._usage_ctor
        params = self._get_param_names(usage_ctor)

        input_key = (
            "input_tokens"
            if "input_tokens" in params
            else ("request_tokens" if "request_tokens" in params else None)
        )
        output_key = (
            "output_tokens"
            if "output_tokens" in params
            else ("response_tokens" if "response_tokens" in params else None)
        )
        total_key = "total_tokens" if "total_tokens" in params else None

        usage_kwargs: dict[str, int] = {}
        if input_key:
            usage_kwargs[input_key] = prompt_tokens
        if output_key:
            usage_kwargs[output_key] = 0
        if total_key:
            usage_kwargs[total_key] = total_tokens

        try:
            created: object = usage_ctor(**usage_kwargs) if callable(usage_ctor) else None
            if isinstance(created, UsageLike):
                return created
            return _UsageFallback(
                input_tokens=prompt_tokens,
                output_tokens=0,
                request_tokens=prompt_tokens,
                response_tokens=0,
                total_tokens=total_tokens,
                requests=usage_kwargs.get("requests"),
            )
        except TypeError:
            # Some versions expect a requests field; include it on retry.
            if "requests" in params and "requests" not in usage_kwargs:
                usage_kwargs["requests"] = 1
            try:
                created_retry: object = usage_ctor(**usage_kwargs) if callable(usage_ctor) else None
                if isinstance(created_retry, UsageLike):
                    return created_retry
            except Exception:
                pass

        return _UsageFallback(
            input_tokens=prompt_tokens,
            output_tokens=0,
            request_tokens=prompt_tokens,
            response_tokens=0,
            total_tokens=total_tokens,
            requests=usage_kwargs.get("requests"),
        )

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for the given texts.

        Parameters
        ----------
        texts : list[str]
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
        # Initialize the API client on first use
        if self._client is None:
            self._client = openai.AsyncOpenAI()
        response = await self._client.embeddings.create(model=self.model_name, input=texts)

        # Extract embeddings from the response
        embeddings = [item.embedding for item in response.data]

        prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
        total_tokens = getattr(response.usage, "total_tokens", prompt_tokens)
        usage = self._build_usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

        # Return the embedding result
        return EmbeddingResult(embeddings=embeddings, usage_info=usage)

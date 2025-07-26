"""Models for embedding operations and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic_ai.usage import Usage


@dataclass
class EmbeddingResult:
    """
    Result from an embedding operation.

    This class implements the UsageReportingProtocol by providing a .usage() method
    that returns the usage information from the embedding operation.
    """

    embeddings: List[List[float]]
    """The embedding vectors returned by the embedding model."""

    usage_info: Usage
    """Usage information from the embedding operation."""

    def usage(self) -> Usage:
        """
        Return the usage information from this embedding operation.

        This method implements the UsageReportingProtocol, making EmbeddingResult
        compatible with the existing cost tracking system.

        Returns
        -------
        Usage
            The usage information from the embedding operation.
        """
        return self.usage_info

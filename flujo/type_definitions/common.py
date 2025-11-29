"""Common type aliases for Flujo.

This module provides type aliases for commonly used patterns to improve
type safety and reduce reliance on `Any` and `Dict[str, Any]`.
"""

from typing import Dict, Any, List
from typing_extensions import TypedDict, NotRequired

# JSON structure aliases
# Use JSONObject for truly dynamic JSON data
# Use TypedDict subclasses for known structures
JSONObject = Dict[str, Any]
JSONArray = List[Any]


# Common TypedDict structures
class BlueprintMetadata(TypedDict):
    """Metadata structure for blueprints."""

    name: str
    version: str
    description: NotRequired[str]
    tags: NotRequired[List[str]]


class AgentResponseMetadata(TypedDict):
    """Metadata structure for agent responses."""

    cost_usd: NotRequired[float]
    tokens_used: NotRequired[int]
    model: NotRequired[str]
    timestamp: NotRequired[str]


# Configuration structures
class ExecutorConfig(TypedDict):
    """Configuration for ExecutorCore."""

    cache_size: int
    cache_ttl: int
    concurrency_limit: int
    enable_optimization: NotRequired[bool]
    strict_context_isolation: NotRequired[bool]
    strict_context_merge: NotRequired[bool]


__all__ = [
    "JSONObject",
    "JSONArray",
    "BlueprintMetadata",
    "AgentResponseMetadata",
    "ExecutorConfig",
]

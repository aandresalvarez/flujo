"""Common type aliases for Flujo.

This module provides type aliases for commonly used patterns to improve
type safety and reduce reliance on `Any` and `Dict[str, Any]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
from typing_extensions import NotRequired, TypedDict

# JSON structure aliases
# Define semantic handles for JSON types.
# Note: JSONValue is currently Any to maintain compatibility with Flujo's internal
# use of JSONObject for DSL composition, where live objects (like Pipelines)
# are stored in fields that are eventually serialized to JSON.
JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Any

# Use JSONObject for truly dynamic JSON data
# Use TypedDict subclasses for known structures
JSONObject = Dict[str, JSONValue]
JSONArray = List[JSONValue]


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
    "JSONValue",
    "JSONPrimitive",
    "BlueprintMetadata",
    "AgentResponseMetadata",
    "ExecutorConfig",
]

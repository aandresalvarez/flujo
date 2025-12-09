"""GranularStep DSL for crash-safe, resumable agent execution.

This module implements the Granular Execution Mode per PRD v12, enabling:
- Per-turn persistence with CAS guards
- Fingerprint validation for deterministic resume
- Idempotency key injection for side-effect safety
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar, Dict, List, Optional, TypedDict

from pydantic import ConfigDict, Field

from .step import Step

__all__ = ["GranularStep", "GranularState", "ResumeError"]


class GranularState(TypedDict):
    """State schema for granular execution, persisted in scratchpad.

    Attributes:
        turn_index: Committed turn count (0 = start, incremented after each turn)
        history: PydanticAI-serialized message history
        is_complete: Whether the agent has finished execution
        final_output: The final output when is_complete is True
        fingerprint: SHA-256 of canonical run-shaping config
    """

    turn_index: int
    history: List[Dict[str, Any]]
    is_complete: bool
    final_output: Any
    fingerprint: str


class ResumeError(Exception):
    """Raised when resumption fails due to state inconsistency.

    Attributes:
        irrecoverable: If True, the run cannot be resumed with current config
        message: Human-readable explanation
    """

    def __init__(self, message: str, *, irrecoverable: bool = False) -> None:
        super().__init__(message)
        self.irrecoverable = irrecoverable
        self.message = message


class GranularStep(Step[Any, Any]):
    """Execute an agent one turn at a time with crash-safe persistence.

    Each turn is persisted atomically with CAS guards to prevent double-execution.
    The step validates fingerprints on resume to ensure deterministic replay.

    Attributes:
        history_max_tokens: Maximum token budget for history (default 128K)
        blob_threshold_bytes: Payload size triggering blob offload (default 20KB)
        enforce_idempotency: Require idempotency keys on tool calls
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    # Granular-specific fields
    history_max_tokens: int = Field(
        default=128_000,
        description="Maximum token budget for message history before truncation",
    )
    blob_threshold_bytes: int = Field(
        default=20_000,
        description="Payload size in bytes that triggers blob offloading",
    )
    enforce_idempotency: bool = Field(
        default=False,
        description="Require idempotency keys on all tool calls",
    )

    # Override meta to route to granular policy
    meta: Dict[str, Any] = Field(
        default_factory=lambda: {"policy": "granular_agent"},
        description="Metadata for policy routing",
    )

    def model_post_init(self, __context: Any) -> None:
        """Ensure policy routing is set."""
        super().model_post_init(__context)
        # Guarantee policy routing even if meta was overridden
        if not isinstance(self.meta, dict):
            object.__setattr__(self, "meta", {"policy": "granular_agent"})
        elif "policy" not in self.meta:
            meta_copy = dict(self.meta)
            meta_copy["policy"] = "granular_agent"
            object.__setattr__(self, "meta", meta_copy)

    @staticmethod
    def compute_fingerprint(
        *,
        input_data: Any,
        system_prompt: Optional[str],
        model_id: str,
        provider: Optional[str],
        tools: List[Dict[str, Any]],
        settings: Dict[str, Any],
    ) -> str:
        """Compute deterministic fingerprint for run-shaping config.

        Returns a SHA-256 hash of canonical JSON representation.
        """
        # Normalize tools to sorted name + signature hash
        normalized_tools = []
        for tool in sorted(tools, key=lambda t: t.get("name", "")):
            tool_repr = {
                "name": tool.get("name", ""),
                "sig_hash": tool.get("sig_hash", ""),
            }
            normalized_tools.append(tool_repr)

        config = {
            "input_data": _sort_keys_recursive(input_data)
            if isinstance(input_data, dict)
            else input_data,
            "system_prompt": system_prompt,
            "model_id": model_id,
            "provider": provider,
            "tools": normalized_tools,
            "settings": dict(sorted(settings.items())) if settings else {},
        }

        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_idempotency_key(run_id: str, step_name: str, turn_index: int) -> str:
        """Generate a deterministic idempotency key for a specific turn.

        Returns a SHA-256 hash of the composite key.
        """
        composite = f"{run_id}:{step_name}:{turn_index}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    @property
    def is_complex(self) -> bool:
        """Granular steps are complex due to internal state management."""
        return True


def _sort_keys_recursive(obj: Any) -> Any:
    """Recursively sort dictionary keys for canonical representation."""
    if isinstance(obj, dict):
        return {k: _sort_keys_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_keys_recursive(item) for item in obj]
    return obj

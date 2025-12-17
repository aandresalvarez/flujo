"""GranularStep DSL for crash-safe, resumable agent execution.

This module implements the Granular Execution Mode per PRD v12, enabling:
- Per-turn persistence with CAS guards
- Fingerprint validation for deterministic resume
- Idempotency key injection for side-effect safety
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TypedDict

from pydantic import Field

from .step import Step

__all__ = ["GranularStep", "GranularState", "ResumeError"]


class GranularState(TypedDict):
    """State schema for granular execution, persisted in context.granular_state.

    Attributes:
        turn_index: Committed turn count (0 = start, incremented after each turn)
        history: PydanticAI-serialized message history
        is_complete: Whether the agent has finished execution
        final_output: The final output when is_complete is True
        fingerprint: SHA-256 of canonical run-shaping config
    """

    turn_index: int
    history: list[dict[str, object]]
    is_complete: bool
    final_output: object
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


class GranularStep(Step[object, object]):
    """Execute an agent one turn at a time with crash-safe persistence.

    Each turn is persisted atomically with CAS guards to prevent double-execution.
    The step validates fingerprints on resume to ensure deterministic replay.

    Attributes:
        history_max_tokens: Maximum token budget for history (default 128K)
        blob_threshold_bytes: Payload size triggering blob offload (default 20KB)
        enforce_idempotency: Require idempotency keys on tool calls
    """

    # model_config inherited from BaseModel

    @staticmethod
    def _default_meta() -> dict[str, object]:
        return {"policy": "granular_agent"}

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
    meta: dict[str, object] = Field(
        default_factory=_default_meta,
        description="Metadata for policy routing",
    )

    def model_post_init(self, __context: object) -> None:
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
        input_data: object,
        system_prompt: str | None,
        model_id: str,
        provider: str | None,
        tools: list[dict[str, object]],
        settings: Mapping[str, object],
    ) -> str:
        """Compute deterministic fingerprint for run-shaping config.

        Returns a SHA-256 hash of canonical JSON representation.
        """

        def _tool_sort_key(tool: dict[str, object]) -> str:
            name = tool.get("name")
            return name if isinstance(name, str) else ""

        # Normalize tools to sorted name + signature hash
        normalized_tools = []
        for tool in sorted(tools, key=_tool_sort_key):
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


def _sort_keys_recursive(obj: object) -> object:
    """Recursively sort dictionary keys for canonical representation."""
    if isinstance(obj, dict):
        return {k: _sort_keys_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_keys_recursive(item) for item in obj]
    return obj

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, MutableMapping


_ACCUMULATING_FIELDS = {
    "support",
    "contradict",
    "verify_hits",
    "cost_tokens",
    "tool_cost_usd",
}


@dataclass
class Badges:
    """Light-weight evidence container used by experimental score-free pipelines."""

    constraints_ok: bool = True
    support: int = 0
    contradict: int = 0
    quality_avg: float = 0.5
    recency: float = 0.5
    directness: float = 0.5
    verify_hits: int = 0
    cost_tokens: int = 0
    tool_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    summary = property(to_dict)

    def merged(self, other: Mapping[str, Any]) -> "Badges":
        merged_badges = replace(self)
        for key, value in other.items():
            if not hasattr(merged_badges, key):
                continue
            setattr(merged_badges, key, value)
        return merged_badges


def copy_badges(source: Badges | Mapping[str, Any] | None) -> Badges:
    if isinstance(source, Badges):
        return replace(source)
    if isinstance(source, Mapping):
        known: Dict[str, Any] = {}
        for key in Badges().__dict__.keys():
            if key in source:
                known[key] = source[key]
        return Badges(**known)
    return Badges()


def update_badges(badges: Badges, deltas: Mapping[str, Any]) -> Badges:
    updated = replace(badges)
    for key, value in deltas.items():
        if not hasattr(updated, key):
            continue
        current = getattr(updated, key)
        if isinstance(current, bool):
            setattr(updated, key, bool(value))
            continue
        if (
            isinstance(current, (int, float))
            and isinstance(value, (int, float))
            and key in _ACCUMULATING_FIELDS
        ):
            setattr(updated, key, current + value)
            continue
        setattr(updated, key, value)
    return updated


def badge_diff(base: Badges, other: Badges) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in base.__dict__.keys():
        left = getattr(base, key)
        right = getattr(other, key)
        if left != right:
            diff[key] = right
    return diff


def ensure_badges(container: MutableMapping[str, Any], key: str = "badges") -> Badges:
    existing = container.get(key)
    badges = copy_badges(existing)
    container[key] = badges
    return badges

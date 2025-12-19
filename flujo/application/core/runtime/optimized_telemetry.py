from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from threading import Lock

from .telemetry_handler import TelemetryHandler

__all__ = [
    "TelemetryHandler",
    "increment_counter",
    "snapshot_counters",
]

_COUNTERS: Counter[str] = Counter()
_COUNTERS_LOCK = Lock()


def _counter_key(name: str, tags: Mapping[str, str] | None) -> str:
    if not tags:
        return name
    try:
        encoded = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
    except Exception:
        encoded = "<unserializable>"
    return f"{name}|{encoded}"


def increment_counter(name: str, value: int = 1, *, tags: Mapping[str, str] | None = None) -> None:
    if value == 0:
        return
    key = _counter_key(name, tags)
    with _COUNTERS_LOCK:
        _COUNTERS[key] += value


def snapshot_counters() -> Mapping[str, int]:
    with _COUNTERS_LOCK:
        return dict(_COUNTERS)

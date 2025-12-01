"""Default cache-related components split from default_components."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from ...domain.models import StepResult, UsageLimits


class OrjsonSerializer:
    """Fast JSON serializer using orjson if available, unified with flujo.utils.serialization."""

    def __init__(self) -> None:
        try:
            import orjson

            self._orjson = orjson
            self._use_orjson = True
        except ImportError:
            import json

            self._json = json
            self._use_orjson = False

    def serialize(self, obj: Any) -> bytes:
        from flujo.utils.serialization import safe_serialize

        serialized_obj = safe_serialize(obj, mode="default")
        if self._use_orjson:
            blob: bytes = self._orjson.dumps(serialized_obj, option=self._orjson.OPT_SORT_KEYS)
            return blob
        s = self._json.dumps(serialized_obj, sort_keys=True, separators=(",", ":"))
        return s.encode("utf-8")

    def deserialize(self, blob: bytes) -> Any:
        from flujo.utils.serialization import safe_deserialize

        if self._use_orjson:
            raw_data = self._orjson.loads(blob)
        else:
            raw_data = self._json.loads(blob.decode("utf-8"))
        return safe_deserialize(raw_data)


class Blake3Hasher:
    """Fast cryptographic hasher using Blake3 if available."""

    def __init__(self) -> None:
        try:
            import blake3

            self._blake3 = blake3
            self._use_blake3 = True
        except ImportError:
            self._use_blake3 = False

    def digest(self, data: bytes) -> str:
        if self._use_blake3:
            return str(self._blake3.blake3(data).hexdigest())
        return hashlib.blake2b(data, digest_size=32).hexdigest()


class _HasherProtocol(Protocol):
    def digest(self, data: bytes) -> str: ...


class DefaultCacheKeyGenerator:
    """Default cache key generator implementation."""

    def __init__(self, hasher: _HasherProtocol | None = None):
        self._hasher: _HasherProtocol = hasher or Blake3Hasher()

    def _get_agent(self, step: Any) -> Any:
        return getattr(step, "agent", None)

    def _get_agent_type(self, agent: Any) -> Optional[str]:
        if agent is None:
            return None
        try:
            return type(agent).__name__
        except Exception:
            return None

    def _get_agent_model_id(self, agent: Any) -> Optional[str]:
        if agent is None:
            return None
        try:
            return getattr(agent, "model_id", None)
        except Exception:
            return None

    def _get_agent_system_prompt(self, agent: Any) -> Optional[Any]:
        if agent is None:
            return None
        try:
            system_prompt = getattr(agent, "system_prompt", None)
            if system_prompt is None and hasattr(agent, "_system_prompt"):
                system_prompt = getattr(agent, "_system_prompt")
            return system_prompt
        except Exception:
            return None

    def _hash_text_sha256(self, text: Any) -> Optional[str]:
        try:
            import hashlib as _hashlib

            if text is None:
                return None
            return _hashlib.sha256(str(text).encode()).hexdigest()
        except Exception:
            return None

    def _get_processor_names(self, step: Any, attribute_name: str) -> List[str]:
        try:
            processors_obj = getattr(step, "processors", None)
            if processors_obj is None:
                return []
            candidate_list = getattr(processors_obj, attribute_name, [])
            if not isinstance(candidate_list, list):
                return []
            return [type(p).__name__ for p in candidate_list]
        except Exception:
            return []

    def _get_validator_names(self, step: Any) -> List[str]:
        try:
            validators = getattr(step, "validators", [])
            if not isinstance(validators, list):
                return []
            return [type(v).__name__ for v in validators]
        except Exception:
            return []

    def _build_step_section(self, step: Any) -> dict[str, Any]:
        agent = self._get_agent(step)
        agent_type = self._get_agent_type(agent)
        model_id = self._get_agent_model_id(agent)
        system_prompt = self._get_agent_system_prompt(agent)
        system_prompt_hash = self._hash_text_sha256(system_prompt)

        return {
            "name": getattr(step, "name", str(type(step).__name__)),
            "agent": {
                "type": agent_type,
                "model_id": model_id,
                "system_prompt_sha256": system_prompt_hash,
            },
            "config": {
                "max_retries": getattr(getattr(step, "config", None), "max_retries", None),
                "timeout_s": getattr(getattr(step, "config", None), "timeout_s", None),
                "temperature": getattr(getattr(step, "config", None), "temperature", None),
            },
            "processors": {
                "prompt_processors": self._get_processor_names(step, "prompt_processors"),
                "output_processors": self._get_processor_names(step, "output_processors"),
            },
            "validators": self._get_validator_names(step),
        }

    def _build_payload(self, step: Any, data: Any, context: Any, resources: Any) -> dict[str, Any]:
        from flujo.utils.serialization import safe_serialize as _safe_serialize

        return {
            "step": self._build_step_section(step),
            "data": _safe_serialize(data, mode="cache"),
            "context": _safe_serialize(context, mode="cache"),
            "resources": _safe_serialize(resources, mode="cache"),
        }

    def generate_key(self, step: Any, data: Any, context: Any, resources: Any) -> str:
        try:
            from flujo.domain.dsl.step import Step as _DSLStep

            if isinstance(step, _DSLStep):
                from flujo.steps.cache_step import _generate_cache_key as _gen_alt

                key: str | None = _gen_alt(step, data, context, resources)
                if key is not None:
                    return key
        except Exception:
            pass

        import hashlib as _hashlib
        import json as _json

        payload = self._build_payload(step, data, context, resources)
        try:
            blob = _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            return _hashlib.sha256(blob).hexdigest()
        except Exception:
            step_name = getattr(step, "name", str(type(step).__name__))
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                data_repr = _safe_serialize(data, mode="cache")
            except Exception:
                data_repr = str(data)
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                ctx_repr = _safe_serialize(context, mode="cache")
            except Exception:
                ctx_repr = str(context)
            try:
                from flujo.utils.serialization import safe_serialize as _safe_serialize

                res_repr = _safe_serialize(resources, mode="cache")
            except Exception:
                res_repr = str(resources)
            seed = _json.dumps(
                {
                    "name": step_name,
                    "data": data_repr,
                    "context": ctx_repr,
                    "resources": res_repr,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            return _hashlib.sha256(seed).hexdigest()


@dataclass
class _LRUCache:
    """LRU cache implementation with TTL support."""

    max_size: int = 1024
    ttl: int = 3600
    _store: OrderedDict[str, tuple[StepResult, float]] = field(
        init=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl < 0:
            raise ValueError("ttl must be non-negative")

    def set(self, key: str, value: StepResult) -> None:
        current_time = time.monotonic()
        while len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = (value, current_time)
        self._store.move_to_end(key)

    def get(self, key: str) -> Optional[StepResult]:
        if key not in self._store:
            return None
        value, timestamp = self._store[key]
        current_time = time.monotonic()
        if self.ttl > 0 and current_time - timestamp > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def clear(self) -> None:
        self._store.clear()


@dataclass
class InMemoryLRUBackend:
    """O(1) LRU cache with TTL support, async interface."""

    max_size: int = 1024
    ttl_s: int = 3600
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _store: OrderedDict[str, tuple[StepResult, float, int]] = field(
        init=False, default_factory=OrderedDict
    )

    async def get(self, key: str) -> Optional[StepResult]:
        async with self._lock:
            if key not in self._store:
                return None
            result, timestamp, access_count = self._store[key]
            current_time = time.monotonic()
            if current_time - timestamp > self.ttl_s:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            self._store[key] = (result, timestamp, access_count + 1)
            return result.model_copy(deep=True)

    async def put(self, key: str, value: StepResult, ttl_s: int) -> None:
        async with self._lock:
            current_time = time.monotonic()
            while len(self._store) >= self.max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, current_time, 0)
            self._store.move_to_end(key)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


@dataclass
class ThreadSafeMeter:
    """Thread-safe usage meter with atomic operations."""

    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_cost_usd += cost_usd
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None) -> None:
        return None

    async def snapshot(self) -> tuple[float, int, int]:
        return self.total_cost_usd, self.prompt_tokens, self.completion_tokens

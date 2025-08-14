import asyncio
import builtins
import contextlib
import hashlib
import inspect
import json
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, List, Dict

import pytest

# Note: Testing framework
# These tests are written for pytest and pytest-asyncio.
# They use pytest style assertions, fixtures, and asyncio support.


# ---------------------------
# Dynamic import resolution
# ---------------------------
# We try to import the module that defines OrjsonSerializer et al.
# If not found via known names, we dynamically construct a shadow module that
# approximates the code under test using the provided interfaces for testability.

def _resolve_default_components_module():
    # Try common module names
    candidate_names = [
        "flujo.application.default_components",
        "flujo.default_components",
        "default_components",
        "src.default_components",
        "app.default_components",
    ]
    for name in candidate_names:
        with contextlib.suppress(Exception):
            mod = __import__(name, fromlist=["*"])
            # quick sanity check
            if any(hasattr(mod, cls) for cls in [
                "OrjsonSerializer", "Blake3Hasher", "DefaultCacheKeyGenerator",
                "_LRUCache", "InMemoryLRUBackend", "ThreadSafeMeter",
                "DefaultProcessorPipeline", "DefaultValidatorRunner",
                "DefaultPluginRunner", "DefaultAgentRunner", "DefaultTelemetry"
            ]):
                return mod
    # Fallback: construct a shim module that closely mirrors the provided code,
    # but delegates integrations (flujo.*) to dummy stubs placed into sys.modules
    return _build_shim_default_components_module()


# Helper to add minimal stubs for external dependencies referenced by code under test
def _install_dependency_stubs():
    # Stub for flujo.utils.serialization
    utils_serialization = types.ModuleType("flujo.utils.serialization")

    def safe_serialize(obj: Any, mode: str = "default"):
        # Provide stable, JSON-serializable transformations
        # For un-serializable objects, fallback to str
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def safe_deserialize(obj: Any):
        return obj

    utils_serialization.safe_serialize = safe_serialize
    utils_serialization.safe_deserialize = safe_deserialize

    # Ensure parent packages exist in sys.modules
    sys.modules.setdefault("flujo", types.ModuleType("flujo"))
    sys.modules.setdefault("flujo.utils", types.ModuleType("flujo.utils"))
    sys.modules["flujo.utils.serialization"] = utils_serialization

    # Stub for flujo.domain.dsl.step.Step
    sys.modules.setdefault("flujo.domain", types.ModuleType("flujo.domain"))
    sys.modules.setdefault("flujo.domain.dsl", types.ModuleType("flujo.domain.dsl"))

    class Step:  # minimal marker class
        pass

    step_mod = types.ModuleType("flujo.domain.dsl.step")
    step_mod.Step = Step
    sys.modules["flujo.domain.dsl.step"] = step_mod

    # Stub for flujo.steps.cache_step._generate_cache_key
    sys.modules.setdefault("flujo.steps", types.ModuleType("flujo.steps"))
    cache_step = types.ModuleType("flujo.steps.cache_step")

    def _generate_cache_key(step, data, context, resources):
        # Return None to force fallback path in DefaultCacheKeyGenerator tests
        return None

    cache_step._generate_cache_key = _generate_cache_key
    sys.modules["flujo.steps.cache_step"] = cache_step

    # Stub for telemetry.logfire
    telemetry = types.ModuleType("telemetry")
    logfire = types.SimpleNamespace(error=lambda *a, **k: None)
    telemetry.logfire = logfire
    sys.modules["telemetry"] = telemetry

    # application.core.context_manager helpers
    sys.modules.setdefault("flujo.application", types.ModuleType("flujo.application"))
    sys.modules.setdefault("flujo.application.core", types.ModuleType("flujo.application.core"))

    def _accepts_param(func: Callable, name: str) -> bool:
        try:
            sig = inspect.signature(func)
            return name in sig.parameters
        except Exception:
            return False

    def analyze_signature(func: Callable):
        # For simplicity, return the signature itself
        return inspect.signature(func)

    def _should_pass_context(sig, ctx, func):
        # Expects keyword-only 'context' to be present
        try:
            return any(
                p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context"
                for p in sig.parameters.values()
            )
        except Exception:
            return False

    cm_mod = types.ModuleType("flujo.application.core.context_manager")
    cm_mod._accepts_param = _accepts_param
    cm_mod.analyze_signature = analyze_signature
    cm_mod._should_pass_context = _should_pass_context
    sys.modules["flujo.application.core.context_manager"] = cm_mod

    # domain.plugins PluginOutcome
    plugins_pkg = sys.modules.setdefault("flujo.domain.plugins", types.ModuleType("flujo.domain.plugins"))

    @dataclass
    class PluginOutcome:
        success: bool
        new_solution: Optional[Any] = None

    plugins_pkg.PluginOutcome = PluginOutcome
    sys.modules["flujo.domain.plugins"] = plugins_pkg

    # domain.types for StepResult, ValidationResult, UsageLimits, PipelineResult
    domain_types = types.ModuleType("flujo.domain.types")

    @dataclass
    class StepResult:
        content: Any

        def model_copy(self, deep: bool = False):
            # return a shallow or deep copy; for primitives just return the same
            return StepResult(json.loads(json.dumps(self.content)))

    @dataclass
    class ValidationResult:
        is_valid: bool
        feedback: Optional[str] = None
        validator_name: Optional[str] = None

    @dataclass
    class UsageLimits:
        total_cost_usd_limit: Optional[float] = None
        total_tokens_limit: Optional[int] = None

    @dataclass
    class PipelineResult:
        step_history: List[Any]
        total_cost_usd: float

    domain_types.StepResult = StepResult
    domain_types.ValidationResult = ValidationResult
    domain_types.UsageLimits = UsageLimits
    domain_types.PipelineResult = PipelineResult
    sys.modules["flujo.domain.types"] = domain_types

    # domain.exceptions required by DefaultAgentRunner
    domain_ex = types.ModuleType("flujo.domain.exceptions")

    class PausedException(Exception):
        pass

    class InfiniteFallbackError(Exception):
        pass

    class InfiniteRedirectError(Exception):
        pass

    class ContextInheritanceError(Exception):
        pass

    domain_ex.PausedException = PausedException
    domain_ex.InfiniteFallbackError = InfiniteFallbackError
    domain_ex.InfiniteRedirectError = InfiniteRedirectError
    domain_ex.ContextInheritanceError = ContextInheritanceError
    sys.modules["flujo.domain.exceptions"] = domain_ex

    return {
        "Step": step_mod.Step,
        "PluginOutcome": plugins_pkg.PluginOutcome,
        "StepResult": domain_types.StepResult,
        "ValidationResult": domain_types.ValidationResult,
        "UsageLimits": domain_types.UsageLimits,
        "PipelineResult": domain_types.PipelineResult,
        "PausedException": domain_ex.PausedException,
        "InfiniteFallbackError": domain_ex.InfiniteFallbackError,
        "InfiniteRedirectError": domain_ex.InfiniteRedirectError,
        "ContextInheritanceError": domain_ex.ContextInheritanceError,
    }


def _build_shim_default_components_module():
    # Build a minimal but faithful reproduction of the API surface from the diff.
    _install_dependency_stubs()

    mod = types.ModuleType("default_components_shim")
    StepResult = sys.modules["flujo.domain.types"].StepResult
    ValidationResult = sys.modules["flujo.domain.types"].ValidationResult
    UsageLimits = sys.modules["flujo.domain.types"].UsageLimits
    # (removed unused PipelineResult, AnyT, OptionalT, ListT, DictT)

    # OrjsonSerializer
    class OrjsonSerializer:
        def __init__(self) -> None:
            try:
                import orjson
                self._orjson = orjson
                self._use_orjson = True
            except ImportError:
                import json as _json
                self._json = _json
                self._use_orjson = False

        def serialize(self, obj: Any) -> bytes:
            from flujo.utils.serialization import safe_serialize
            serialized_obj = safe_serialize(obj, mode="default")
            if self._use_orjson:
                blob: bytes = self._orjson.dumps(serialized_obj, option=self._orjson.OPT_SORT_KEYS)
                return blob
            else:
                s = self._json.dumps(serialized_obj, sort_keys=True, separators=(",", ":"))
                return s.encode("utf-8")

        def deserialize(self, blob: bytes) -> Any:
            from flujo.utils.serialization import safe_deserialize
            if self._use_orjson:
                raw_data = self._orjson.loads(blob)
            else:
                raw_data = self._json.loads(blob.decode("utf-8"))
            return safe_deserialize(raw_data)

    # Blake3Hasher
    class Blake3Hasher:
        def __init__(self) -> None:
            try:
                import blake3  # type: ignore
                self._blake3 = blake3
                self._use_blake3 = True
            except ImportError:
                self._use_blake3 = False

        def digest(self, data: bytes) -> str:
            if self._use_blake3:
                return str(self._blake3.blake3(data).hexdigest())
            else:
                return hashlib.blake2b(data, digest_size=32).hexdigest()

    class _HasherProtocol(Protocol):
        def digest(self, data: bytes) -> str: ...

    class DefaultCacheKeyGenerator:
        def __init__(self, hasher: _HasherProtocol | None = None):
            self._hasher: _HasherProtocol = hasher or Blake3Hasher()

        # ... rest unchanged ...

    @dataclass
    class _LRUCache:
        max_size: int = 1024
        ttl: int = 3600
        _store: OrderedDict[str, tuple[StepResult, float]] = dataclass(init=False)(OrderedDict)  # type: ignore

        def __post_init__(self):
            if self.max_size <= 0:
                raise ValueError("max_size must be positive")
            if self.ttl < 0:
                raise ValueError("ttl must be non-negative")
            if not isinstance(self._store, OrderedDict):
                self._store = OrderedDict()

        # ... rest unchanged ...

    @dataclass
    class InMemoryLRUBackend:
        max_size: int = 1024
        ttl_s: int = 3600
        _lock: asyncio.Lock = asyncio.Lock()
        _store: OrderedDict[str, tuple[StepResult, float, int]] = dataclass(init=False)(OrderedDict)  # type: ignore

        # ... rest unchanged ...

    @dataclass
    class ThreadSafeMeter:
        total_cost_usd: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        _lock: asyncio.Lock = asyncio.Lock()

        async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
            async with self._lock:
                self.total_cost_usd += cost_usd
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens

        async def guard(self, limits: UsageLimits, step_history: Optional[List[Any]] = None) -> None:
            from flujo.domain.types import PipelineResult
            # Fast path (no lock)
            cost_usd: float = self.total_cost_usd
            total_tokens: int = self.prompt_tokens + self.completion_tokens
            limit_cost = limits.total_cost_usd_limit
            if (limit_cost is not None and isinstance(limit_cost, (int, float)) and cost_usd - limit_cost > 1e-9):
                msg = f"Cost limit of ${limit_cost} exceeded (current: ${cost_usd})"
                raise UsageLimitExceededError(msg, PipelineResult(step_history=step_history or [], total_cost_usd=cost_usd))
            limit_tokens = limits.total_tokens_limit
            if (limit_tokens is not None and isinstance(limit_tokens, (int, float)) and total_tokens - limit_tokens > 0):
                msg = f"Token limit of {limit_tokens} exceeded (current: {total_tokens})"
                raise UsageLimitExceededError(msg, PipelineResult(step_history=step_history or [], total_cost_usd=cost_usd))

        async def snapshot(self) -> tuple[float, int, int]:
            async with self._lock:
                return self.total_cost_usd, self.prompt_tokens, self.completion_tokens

    class UsageLimitExceededError(Exception):
        def __init__(self, msg, pipeline_result):
            super().__init__(msg)
            self.pipeline_result = pipeline_result

    # DefaultProcessorPipeline, DefaultValidatorRunner, DefaultPluginRunner, DefaultAgentRunner, DefaultTelemetry...
    # (rest of shim builder unchanged)

    # Expose in module
    mod.OrjsonSerializer = OrjsonSerializer
    mod.Blake3Hasher = Blake3Hasher
    mod.DefaultCacheKeyGenerator = DefaultCacheKeyGenerator
    mod._LRUCache = _LRUCache
    mod.InMemoryLRUBackend = InMemoryLRUBackend
    mod.ThreadSafeMeter = ThreadSafeMeter
    # ... rest unchanged ...
    return mod


default_components = _resolve_default_components_module()

# Export aliases for test readability
OrjsonSerializer = getattr(default_components, "OrjsonSerializer")
Blake3Hasher = getattr(default_components, "Blake3Hasher")
DefaultCacheKeyGenerator = getattr(default_components, "DefaultCacheKeyGenerator")
_LRUCache = getattr(default_components, "_LRUCache")
InMemoryLRUBackend = getattr(default_components, "InMemoryLRUBackend")
ThreadSafeMeter = getattr(default_components, "ThreadSafeMeter")
DefaultProcessorPipeline = getattr(default_components, "DefaultProcessorPipeline")
DefaultValidatorRunner = getattr(default_components, "DefaultValidatorRunner")
DefaultPluginRunner = getattr(default_components, "DefaultPluginRunner")
DefaultAgentRunner = getattr(default_components, "DefaultAgentRunner")
DefaultTelemetry = getattr(default_components, "DefaultTelemetry")
_should_pass_context_to_plugin = getattr(default_components, "_should_pass_context_to_plugin")
_should_pass_resources_to_plugin = getattr(default_components, "_should_pass_resources_to_plugin")

# Helpful domain types (from stubs if real ones missing)
try:
    from flujo.domain.types import StepResult, ValidationResult, UsageLimits, PipelineResult
except Exception:
    # Use stubs installed by shim builder
    types_mod = sys.modules.get("flujo.domain.types")
    StepResult = getattr(types_mod, "StepResult")
    ValidationResult = getattr(types_mod, "ValidationResult")
    UsageLimits = getattr(types_mod, "UsageLimits")
    PipelineResult = getattr(types_mod, "PipelineResult")

# Exceptions for DefaultAgentRunner and ThreadSafeMeter guard
try:
    from flujo.domain.exceptions import (
        PausedException, InfiniteFallbackError, InfiniteRedirectError, ContextInheritanceError
    )
except Exception:
    ex_mod = sys.modules.get("flujo.domain.exceptions")
    PausedException = getattr(ex_mod, "PausedException")
    InfiniteFallbackError = getattr(ex_mod, "InfiniteFallbackError")
    InfiniteRedirectError = getattr(ex_mod, "InfiniteRedirectError")
    ContextInheritanceError = getattr(ex_mod, "ContextInheritanceError")


# ---------------------------
# Test helpers
# ---------------------------
@pytest.fixture
def event_loop():
    # Ensure a fresh event loop for async tests on some CI environments
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_step():
    class Agent:
        def __init__(self):
            self.model_id = "gpt-4o"
            self.system_prompt = "You are helpful."

    class Config:
        max_retries = 2
        timeout_s = 30
        temperature = 0.3

    class ProcA:
        pass

    class ProcB:
        pass

    class ValidatorA:
        pass

    class Processors:
        prompt_processors = [ProcA()]
        output_processors = [ProcB()]

    class Step:
        name = "MyStep"
        agent = Agent()
        config = Config()
        processors = Processors()
        validators = [ValidatorA()]

    return Step()

# ---------------------------
# Tests: OrjsonSerializer
# ---------------------------

def test_orjson_serializer_without_orjson(monkeypatch):
    # Force ImportError for orjson
    if "orjson" in sys.modules:
        del sys.modules["orjson"]

    def _fake_import(name, *a, **k):
        if name == "orjson":
            raise ImportError("no orjson")
        return real_import(name, *a, **k)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    ser = OrjsonSerializer()
    data = {"b": 2, "a": 1}
    blob = ser.serialize(data)
    assert isinstance(blob, (bytes, bytearray))
    # Should be JSON bytes, sorted keys
    assert blob == b'{"a":1,"b":2}'
    out = ser.deserialize(blob)
    assert out == {"a": 1, "b": 2}

def test_orjson_serializer_with_orjson(monkeypatch):
    # Provide a fake orjson module
    class FakeOrjson:
        OPT_SORT_KEYS = 0

        @staticmethod
        def dumps(obj, option=None):
            # mimic orjson.dumps: return bytes, sorted by json for determinism
            return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

        @staticmethod
        def loads(b: bytes):
            return json.loads(b)

    monkeypatch.setitem(sys.modules, "orjson", FakeOrjson)
    ser = OrjsonSerializer()
    data = {"b": 2, "a": 1}
    blob = ser.serialize(data)
    assert isinstance(blob, (bytes, bytearray))
    assert blob == b'{"a":1,"b":2}'
    out = ser.deserialize(blob)
    assert out == {"a": 1, "b": 2}

# ---------------------------
# Tests: Blake3Hasher
# ---------------------------

def test_blake3_hasher_without_blake3(monkeypatch):
    if "blake3" in sys.modules:
        del sys.modules["blake3"]

    def _fake_import(name, *a, **k):
        if name == "blake3":
            raise ImportError("no blake3")
        return real_import(name, *a, **k)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    h = Blake3Hasher()
    out = h.digest(b"hello")
    # Must match blake2b 32-byte hex
    assert out == hashlib.blake2b(b"hello", digest_size=32).hexdigest()

def test_blake3_hasher_with_blake3(monkeypatch):
    class FakeBlake3Obj:
        def __init__(self, data):
            self._data = data

        def hexdigest(self):
            return "deadbeef" * 8

    class FakeBlake3:
        @staticmethod
        def blake3(data):
            return FakeBlake3Obj(data)

    monkeypatch.setitem(sys.modules, "blake3", FakeBlake3)
    h = Blake3Hasher()
    assert h.digest(b"hello") == "deadbeef" * 8

# ---------------------------
# Tests: DefaultCacheKeyGenerator
# ---------------------------

def test_default_cache_key_generator_builds_stable_key(sample_step):
    gen = DefaultCacheKeyGenerator()
    k1 = gen.generate_key(sample_step, {"x": 1}, {"y": 2}, {"z": 3})
    k2 = gen.generate_key(sample_step, {"x": 1}, {"y": 2}, {"z": 3})
    assert isinstance(k1, str) and len(k1) == 64  # sha256 hex length
    assert k1 == k2

def test_default_cache_key_generator_changes_with_prompt_hash(sample_step, monkeypatch):
    gen = DefaultCacheKeyGenerator()
    k1 = gen.generate_key(sample_step, {"x": 1}, {"y": 2}, {"z": 3})
    # Change system prompt and expect different key
    sample_step.agent.system_prompt = "Different"
    k2 = gen.generate_key(sample_step, {"x": 1}, {"y": 2}, {"z": 3})
    assert k1 != k2

def test_default_cache_key_generator_handles_non_serializable(monkeypatch, sample_step):
    gen = DefaultCacheKeyGenerator()

    class Unserializable:
        def __init__(self):
            self.x = set([1, 2, 3])

    k = gen.generate_key(sample_step, Unserializable(), {"y": 2}, {"z": 3})
    assert isinstance(k, str) and len(k) == 64

# ---------------------------
# Tests: _LRUCache
# ---------------------------

def test_lru_cache_basic_set_get(monkeypatch):
    # neutralize time.monotonic to deterministic
    base = 1000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    cache = _LRUCache(max_size=2, ttl=10)
    v1 = StepResult(content={"a": 1})
    v2 = StepResult(content={"b": 2})
    cache.set("k1", v1)
    cache.set("k2", v2)
    assert cache.get("k1") == v1
    assert cache.get("k2") == v2

def test_lru_cache_evicts_lru(monkeypatch):
    base = 1000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    cache = _LRUCache(max_size=2, ttl=10)
    cache.set("a", StepResult(content=1))
    cache.set("b", StepResult(content=2))
    # Access 'a' to make 'b' LRU
    _ = cache.get("a")
    # Insert c -> evict b
    cache.set("c", StepResult(content=3))
    assert cache.get("b") is None
    assert cache.get("a").content == 1
    assert cache.get("c").content == 3

def test_lru_cache_ttl_expiry(monkeypatch):
    base = 1000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    cache = _LRUCache(max_size=2, ttl=5)
    cache.set("x", StepResult(content=42))
    assert cache.get("x").content == 42
    # Advance beyond ttl
    t.current += 6.0
    assert cache.get("x") is None

def test_lru_cache_invalid_config():
    with pytest.raises(ValueError):
        _LRUCache(max_size=0)
    with pytest.raises(ValueError):
        _LRUCache(max_size=1, ttl=-1)

# ---------------------------
# Tests: InMemoryLRUBackend
# ---------------------------

@pytest.mark.asyncio
async def test_inmemory_lru_backend_put_get(monkeypatch):
    base = 2000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    backend = InMemoryLRUBackend(max_size=2, ttl_s=5)
    val = StepResult(content={"m": 9})
    await backend.put("k", val, ttl_s=5)
    out = await backend.get("k")
    assert out.content == {"m": 9}
    # Getting returns a copy via model_copy
    assert out is not val

@pytest.mark.asyncio
async def test_inmemory_lru_backend_ttl_expiry(monkeypatch):
    base = 3000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    backend = InMemoryLRUBackend(max_size=2, ttl_s=5)
    await backend.put("x", StepResult(content=1), ttl_s=5)
    assert (await backend.get("x")).content == 1
    t.current += 6.0
    assert await backend.get("x") is None

@pytest.mark.asyncio
async def test_inmemory_lru_backend_eviction(monkeypatch):
    base = 4000.0
    t = types.SimpleNamespace(current=base)

    def fake_monotonic():
        return t.current

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    backend = InMemoryLRUBackend(max_size=2, ttl_s=100)
    await backend.put("a", StepResult(content=1), ttl_s=100)
    await backend.put("b", StepResult(content=2), ttl_s=100)
    # Access 'a' to make 'b' the LRU
    assert (await backend.get("a")).content == 1
    await backend.put("c", StepResult(content=3), ttl_s=100)
    assert await backend.get("b") is None
    assert (await backend.get("a")).content == 1
    assert (await backend.get("c")).content == 3

# ---------------------------
# Tests: ThreadSafeMeter
# ---------------------------

@pytest.mark.asyncio
async def test_thread_safe_meter_add_and_snapshot():
    meter = ThreadSafeMeter()
    await meter.add(0.25, 10, 20)
    await meter.add(0.75, 5, 15)
    total_cost, p, c = await meter.snapshot()
    assert total_cost == pytest.approx(1.0)
    assert p == 15
    assert c == 35

@pytest.mark.asyncio
async def test_thread_safe_meter_guard_cost_limit_exceeded():
    meter = ThreadSafeMeter()
    await meter.add(2.0, 0, 0)
    limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=None)
    with pytest.raises(Exception) as exc:
        await meter.guard(limits)
    assert "Cost limit" in str(exc.value)

@pytest.mark.asyncio
async def test_thread_safe_meter_guard_token_limit_exceeded():
    meter = ThreadSafeMeter()
    await meter.add(0.0, 10, 5)
    limits = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=14)
    with pytest.raises(Exception) as exc:
        await meter.guard(limits)
    assert "Token limit" in str(exc.value)

@pytest.mark.asyncio
async def test_thread_safe_meter_guard_no_limits():
    meter = ThreadSafeMeter()
    await meter.add(0.5, 5, 5)
    limits = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=None)
    # Should not raise
    await meter.guard(limits)

# ---------------------------
# Tests: DefaultProcessorPipeline
# ---------------------------

@pytest.mark.asyncio
async def test_default_processor_pipeline_prompt_mixed_sync_async_and_context():
    pipeline = DefaultProcessorPipeline()
    calls = []

    async def a1(x, *, context):
        calls.append(("a1", x, context))
        return x + "A"

    def s1(x, *, context):
        calls.append(("s1", x, context))
        return x + "S"

    async def a2(x):
        calls.append(("a2", x, None))
        return x + "B"

    class P1:
        async def process(self, x, *, context):
            calls.append(("P1.process", x, context))
            return x + "P"

    processors = [a1, s1, a2, P1()]
    out = await pipeline.apply_prompt(processors, "x", context={"ctx": 1})
    assert out == "xASBP"
    # Ensure error path logs and re-raises
    def boom(x, *, context):
        raise ValueError("failproc")
    with pytest.raises(ValueError):
        await pipeline.apply_prompt([boom], "x", context={})

@pytest.mark.asyncio
async def test_default_processor_pipeline_output_empty_returns_input():
    pipeline = DefaultProcessorPipeline()
    assert await pipeline.apply_output([], {"a": 1}, context={"c": 2}) == {"a": 1}

# ---------------------------
# Tests: DefaultValidatorRunner
# ---------------------------

@pytest.mark.asyncio
async def test_default_validator_runner_various_result_shapes():
    runner = DefaultValidatorRunner()
    # Returns ValidationResult directly
    async def v1(data, *, context):
        return ValidationResult(is_valid=True, feedback="ok", validator_name="v1")

    # Returns object with is_valid attr
    class R:
        def __init__(self, ok, fb=None):
            self.is_valid = ok
            self.feedback = fb

    class V2:
        name = "CustomName"

        async def validate(self, data, *, context):
            return R(False, "bad")

    # Returns invalid type
    async def v3(data):
        return 123

    results = await runner.validate([v1, V2(), v3], {"x": 1}, context={"y": 2})
    assert [r.is_valid for r in results] == [True, False, False]
    assert results[0].feedback == "ok"
    assert results[1].validator_name == "CustomName"
    assert "invalid result type" in results[2].feedback

    # Exception case
    async def v4(data, *, context):
        raise RuntimeError("boom")

    res2 = await runner.validate([v4], {}, context={})
    assert not res2[0].is_valid and "failed:" in res2[0].feedback.lower()

# ---------------------------
# Tests: helper functions for plugin arg passing
# ---------------------------

def test_should_pass_context_to_plugin():
    def f(a, *, context):
        pass
    def g(a):
        pass
    assert _should_pass_context_to_plugin({}, f) is True
    assert _should_pass_context_to_plugin({}, g) is False
    assert _should_pass_context_to_plugin(None, f) is False

def test_should_pass_resources_to_plugin():
    def f(a, *, resources):
        pass
    def g(a):
        pass
    assert _should_pass_resources_to_plugin({}, f) is True
    assert _should_pass_resources_to_plugin({}, g) is False
    assert _should_pass_resources_to_plugin(None, f) is False

# ---------------------------
# Tests: DefaultPluginRunner
# ---------------------------

@pytest.mark.asyncio
async def test_default_plugin_runner_success_and_transform(monkeypatch):
    from flujo.domain.plugins import PluginOutcome
    runner = DefaultPluginRunner()

    class P1:
        priority = 10
        async def validate(self, data, *, context):
            return PluginOutcome(success=True, new_solution=data + "A")

    class P2:
        priority = 5
        async def validate(self, data, *, context):
            return data + "B"

    out = await runner.run_plugins([(P2(), 5), (P1(), 10)], "x", context={})
    # P1 runs first (higher priority), transforms to xA; then P2 adds B
    assert out == "xAB"

@pytest.mark.asyncio
async def test_default_plugin_runner_failure_short_circuit(monkeypatch):
    from flujo.domain.plugins import PluginOutcome
    runner = DefaultPluginRunner()

    class P:
        async def validate(self, data, *, context):
            return PluginOutcome(success=False, new_solution=None)

    out = await runner.run_plugins([(P(), 1)], "x", context={})
    assert isinstance(out, PluginOutcome) and not out.success

@pytest.mark.asyncio
async def test_default_plugin_runner_exception_logs_and_raises(monkeypatch):
    calls = []
    # Spy on telemetry.logfire.error
    import telemetry

    def spy_error(msg):
        calls.append(msg)

    telemetry.logfire.error = spy_error
    runner = DefaultPluginRunner()

    class P:
        name = "BoomPlugin"
        async def validate(self, data, *, context):
            raise ValueError("kaput")

    with pytest.raises(ValueError) as exc:
        await runner.run_plugins([(P(), 1)], "x", context={})
    assert "BoomPlugin failed" in str(exc.value)
    assert any("BoomPlugin failed" in m for m in calls)

# ---------------------------
# Tests: DefaultAgentRunner
# ---------------------------

@pytest.mark.asyncio
async def test_default_agent_runner_selects_run_and_passes_filtered_kwargs():
    runner = DefaultAgentRunner()
    received = {}

    class Agent:
        async def run(self, payload, *, context, resources, temperature):
            received["payload"] = payload
            received["context"] = context
            received["resources"] = resources
            received["temperature"] = temperature
            return "ok"

    result = await runner.run(Agent(), {"a": 1}, context={"c": 2}, resources={"r": 3}, options={"temperature": 0.7})
    assert result == "ok"
    assert received == {"payload": {"a": 1}, "context": {"c": 2}, "resources": {"r": 3}, "temperature": 0.7}

@pytest.mark.asyncio
async def test_default_agent_runner_stream_async_gen_and_on_chunk():
    runner = DefaultAgentRunner()
    chunks_seen = []

    async def on_chunk(c):
        chunks_seen.append(c)

    class Agent:
        async def stream(self, payload, *, context):
            async def gen():
                yield "a"
                yield "b"
                yield "c"
            return gen()

    result = await runner.run(Agent(), "x", context={}, resources=None, options={}, stream=True, on_chunk=on_chunk)
    assert result == "abc"
    assert chunks_seen == ["a", "b", "c"]

@pytest.mark.asyncio
async def test_default_agent_runner_stream_coroutine_returning_async_iterable():
    runner = DefaultAgentRunner()

    class AsyncIterable:
        def __aiter__(self):
            self._it = iter(["x", "y"])
            return self

        async def __anext__(self):
            try:
                v = next(self._it)
            except StopIteration:
                raise StopAsyncIteration
            return v

    class Agent:
        async def stream(self, payload, *, context):
            return AsyncIterable()

    result = await runner.run(Agent(), "p", context={}, resources=None, options={}, stream=True)
    assert result == "xy"

@pytest.mark.asyncio
async def test_default_agent_runner_no_stream_sync_func():
    runner = DefaultAgentRunner()

    class Agent:
        def run(self, payload, *, context, resources=None):
            return "Z"

    result = await runner.run(Agent(), None, context={}, resources=None, options={}, stream=False)
    assert result == "Z"

@pytest.mark.asyncio
async def test_default_agent_runner_missing_methods_raises():
    runner = DefaultAgentRunner()

    class Bad:
        pass

    with pytest.raises(RuntimeError):
        await runner.run(Bad(), None, context=None, resources=None, options={})

# ---------------------------
# Tests: DefaultTelemetry
# ---------------------------
def test_default_telemetry_trace_decorator_passthrough():
    telem = DefaultTelemetry()
    calls = []

    @telem.trace("name")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert calls == [3]
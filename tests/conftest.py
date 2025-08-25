import os
from typing import Any, Optional, Dict
from flujo import Flujo
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl import Step
from flujo.state.backends.base import StateBackend
from flujo.utils.serialization import register_custom_serializer, reset_custom_serializer_registry
from collections import OrderedDict, Counter, defaultdict
from enum import Enum
import pytest
import threading
import os as _os
import re

# Set test mode environment variable
os.environ["FLUJO_TEST_MODE"] = "1"


# Define mock classes that need serialization support
# These are defined at module level to match the actual test implementations


# MockEnum class (from test_serialization_edge_cases.py)
class MockEnum(Enum):
    """Mock enum for edge case testing."""

    A = "a"
    B = "b"
    C = "c"


# Register the MockEnum serializer at module level for all test runs
register_custom_serializer(MockEnum, lambda obj: obj.value)


# UsageResponse class (from test_usage_limits_enforcement.py)
class UsageResponse:
    def __init__(self, output: Any, cost: float, tokens: int):
        self.output = output
        self.cost_usd = cost
        self.token_counts = tokens

    def usage(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.token_counts,
            "completion_tokens": 0,
            "total_tokens": self.token_counts,
            "cost_usd": self.cost_usd,
        }


# MockImageResult class (from test_explicit_cost_integration.py)
class MockImageResult:
    def __init__(self, cost_usd: float, token_counts: int = 0):
        self.cost_usd = cost_usd
        self.token_counts = token_counts
        self.output = f"Mock image result with cost ${cost_usd} and {token_counts} tokens"


# WrappedResult class (from test_pipeline_runner.py and test_fallback.py)
class WrappedResult:
    def __init__(self, output: str, token_counts: int = 2, cost_usd: float = 0.1) -> None:
        self.output = output
        self.token_counts = token_counts
        self.cost_usd = cost_usd


# AgentResponse class (from test_image_cost_integration.py)
class AgentResponse:
    def __init__(self, output: Any, cost_usd: float = 0.0, token_counts: int = 0):
        self.output = output
        self.cost_usd = cost_usd
        self.token_counts = token_counts


# MockResponseWithBoth class (from test_explicit_cost_integration.py)
class MockResponseWithBoth:
    def __init__(self):
        self.cost_usd = 0.1
        self.token_counts = 50
        self.output = "Mock response with both protocol and usage method"

    def usage(self):
        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 25
                self.completion_tokens = 25
                self.total_tokens = 50
                self.cost_usd = 0.1

        return MockUsage()


# MockResponseWithNone class (from test_explicit_cost_integration.py)
class MockResponseWithNone:
    def __init__(self):
        self.cost_usd = None
        self.token_counts = None
        self.output = "Mock response with None values"


# MockResponseWithUsageOnly class (from test_explicit_cost_integration.py)
class MockResponseWithUsageOnly:
    def __init__(self):
        self.output = "test"

    def usage(self):
        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 5
                self.total_tokens = 15
                self.cost_usd = 0.05

        return MockUsage()


def _register_baseline_serializers() -> None:
    """Register baseline serializers used across tests.

    This mirrors the session-level setup so we can restore a clean, known state
    before each test when using randomized ordering and xdist.
    """
    # Register the MockEnum serializer at module level for all test runs
    register_custom_serializer(MockEnum, lambda obj: obj.value)

    # Register serializers for all mock classes
    # Use simple __dict__ serialization for all mock objects
    register_custom_serializer(UsageResponse, lambda obj: obj.__dict__)
    register_custom_serializer(MockImageResult, lambda obj: obj.__dict__)
    register_custom_serializer(WrappedResult, lambda obj: obj.__dict__)
    register_custom_serializer(AgentResponse, lambda obj: obj.__dict__)
    register_custom_serializer(MockResponseWithBoth, lambda obj: obj.__dict__)
    register_custom_serializer(MockResponseWithNone, lambda obj: obj.__dict__)
    register_custom_serializer(MockResponseWithUsageOnly, lambda obj: obj.__dict__)

    # Register serializers for edge case types
    register_custom_serializer(OrderedDict, lambda obj: dict(obj))
    register_custom_serializer(Counter, lambda obj: dict(obj))
    register_custom_serializer(defaultdict, lambda obj: dict(obj))

    # Register serializers for common types that should be preserved
    import uuid
    from datetime import datetime, date, time
    from decimal import Decimal

    register_custom_serializer(uuid.UUID, lambda obj: obj)  # Keep UUID objects as-is
    register_custom_serializer(datetime, lambda obj: obj)  # Keep datetime objects as-is
    register_custom_serializer(date, lambda obj: obj)  # Keep date objects as-is
    register_custom_serializer(time, lambda obj: obj)  # Keep time objects as-is
    register_custom_serializer(Decimal, lambda obj: obj)  # Keep Decimal objects as-is


@pytest.fixture(scope="session", autouse=True)
def register_mock_serializers():
    """
    Register custom serializers for mock objects used in tests.

    This fixture automatically runs for all tests and ensures that mock objects
    like UsageResponse, MockImageResult, and WrappedResult can be properly
    serialized by the framework's serialization system.
    """
    _register_baseline_serializers()

    # Register fallback serializer for unknown types with __dict__
    def fallback_dict_serializer(obj):
        """Fallback serializer for objects with __dict__ attribute."""
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    # Note: The serialization system already handles objects with __dict__ automatically
    # in the safe_serialize function, so we don't need a fallback serializer

    # Yield to allow tests to run
    yield

    # Clean up the registry after all tests complete
    reset_custom_serializer_registry()


@pytest.fixture(autouse=True)
def _reset_registry_per_test():
    """Ensure custom serializer registry is clean for each test.

    Randomized test ordering can otherwise leak serializers between tests
    (e.g., MyCustomObject) and cause unexpected behavior. We reset to a known
    baseline before each test.
    """
    reset_custom_serializer_registry()
    _register_baseline_serializers()
    yield


@pytest.fixture(autouse=True)
def _clear_state_uri_env(monkeypatch):
    """Prevent FLUJO_STATE_URI leakage between tests when using xdist/random.

    Some tests set FLUJO_STATE_URI explicitly; others expect TOML defaults.
    Clearing it at test start ensures isolation and avoids cross-test failures.
    """
    monkeypatch.delenv("FLUJO_STATE_URI", raising=False)
    yield


def create_test_flujo(
    pipeline: Pipeline[Any, Any] | Step[Any, Any],
    *,
    pipeline_name: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    **kwargs: Any,
) -> Flujo[Any, Any, Any]:
    """Create a Flujo instance with proper test names and IDs.

    This utility function provides meaningful pipeline names and IDs for tests
    while ensuring the warnings are suppressed in test environments.

    Parameters
    ----------
    pipeline : Pipeline | Step
        The pipeline or step to run
    pipeline_name : str, optional
        Custom pipeline name. If not provided, generates one based on test function name.
    pipeline_id : str, optional
        Custom pipeline ID. If not provided, generates a unique test ID.
    **kwargs : Any
        Additional arguments to pass to Flujo constructor

    Returns
    -------
    Flujo
        Configured Flujo instance with proper test identifiers
    """
    if pipeline_name is None:
        # Generate a descriptive name based on the test function
        import inspect

        frame = inspect.currentframe()
        while frame and not frame.f_code.co_name.startswith("test_"):
            frame = frame.f_back
        if frame:
            pipeline_name = f"test_{frame.f_code.co_name}"
        else:
            pipeline_name = "test_pipeline"

    if pipeline_id is None:
        import uuid

        pipeline_id = f"test_{uuid.uuid4().hex[:8]}"

    # Always use NoOpStateBackend for test isolation unless explicitly overridden
    if "state_backend" not in kwargs:
        kwargs["state_backend"] = NoOpStateBackend()

    return Flujo(pipeline, pipeline_name=pipeline_name, pipeline_id=pipeline_id, **kwargs)


def pytest_ignore_collect(collection_path, config):  # type: ignore[override]
    """Ignore accidentally duplicated test files like 'test_foo 2.py'.

    This prevents pytest from collecting backup copies that end with ' 2.py'.
    The cleanup script in scripts/cleanup_duplicate_tests.py can remove or move them.
    """
    try:
        p = str(collection_path)
        if p.endswith(" 2.py"):
            return True
    except Exception:
        return None
    return None


def get_registered_factory(skill_id: str):
    """Get a registered factory from the skill registry.

    This helper function ensures builtins are registered and retrieves the factory
    for a given skill ID. It's used across multiple test files to reduce duplication.

    Parameters
    ----------
    skill_id : str
        The skill ID to look up in the registry

    Returns
    -------
    Any
        The factory function for the skill

    Raises
    ------
    AssertionError
        If the skill is not registered in the registry
    """
    from flujo.builtins import _register_builtins
    from flujo.infra.skill_registry import get_skill_registry

    # Ensure builtins are registered
    _register_builtins()

    reg = get_skill_registry()
    entry = reg.get(skill_id)
    assert entry is not None, f"Skill not registered: {skill_id}"
    return entry["factory"]


@pytest.fixture()
def no_wait_backoff(monkeypatch: pytest.MonkeyPatch):
    """Disable tenacity backoff in retry loops for fast unit tests.

    Patches flujo.agents.wrapper.wait_exponential to wait_none so retries do not sleep.
    """
    import tenacity as _tenacity

    monkeypatch.setattr(
        "flujo.agents.wrapper.wait_exponential",
        lambda **_k: _tenacity.wait_none(),
        raising=True,
    )
    yield


class NoOpStateBackend(StateBackend):
    """A state backend that simulates real backend behavior for testing while maintaining isolation."""

    def __init__(self):
        # Store serialized copies to mimic persistent backends (but in memory for tests)
        self._store: Dict[str, Any] = {}
        self._trace_store: Dict[str, Any] = {}

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        # Simulate real backend behavior by serializing and storing state
        from flujo.utils.serialization import safe_serialize

        self._store[run_id] = safe_serialize(state)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        # Simulate real backend behavior by deserializing stored state
        stored = self._store.get(run_id)
        if stored is None:
            return None
        from flujo.utils.serialization import safe_deserialize
        from copy import deepcopy

        # Return a deserialized copy to avoid accidental mutation
        return deepcopy(safe_deserialize(stored))

    async def delete_state(self, run_id: str) -> None:
        # Simulate real backend behavior by removing stored state
        self._store.pop(run_id, None)
        self._trace_store.pop(run_id, None)

    async def get_trace(self, run_id: str) -> Any:
        # Simulate real backend behavior by returning stored trace data
        return self._trace_store.get(run_id)

    async def save_trace(self, run_id: str, trace: Any) -> None:
        # Simulate real backend behavior by storing trace data
        from flujo.utils.serialization import safe_serialize

        self._trace_store[run_id] = safe_serialize(trace)


def _diagnose_threads() -> None:
    try:
        alive = [t for t in threading.enumerate() if t.is_alive()]
        non_daemon = [t for t in alive if not t.daemon]
        if non_daemon:
            print("\n[pytest-sessionfinish] Non-daemon threads still alive:")
            for t in non_daemon:
                print(f"  - {t.name} (id={getattr(t, 'ident', '?')})")
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):  # type: ignore
    """Best-effort cleanup for background services to avoid process hang."""
    # Attempt to stop any prometheus servers started during tests
    try:
        from flujo.telemetry.prometheus import shutdown_all_prometheus_servers

        shutdown_all_prometheus_servers()
    except Exception:
        pass
    # Optional: print any non-daemon threads for debugging when enabled
    try:
        if _os.environ.get("FLUJO_TEST_DEBUG_THREADS") == "1":
            _diagnose_threads()
    except Exception:
        pass
    # Ensure SQLite backends are shut down to avoid lingering aiosqlite threads
    try:
        from flujo.state.backends.sqlite import SQLiteBackend

        SQLiteBackend.shutdown_all()
    except Exception:
        pass
    # As a last-resort, force-exit the interpreter in CI/test runs to avoid hangs
    try:
        if _os.environ.get("FLUJO_TEST_FORCE_EXIT") == "1":
            _os._exit(exitstatus)
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):  # type: ignore[override]
    """Optionally skip tests via env var patterns without editing tests.

    - Set `FLUJO_SKIP_TESTS` to a comma-separated list of regex patterns that
      will be matched against each test's nodeid. Matching items are deselected.
      Example:
        FLUJO_SKIP_TESTS="tests/unit/test_sql_injection_security.py::TestSQLInjectionSecurity::test_save_state_sql_injection_resistance"

    - Convenience toggle to skip the entire SQL injection security file:
        FLUJO_SKIP_SQL_INJECTION_SECURITY=1
    """
    patterns: list[str] = []
    if _os.environ.get("FLUJO_SKIP_SQL_INJECTION_SECURITY") == "1":
        patterns.append(r"tests/unit/test_sql_injection_security\.py")
    env_patterns = _os.environ.get("FLUJO_SKIP_TESTS", "").strip()
    if env_patterns:
        patterns.extend([p.strip() for p in env_patterns.split(",") if p.strip()])
    if not patterns:
        return

    deselected = []
    kept = []
    for item in items:
        nodeid = item.nodeid
        if any(re.search(p, nodeid) for p in patterns):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

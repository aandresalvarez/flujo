from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse
import os
import importlib.util

from flujo.application.core.default_components import (
    DefaultAgentRunner,
    DefaultPluginRunner,
    DefaultProcessorPipeline,
    DefaultTelemetry,
    DefaultValidatorRunner,
)
from flujo.application.core.default_cache_components import (
    Blake3Hasher,
    InMemoryLRUBackend,
    OrjsonSerializer,
    ThreadSafeMeter,
)
from flujo.application.core.estimation import build_default_estimator_factory
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_protocols import (
    IAgentRunner,
    ICacheBackend,
    ITelemetry,
    IProcessorPipeline,
    IValidatorRunner,
    IPluginRunner,
    IUsageMeter,
)
from flujo.application.core.policy_registry import PolicyRegistry, StepPolicy, StepType
from flujo.infra.backends import LocalBackend
from flujo.state.backends.memory import InMemoryBackend
from flujo.state.backends.postgres import PostgresBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.state.backends.base import StateBackend
from flujo.utils.config import get_settings, Settings
from flujo.infra.config_manager import get_config_manager, get_state_uri
from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.domain.interfaces import StateProvider


class ExecutorFactory:
    """Factory to construct ExecutorCore with consistent defaults and optional overrides."""

    def __init__(
        self,
        *,
        telemetry: ITelemetry | None = None,
        cache_backend: ICacheBackend | None = None,
        agent_runner: IAgentRunner | None = None,
        processor_pipeline: IProcessorPipeline | None = None,
        validator_runner: IValidatorRunner | None = None,
        plugin_runner: IPluginRunner | None = None,
        usage_meter: IUsageMeter | None = None,
        state_providers: dict[str, StateProvider[object]] | None = None,
        policy_registry: PolicyRegistry | None = None,
        policy_overrides: Sequence[StepPolicy[StepType]] | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._cache_backend = cache_backend
        self._agent_runner = agent_runner
        self._processor_pipeline = processor_pipeline
        self._validator_runner = validator_runner
        self._plugin_runner = plugin_runner
        self._usage_meter = usage_meter
        self._state_providers = state_providers or {}
        self._policy_registry = policy_registry
        self._policy_overrides = list(policy_overrides) if policy_overrides else []

    def create_executor(self) -> ExecutorCore[DomainBaseModel]:
        """Return a configured ExecutorCore."""
        registry = self._policy_registry or PolicyRegistry()
        executor: ExecutorCore[DomainBaseModel] = ExecutorCore(
            serializer=OrjsonSerializer(),
            hasher=Blake3Hasher(),
            cache_backend=self._cache_backend or InMemoryLRUBackend(),
            usage_meter=self._usage_meter or ThreadSafeMeter(),
            agent_runner=self._agent_runner or DefaultAgentRunner(),
            processor_pipeline=self._processor_pipeline or DefaultProcessorPipeline(),
            validator_runner=self._validator_runner or DefaultValidatorRunner(),
            plugin_runner=self._plugin_runner or DefaultPluginRunner(),
            telemetry=self._telemetry or DefaultTelemetry(),
            estimator_factory=build_default_estimator_factory(),
            state_providers=self._state_providers,
            policy_registry=registry,
        )
        for policy in self._policy_overrides:
            registry.register(policy)
        return executor


class BackendFactory:
    """Factory for execution and state backends using centralized defaults."""

    def __init__(self, executor_factory: Optional[ExecutorFactory] = None) -> None:
        self._executor_factory = executor_factory or ExecutorFactory()
        # Reuse backends per target to avoid leaking connection pools in per-request runners.
        # Keyed by normalized target (dsn or sqlite path).
        self._state_backend_cache: dict[str, StateBackend] = {}

    def create_execution_backend(
        self, executor: ExecutorCore[DomainBaseModel] | None = None
    ) -> LocalBackend[DomainBaseModel]:
        exec_core = executor or self._executor_factory.create_executor()
        return LocalBackend(executor=exec_core)

    def create_state_backend(
        self, *, settings: Settings | None = None, db_path: Path | str | None = None
    ) -> StateBackend:
        cfg = settings or get_settings()
        if bool(getattr(cfg, "test_mode", False)):
            return InMemoryBackend()

        state_uri = get_state_uri(force_reload=True)
        if state_uri:
            parsed = urlparse(state_uri)
            scheme = parsed.scheme.lower()
            if scheme in {"postgres", "postgresql"}:
                # Guard: Check if asyncpg is available before creating PostgresBackend
                spec = importlib.util.find_spec("asyncpg")
                if spec is None:
                    raise ImportError(
                        "asyncpg is required for PostgreSQL support. "
                        "Install with `pip install flujo[postgres]`."
                    )
                extended_settings = get_config_manager().get_settings()
                pool_min = getattr(extended_settings, "postgres_pool_min", 1)
                pool_max = getattr(extended_settings, "postgres_pool_max", 10)
                auto_migrate = os.getenv("FLUJO_AUTO_MIGRATE", "true").lower() != "false"
                cache_key = f"postgres:{state_uri}:{pool_min}:{pool_max}:{auto_migrate}"
                backend = self._state_backend_cache.get(cache_key)
                if backend is None:
                    backend = PostgresBackend(
                        state_uri,
                        auto_migrate=auto_migrate,
                        pool_min_size=pool_min,
                        pool_max_size=pool_max,
                    )
                    self._state_backend_cache[cache_key] = backend
                return backend
            if scheme in {"memory", "mem", "inmemory"}:
                return InMemoryBackend()
            if scheme.startswith("sqlite"):
                from flujo.state.sqlite_uri import normalize_sqlite_path

                cfg_path = getattr(get_config_manager(), "config_path", None)
                cfg_dir = cfg_path.parent if cfg_path else Path.cwd()
                sqlite_path = normalize_sqlite_path(state_uri, Path.cwd(), config_dir=cfg_dir)
                cache_key = f"sqlite:{sqlite_path}"
                backend = self._state_backend_cache.get(cache_key)
                if backend is None:
                    backend = SQLiteBackend(sqlite_path)
                    self._state_backend_cache[cache_key] = backend
                return backend

        if db_path is not None:
            target_path = Path(db_path)
        else:
            root_hint = os.getenv("FLUJO_PROJECT_ROOT")
            try:
                root_base = Path(root_hint).resolve() if root_hint else Path.cwd()
            except Exception:
                root_base = Path.cwd()
            target_path = root_base / "flujo_ops.db"
        cache_key = f"sqlite:{target_path}"
        backend = self._state_backend_cache.get(cache_key)
        if backend is None:
            backend = SQLiteBackend(target_path)
            self._state_backend_cache[cache_key] = backend
        return backend

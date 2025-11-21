from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from flujo.application.core.default_components import (
    Blake3Hasher,
    DefaultAgentRunner,
    DefaultPluginRunner,
    DefaultProcessorPipeline,
    DefaultTelemetry,
    DefaultValidatorRunner,
    InMemoryLRUBackend,
    OrjsonSerializer,
    ThreadSafeMeter,
)
from flujo.application.core.estimation import build_default_estimator_factory
from flujo.application.core.executor_core import ExecutorCore
from flujo.infra.backends import LocalBackend
from flujo.state.backends.memory import InMemoryBackend
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.utils.config import get_settings


class ExecutorFactory:
    """Factory to construct ExecutorCore with consistent defaults and optional overrides."""

    def __init__(
        self,
        *,
        telemetry: Any | None = None,
        cache_backend: Any | None = None,
        optimization_config: Any | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._cache_backend = cache_backend
        self._optimization_config = optimization_config

    def create_executor(self) -> ExecutorCore[Any]:
        """Return a configured ExecutorCore."""
        return ExecutorCore(
            serializer=OrjsonSerializer(),
            hasher=Blake3Hasher(),
            cache_backend=self._cache_backend or InMemoryLRUBackend(),
            usage_meter=ThreadSafeMeter(),
            agent_runner=DefaultAgentRunner(),
            processor_pipeline=DefaultProcessorPipeline(),
            validator_runner=DefaultValidatorRunner(),
            plugin_runner=DefaultPluginRunner(),
            telemetry=self._telemetry or DefaultTelemetry(),
            optimization_config=self._optimization_config,
            estimator_factory=build_default_estimator_factory(),
        )


class BackendFactory:
    """Factory for execution and state backends using centralized defaults."""

    def __init__(self, executor_factory: Optional[ExecutorFactory] = None) -> None:
        self._executor_factory = executor_factory or ExecutorFactory()

    def create_execution_backend(
        self, executor: ExecutorCore[Any] | None = None
    ) -> LocalBackend[Any]:
        exec_core = executor or self._executor_factory.create_executor()
        return LocalBackend(executor=exec_core)

    def create_state_backend(
        self, *, settings: Any | None = None, db_path: Path | str | None = None
    ) -> Any:
        cfg = settings or get_settings()
        if bool(getattr(cfg, "test_mode", False)):
            return InMemoryBackend()

        target_path = Path(db_path) if db_path is not None else Path.cwd() / "flujo_ops.db"
        return SQLiteBackend(target_path)

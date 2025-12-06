from __future__ import annotations

from unittest.mock import Mock

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.quota_manager import QuotaManager
from flujo.application.core.runtime_builder import FlujoRuntimeBuilder


def test_executor_core_accepts_injected_dependencies() -> None:
    custom_agent_runner = Mock()
    custom_quota_manager = QuotaManager()
    custom_cache_backend = Mock()

    deps = FlujoRuntimeBuilder().build(
        agent_runner=custom_agent_runner,
        quota_manager=custom_quota_manager,
        cache_backend=custom_cache_backend,
        enable_cache=False,
    )

    core = ExecutorCore(deps=deps, enable_cache=False)

    assert core._agent_runner is custom_agent_runner
    assert core._quota_manager is custom_quota_manager
    assert core._cache_manager.backend is custom_cache_backend
    assert core._cache_manager.is_cache_enabled() is False


def test_executor_core_legacy_kwargs_still_override_defaults() -> None:
    custom_usage_meter = Mock()

    core = ExecutorCore(usage_meter=custom_usage_meter)

    assert core._usage_meter is custom_usage_meter


def test_executor_core_accepts_custom_registry_and_handler_factories() -> None:
    custom_registry = Mock()
    custom_policy_handlers = Mock()
    custom_dispatcher = Mock()
    custom_dispatch_handler = Mock()
    custom_result_handler = Mock()
    custom_telemetry_handler = Mock()
    custom_step_handler = Mock()
    custom_agent_handler = Mock()

    seen: dict[str, object] = {}

    def registry_factory(core: ExecutorCore) -> object:
        seen["registry_core"] = core
        return custom_registry

    def dispatcher_factory(registry: object, core: ExecutorCore) -> object:
        seen["dispatcher_registry"] = registry
        seen["dispatcher_core"] = core
        return custom_dispatcher

    deps = FlujoRuntimeBuilder().build(
        policy_registry_factory=registry_factory,
        policy_handlers_factory=lambda core: custom_policy_handlers,
        dispatcher_factory=dispatcher_factory,
        dispatch_handler_factory=lambda core: custom_dispatch_handler,
        result_handler_factory=lambda core: custom_result_handler,
        telemetry_handler_factory=lambda core: custom_telemetry_handler,
        step_handler_factory=lambda core: custom_step_handler,
        agent_handler_factory=lambda core: custom_agent_handler,
    )

    core = ExecutorCore(deps=deps)

    assert core.policy_registry is custom_registry
    assert seen["registry_core"] is core
    assert seen["dispatcher_registry"] is custom_registry
    assert seen["dispatcher_core"] is core
    assert core._policy_handlers is custom_policy_handlers
    assert core._dispatcher is custom_dispatcher
    assert core._dispatch_handler is custom_dispatch_handler
    assert core._result_handler is custom_result_handler
    assert core._telemetry_handler is custom_telemetry_handler
    assert core._step_handler is custom_step_handler
    assert core._agent_handler is custom_agent_handler

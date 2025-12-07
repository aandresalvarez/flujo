from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.runtime_builder import FlujoRuntimeBuilder


def test_executor_core_uses_injected_agent_runner():
    custom_runner = object()
    core = ExecutorCore(agent_runner=custom_runner, enable_cache=False)
    assert core._agent_runner is custom_runner


def test_executor_core_skips_builder_when_deps_provided(monkeypatch):
    # Build deps once up front (uses real builder)
    deps = FlujoRuntimeBuilder().build(enable_cache=False)

    # If ExecutorCore tried to call builder.build again, this will fail the test
    def _fail_build(*args, **kwargs):
        raise AssertionError("builder.build should not be invoked when deps are provided")

    monkeypatch.setattr(FlujoRuntimeBuilder, "build", _fail_build)

    core = ExecutorCore(deps=deps, enable_cache=False)
    assert core._agent_runner is deps.agent_runner


def test_executor_core_uses_custom_builder():
    custom_runner = object()
    real_builder = FlujoRuntimeBuilder()
    called = {"count": 0}

    class _CustomBuilder:
        def build(self, **kwargs):
            called["count"] += 1
            # Delegate to real builder while forcing the custom runner
            kwargs["agent_runner"] = custom_runner
            kwargs.setdefault("enable_cache", False)
            return real_builder.build(**kwargs)

    core = ExecutorCore(builder=_CustomBuilder(), enable_cache=False)
    assert called["count"] == 1
    assert core._agent_runner is custom_runner

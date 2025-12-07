def test_import_core_and_dsl_without_cycles():
    # Smoke import test to catch circular-import regressions in core/dsl wiring.
    from flujo.application.core.executor_core import ExecutorCore  # noqa: F401
    from flujo.domain.dsl.step import Step  # noqa: F401
    from flujo.domain.dsl.pipeline import Pipeline  # noqa: F401


def test_construct_basic_pipeline():
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from typing import Any

    class DummyAgent:
        async def run(self, data, **kwargs):
            return data

    s1 = Step[Any, Any](name="s1", agent=DummyAgent())
    s2 = Step[Any, Any](name="s2", agent=DummyAgent())
    pipe = Pipeline.from_step(s1) >> s2
    assert len(pipe.steps) == 2


def test_import_runner_and_cli_helpers():
    # Ensure runner and CLI helpers import without triggering circulars
    from flujo.application.runner import Flujo  # noqa: F401
    from flujo.cli.helpers import display_pipeline_results  # noqa: F401


def test_import_interfaces_and_infra_hooks():
    # Ensure domain interfaces and default providers are resolvable without cycles
    from flujo.domain.interfaces import (  # noqa: F401
        get_settings_provider,
        get_telemetry_sink,
        get_skill_registry_provider,
        get_skill_resolver,
        get_skills_discovery,
        get_config_provider,
    )


def test_import_cli_entrypoints():
    # Smoke CLI import surfaces (validate circular import hardening)
    from flujo.cli import helpers_io  # noqa: F401
    from flujo.cli import helpers_validation  # noqa: F401
    from flujo.cli import run_command  # noqa: F401
    from flujo.cli import dev_commands_dev  # noqa: F401

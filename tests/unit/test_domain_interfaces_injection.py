from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import pytest

from flujo.domain import blueprint
from flujo.domain.interfaces import (
    get_config_provider,
    get_skills_discovery,
    get_telemetry_sink,
    set_default_config_provider,
    set_default_skills_discovery,
    set_default_telemetry_sink,
)
from flujo.architect import builder
from unittest import mock
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture()
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["flujo", "math"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["flujo", "math"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


class _FakeTelemetry:
    def __init__(self) -> None:
        self.events: List[Tuple[str, str]] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("info", str(message)))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("warning", str(message)))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("error", str(message)))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("debug", str(message)))

    def span(self, name: str, *args: Any, **kwargs: Any) -> Any:
        class _Span:
            def __enter__(self) -> "_Span":
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

        return _Span()


class _FakeConfig:
    blueprint_allowed_imports = ["math"]


class _FakeConfigProvider:
    def __init__(self) -> None:
        self.called = 0

    def load_config(self) -> Any:
        self.called += 1
        return _FakeConfig()


class _FakeSkillsDiscovery:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Optional[str]]] = []

    def load_catalog(self, base_dir: str) -> None:
        self.calls.append(("catalog", base_dir))

    def load_entry_points(self) -> None:
        self.calls.append(("entry_points", None))


@pytest.mark.asyncio
async def test_architect_builder_uses_injected_telemetry_sink() -> None:
    original_sink = get_telemetry_sink()
    fake_sink = _FakeTelemetry()
    try:
        set_default_telemetry_sink(fake_sink)
        await builder._goto("InjectedTelemetryTest")
    finally:
        set_default_telemetry_sink(original_sink)

    assert any("InjectedTelemetryTest" in msg for _level, msg in fake_sink.events)


def test_blueprint_import_enforces_allowlist_from_injected_config_provider() -> None:
    original_provider = get_config_provider()
    fake_provider = _FakeConfigProvider()
    try:
        set_default_config_provider(fake_provider)
        obj = blueprint.loader._import_object("math:pow")
        assert obj is math.pow
        with pytest.raises(blueprint.loader.BlueprintError):
            blueprint.loader._import_object("json.dumps")
    finally:
        set_default_config_provider(original_provider)

    # load_config should be invoked for each import attempt
    assert fake_provider.called >= 2


@pytest.mark.usefixtures("mock_allowed_imports")
def test_blueprint_loader_uses_injected_skills_discovery(tmp_path: Any) -> None:
    original_discovery = get_skills_discovery()
    fake_discovery = _FakeSkillsDiscovery()
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "- kind: step\n"
        "  name: Echo Input\n"
        "  agent:\n"
        "    id: flujo.builtins.stringify\n"
    )
    try:
        set_default_skills_discovery(fake_discovery)
        blueprint.loader.load_pipeline_blueprint_from_yaml(yaml_text, base_dir=str(tmp_path))
    finally:
        set_default_skills_discovery(original_discovery)

    assert ("catalog", str(tmp_path)) in fake_discovery.calls
    assert ("entry_points", None) in fake_discovery.calls


def test_skill_registry_provider_scopes_are_distinct() -> None:
    from flujo.infra.skill_registry import get_skill_registry_provider

    provider = get_skill_registry_provider()
    reg_default = provider.get_registry()
    reg_tenant = provider.get_registry(scope="tenant-a")

    reg_default.register("foo", lambda: "default")
    reg_tenant.register("foo", lambda: "tenant")

    assert reg_default.get("foo")["factory"]() == "default"
    assert reg_tenant.get("foo")["factory"]() == "tenant"
    # Default lookups should not bleed across scopes
    assert provider.get_registry().get("foo", scope="tenant-a") is None

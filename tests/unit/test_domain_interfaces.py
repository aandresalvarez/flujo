import contextlib
from typing import Any, Optional
from flujo.domain.interfaces import JSONObject

import pytest

from flujo.domain import interfaces
from flujo.domain.blueprint import loader


class _StubResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, skill_id: str, *, scope: Optional[str] = None) -> Optional[JSONObject]:
        self.calls.append(skill_id)
        return {"factory": lambda **kwargs: {"resolved": skill_id, "params": kwargs}}


class _StubTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("info", message))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("warning", message))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("error", message))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(("debug", message))

    @contextlib.contextmanager
    def span(self, name: str, *args: Any, **kwargs: Any):
        self.events.append(("span", name))
        yield self


def test_skill_resolver_is_injected_into_blueprint_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubResolver()
    monkeypatch.setattr(interfaces, "_DEFAULT_SKILL_RESOLVER", stub)

    resolved = loader._resolve_agent_entry({"id": "stub.skill", "params": {"x": 1}})

    assert stub.calls == ["stub.skill"]
    assert resolved == {"resolved": "stub.skill", "params": {"x": 1}}


def test_telemetry_sink_override(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTelemetry()
    monkeypatch.setattr(interfaces, "_DEFAULT_TELEMETRY_SINK", stub)

    sink = interfaces.get_telemetry_sink()
    sink.info("hello")
    sink.warning("warn")
    sink.debug("debug")
    with sink.span("span-op"):
        pass

    assert stub.events[0] == ("info", "hello")
    assert ("span", "span-op") in stub.events

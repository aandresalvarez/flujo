import pytest
from flujo.infra.registry import PipelineRegistry
from flujo.domain import Step, Pipeline


async def dummy(data: str) -> str:
    return data


def make_pipeline(name: str) -> Pipeline[str, str]:
    step = Step.from_callable(dummy, name=name)
    return step


def test_register_and_get() -> None:
    registry = PipelineRegistry()
    pipe = make_pipeline("a")
    registry.register(pipe, "p", "1.0.0")
    assert registry.get("p", "1.0.0") is pipe


def test_get_latest() -> None:
    registry = PipelineRegistry()
    p1 = make_pipeline("a")
    p2 = make_pipeline("b")
    registry.register(p1, "pipe", "1.0.0")
    registry.register(p2, "pipe", "2.0.0")
    latest = registry.get_latest("pipe")
    assert latest is p2


def test_invalid_version() -> None:
    registry = PipelineRegistry()
    with pytest.raises(ValueError):
        registry.register(make_pipeline("a"), "pipe", "notaversion")

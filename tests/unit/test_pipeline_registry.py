from flujo.registry import PipelineRegistry
from flujo.domain.dsl.step import Step


async def a(data: str) -> str:
    return data + "a"


async def b(data: str) -> str:
    return data + "b"


def test_register_and_get_latest() -> None:
    reg = PipelineRegistry()
    pipe_v1 = Step.from_callable(a, name="a") >> Step.from_callable(b, name="b")
    reg.register(pipe_v1, "pipe", "1.0.0")
    assert reg.get("pipe", "1.0.0") is pipe_v1
    assert reg.get_latest("pipe") is pipe_v1


def test_get_latest_none() -> None:
    reg = PipelineRegistry()
    assert reg.get_latest("missing") is None

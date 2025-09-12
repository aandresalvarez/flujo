from __future__ import annotations

import asyncio
from pydantic import BaseModel


def _get_factory(sid: str):
    from tests.conftest import get_registered_factory

    return get_registered_factory(sid)


def test_wrap_dict_simple() -> None:
    factory = _get_factory("flujo.builtins.wrap_dict")
    fn = factory()
    out = asyncio.run(fn("hello", key="greeting"))
    assert out == {"greeting": "hello"}


def test_ensure_object_passthrough_and_wrap() -> None:
    factory = _get_factory("flujo.builtins.ensure_object")
    fn = factory()

    # Dict passthrough
    d = {"a": 1}
    assert asyncio.run(fn(d)) is d

    # JSON string to object
    js = '{"x": 2}'
    assert asyncio.run(fn(js)) == {"x": 2}

    # Non-dict wraps under key
    assert asyncio.run(fn(3.14, key="pi")) == {"pi": 3.14}


def test_ensure_object_pydantic_model() -> None:
    class M(BaseModel):
        x: int
        y: str

    factory = _get_factory("flujo.builtins.ensure_object")
    fn = factory()
    out = asyncio.run(fn(M(x=1, y="b")))
    assert out == {"x": 1, "y": "b"}

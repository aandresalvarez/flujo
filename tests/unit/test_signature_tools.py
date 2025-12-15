from __future__ import annotations
from typing import Optional

import pytest

from flujo.exceptions import ConfigurationError
from flujo.signature_tools import analyze_signature, SignatureAnalysis
from flujo.domain.models import BaseModel
from flujo.domain.resources import AppResources


class Ctx(BaseModel):
    x: Optional[int] = None


class Res(AppResources):
    pass


def test_analyze_signature_simple_types_and_io_hints() -> None:
    async def f(x: str) -> int:  # noqa: D401
        return int(x)

    spec = analyze_signature(f)
    assert isinstance(spec, SignatureAnalysis)
    assert spec.needs_context is False
    assert spec.needs_resources is False
    assert spec.context_kw is None
    # Input/Output typing extracted from annotations
    assert spec.input_type is str
    assert spec.output_type is int


def test_invalid_context_annotation_missing_raises() -> None:
    async def f(x: str, *, context):  # type: ignore[no-untyped-def]
        return x

    with pytest.raises(ConfigurationError):
        analyze_signature(f)


def test_invalid_context_wrong_type_raises() -> None:
    async def f(x: str, *, context: int) -> str:
        return x

    with pytest.raises(ConfigurationError):
        analyze_signature(f)


def test_valid_context_basemodel_annotation_sets_flag() -> None:
    async def f(x: str, *, context: Ctx) -> str:
        return x

    spec = analyze_signature(f)
    assert spec.needs_context is True
    assert spec.context_kw == "context"


def test_invalid_context_union_without_basemodel_raises() -> None:
    from typing import Union

    async def f(x: str, *, context: Union[int, str]) -> str:
        return x

    with pytest.raises(ConfigurationError):
        analyze_signature(f)


def test_resources_annotation_validation() -> None:
    # Valid: direct subclass of AppResources
    async def f1(x: str, *, resources: Res) -> str:
        return x

    s1 = analyze_signature(f1)
    assert s1.needs_resources is True

    # Invalid: wrong type
    async def f3(x: str, *, resources: int) -> str:
        return x

    with pytest.raises(ConfigurationError):
        analyze_signature(f3)


def test_context_union_with_basemodel_component_is_accepted() -> None:
    # PEP 604 union syntax
    async def f(x: str, *, context: Ctx | int) -> str:
        return x

    spec = analyze_signature(f)
    assert spec.needs_context is True
    assert spec.context_kw == "context"


def test_resources_union_with_appresources_component_is_accepted() -> None:
    async def f(x: str, *, resources: Res | int) -> str:
        return x

    spec = analyze_signature(f)
    assert spec.needs_resources is True


def test_cache_returns_same_spec_instance_for_same_callable() -> None:
    async def f(x: int) -> int:
        return x

    a1 = analyze_signature(f)
    a2 = analyze_signature(f)
    # Should return the exact same cached object
    assert a1 is a2


def test_cache_returns_same_spec_instance_for_same_callable_object() -> None:
    # Functions should be cached and return the same instance
    async def g(y: int) -> int:
        return y

    s1 = analyze_signature(g)
    s2 = analyze_signature(g)
    assert s1 is s2


def test_cache_handles_non_weakref_callables_by_id_cache() -> None:
    class NoWeakRefCallable:
        __slots__ = ()

        def __call__(self, x: int) -> int:  # pragma: no cover - not executed by analyze
            return x

    obj = NoWeakRefCallable()
    spec1 = analyze_signature(obj)
    spec2 = analyze_signature(obj)
    assert spec1 is spec2


def test_defensive_type_hints_failure_falls_back(monkeypatch) -> None:
    import flujo.signature_tools as st

    async def f(x):  # type: ignore[no-untyped-def]
        return x

    # Force get_type_hints to raise to exercise defensive path
    monkeypatch.setattr(
        st, "get_type_hints", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    spec = st.analyze_signature(f)
    assert isinstance(spec, SignatureAnalysis)
    # Should not require injections and keep Any for types when hints fail
    assert spec.needs_context is False
    assert spec.needs_resources is False
    assert spec.input_type is not None
    assert spec.output_type is not None

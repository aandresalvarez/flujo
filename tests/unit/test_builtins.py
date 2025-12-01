from __future__ import annotations
from flujo.type_definitions.common import JSONObject

import asyncio
from typing import Any, List, AsyncIterator

import pytest
from pydantic import BaseModel

from flujo.builtins import extract_decomposed_steps


class _DecomposerModel(BaseModel):
    steps: List[JSONObject]


def test_extract_decomposed_steps_from_model() -> None:
    model = _DecomposerModel(steps=[{"step_name": "a"}, {"step_name": "b"}])
    out = asyncio.run(extract_decomposed_steps(model))
    assert isinstance(out, dict)
    assert "prepared_steps_for_mapping" in out
    assert isinstance(out["prepared_steps_for_mapping"], list)
    assert out["prepared_steps_for_mapping"][0]["step_name"] == "a"


def test_extract_decomposed_steps_from_dict() -> None:
    payload = {"steps": [{"step_name": "x"}]}
    out = asyncio.run(extract_decomposed_steps(payload))
    assert out["prepared_steps_for_mapping"][0]["step_name"] == "x"


async def _aggregate_plan(mapped, *, context=None):
    from flujo.builtins import _register_builtins  # ensure registration

    _register_builtins()
    # Access the registered factory and call the returned coroutine function directly
    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    factory = reg.get("flujo.builtins.aggregate_plan")["factory"]
    agg = factory() if callable(factory) else factory
    return await agg(mapped, context=context)


def test_aggregate_plan_combines_goal_and_steps() -> None:
    class Ctx(BaseModel):
        initial_prompt: str
        user_goal: str

    ctx = Ctx(initial_prompt="demo", user_goal="make a plan")
    mapped = [{"step_name": "a"}, {"step_name": "b"}]
    out = asyncio.run(_aggregate_plan(mapped, context=ctx))
    assert out["user_goal"] == "make a plan"
    assert [s["step_name"] for s in out["step_plans"]] == ["a", "b"]


# --- New builtins: web_search and extract_from_text ---


@pytest.mark.fast
def test_web_search_returns_simplified_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDGSAsync:
        async def __aenter__(self) -> "FakeDDGSAsync":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

        def text(self, query: str, max_results: int = 3) -> AsyncIterator[JSONObject]:
            async def _gen() -> AsyncIterator[JSONObject]:
                for i in range(max_results):
                    yield {
                        "title": f"Result {i} for {query}",
                        "href": f"https://example.com/{i}",
                        "body": f"Snippet {i}",
                    }

            return _gen()

    import flujo.builtins as builtins

    monkeypatch.setattr(builtins, "_DDGSAsync", FakeDDGSAsync, raising=True)

    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.web_search")
    fn = factory()

    results = asyncio.run(fn("test query", max_results=2))

    assert isinstance(results, list)
    assert len(results) == 2
    assert set(results[0].keys()) == {"title", "link", "snippet"}
    assert results[0]["link"].startswith("https://example.com/")


@pytest.mark.fast
def test_web_search_when_dependency_missing_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import flujo.builtins as builtins

    # Simulate missing dependency for both async and sync clients
    monkeypatch.setattr(builtins, "_DDGSAsync", None, raising=True)
    monkeypatch.setattr(builtins, "_DDGS_CLASS", None, raising=True)

    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.web_search")
    fn = factory()

    results = asyncio.run(fn("anything"))
    assert results == []


@pytest.mark.fast
def test_extract_from_text_returns_agent_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        async def run(self, *_args: Any, **_kwargs: Any) -> JSONObject:
            return {"ceo_name": "Jane Doe", "stock_price": 123.45}

    import flujo.builtins as builtins

    monkeypatch.setattr(builtins, "make_agent_async", lambda **_k: FakeAgent())

    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.extract_from_text")
    fn = factory()

    out = asyncio.run(
        fn(
            text="Apple CEO is Jane Doe. Stock is 123.45.",
            schema={
                "type": "object",
                "properties": {
                    "ceo_name": {"type": "string"},
                    "stock_price": {"type": "number"},
                },
                "required": ["ceo_name", "stock_price"],
            },
        )
    )

    assert out["ceo_name"] == "Jane Doe"
    assert out["stock_price"] == 123.45


@pytest.mark.fast
def test_extract_from_text_wraps_non_dict_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgentStr:
        async def run(self, *_args: Any, **_kwargs: Any) -> str:
            return "not a dict"

    import flujo.builtins as builtins

    monkeypatch.setattr(builtins, "make_agent_async", lambda **_k: FakeAgentStr())

    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.extract_from_text")
    fn = factory()

    out = asyncio.run(fn(text="x", schema={"type": "object"}))
    assert out == {"result": "not a dict"}


# --- New: check_user_confirmation skill ---


@pytest.mark.fast
def test_check_user_confirmation_affirmatives() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.check_user_confirmation")
    fn = factory()
    assert asyncio.run(fn("y")) == "approved"
    assert asyncio.run(fn("Y")) == "approved"
    assert asyncio.run(fn("yes")) == "approved"
    assert asyncio.run(fn("  Yes  ")) == "approved"


@pytest.mark.fast
def test_check_user_confirmation_negatives() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.check_user_confirmation")
    fn = factory()
    assert asyncio.run(fn("n")) == "denied"
    assert asyncio.run(fn("no")) == "denied"
    assert asyncio.run(fn("Nope")) == "denied"
    assert asyncio.run(fn("maybe")) == "denied"


@pytest.mark.fast
def test_check_user_confirmation_whitespace_defaults_to_approved() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.check_user_confirmation")
    fn = factory()
    assert asyncio.run(fn("")) == "approved"
    assert asyncio.run(fn("   \t\n")) == "approved"


# --- New small tests for passthrough and validate_yaml ---


@pytest.mark.fast
def test_passthrough_skill_identity() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.passthrough")
    fn = factory()
    payload = {"k": [1, 2, 3], "nested": {"a": 1}}
    out = asyncio.run(fn(payload))
    # Identity function: same object returned
    assert out is payload


@pytest.mark.fast
def test_validate_yaml_skill_handles_invalid_yaml_gracefully() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.validate_yaml")
    fn = factory()
    bad_yaml = "version: '0.1'\nsteps: ["  # malformed YAML
    report = asyncio.run(fn(bad_yaml))
    # Accept either ValidationReport object or report-like dict
    is_valid = False
    try:
        is_valid = bool(getattr(report, "is_valid", False))
    except Exception:
        is_valid = False
    if not is_valid and isinstance(report, dict):
        is_valid = bool(report.get("is_valid", False))
    assert is_valid is False


@pytest.mark.fast
def test_validate_yaml_skill_accepts_minimal_valid_yaml() -> None:
    # Use shared helper function from conftest.py
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.validate_yaml")
    fn = factory()
    good_yaml = 'version: "0.1"\nsteps: []\n'
    report = asyncio.run(fn(good_yaml))
    # Accept either object or dict forms
    is_valid = False
    try:
        is_valid = bool(getattr(report, "is_valid", False))
    except Exception:
        is_valid = False
    if not is_valid and isinstance(report, dict):
        is_valid = bool(report.get("is_valid", False))
    assert is_valid is True

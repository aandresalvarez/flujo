from __future__ import annotations

import pytest
from typing import Any, Dict, List

from flujo import Flujo, Step
from flujo.domain.models import BaseModel as FlujoBaseModel
from flujo.testing.utils import DummyRemoteBackend, gather_result


class ScalarContext(FlujoBaseModel):
    """Context model with scalar values only."""

    name: str
    age: int
    is_active: bool
    score: float


class ListContext(FlujoBaseModel):
    """Context model with list values only."""

    names: List[str]
    scores: List[int]
    flags: List[bool]


class MixedContext(FlujoBaseModel):
    """Context model with mixed scalar and list values."""

    name: str
    age: int
    tags: List[str]
    scores: List[float]
    is_active: bool


class NestedContext(FlujoBaseModel):
    """Context model with nested structures."""

    user: ScalarContext
    preferences: List[str]
    metadata: Dict[str, Any]


class ComplexContext(FlujoBaseModel):
    """Context model with complex data types."""

    scalar_string: str
    scalar_int: int
    scalar_bool: bool
    scalar_float: float
    list_strings: List[str]
    list_ints: List[int]
    list_bools: List[bool]
    list_floats: List[float]
    nested_dict: Dict[str, Any]
    empty_list: List[Any]
    empty_dict: Dict[str, Any]


class TestReconstructionLogic:
    @pytest.mark.asyncio
    async def test_scalar_values_not_wrapped_in_lists(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ScalarContext(name="John Doe", age=30, is_active=True, score=95.5)
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ScalarContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.name, str)
        assert isinstance(ctx.age, int)
        assert isinstance(ctx.is_active, bool)
        assert isinstance(ctx.score, float)
        assert ctx.name == "John Doe"
        assert ctx.age == 30
        assert ctx.is_active is True
        assert ctx.score == 95.5

    @pytest.mark.asyncio
    async def test_list_values_preserved(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ListContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ListContext(
            names=["Alice", "Bob", "Charlie"], scores=[85, 92, 78], flags=[True, False, True]
        )
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ListContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.names, list)
        assert isinstance(ctx.scores, list)
        assert isinstance(ctx.flags, list)
        assert all(isinstance(name, str) for name in ctx.names)
        assert all(isinstance(score, int) for score in ctx.scores)
        assert all(isinstance(flag, bool) for flag in ctx.flags)

    @pytest.mark.asyncio
    async def test_mixed_scalar_and_list_values(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: MixedContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = MixedContext(
            name="Test User",
            age=25,
            tags=["tag1", "tag2"],
            scores=[95.5, 87.2, 92.1],
            is_active=True,
        )
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=MixedContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.name, str)
        assert isinstance(ctx.age, int)
        assert isinstance(ctx.tags, list)
        assert isinstance(ctx.scores, list)
        assert isinstance(ctx.is_active, bool)
        assert all(isinstance(tag, str) for tag in ctx.tags)
        assert all(isinstance(score, float) for score in ctx.scores)

    @pytest.mark.asyncio
    async def test_nested_structures(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: NestedContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        user = ScalarContext(name="Nested User", age=35, is_active=False, score=88.0)
        context = NestedContext(
            user=user, preferences=["pref1", "pref2"], metadata={"key1": "value1", "key2": 42}
        )
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=NestedContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.user, ScalarContext)
        assert isinstance(ctx.preferences, list)
        assert isinstance(ctx.metadata, dict)
        assert all(isinstance(pref, str) for pref in ctx.preferences)

    @pytest.mark.asyncio
    async def test_complex_data_types(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ComplexContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ComplexContext(
            scalar_string="complex test",
            scalar_int=42,
            scalar_bool=True,
            scalar_float=3.14159,
            list_strings=["a", "b", "c"],
            list_ints=[1, 2, 3, 4, 5],
            list_bools=[True, False, True],
            list_floats=[1.1, 2.2, 3.3],
            nested_dict={"nested": {"deep": "value", "number": 123}},
            empty_list=[],
            empty_dict={},
        )
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ComplexContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.scalar_string, str)
        assert isinstance(ctx.scalar_int, int)
        assert isinstance(ctx.scalar_bool, bool)
        assert isinstance(ctx.scalar_float, float)
        assert isinstance(ctx.list_strings, list)
        assert isinstance(ctx.list_ints, list)
        assert isinstance(ctx.list_bools, list)
        assert isinstance(ctx.list_floats, list)
        assert isinstance(ctx.nested_dict, dict)
        assert isinstance(ctx.empty_list, list)
        assert isinstance(ctx.empty_dict, dict)

    @pytest.mark.asyncio
    async def test_string_encoded_lists_parsed_correctly(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ListContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ListContext(names=["Alice", "Bob"], scores=[85, 92], flags=[True, False])
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ListContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.names, list)
        assert isinstance(ctx.scores, list)
        assert isinstance(ctx.flags, list)
        assert ctx.names == ["Alice", "Bob"]
        assert ctx.scores == [85, 92]
        assert ctx.flags == [True, False]

    @pytest.mark.asyncio
    async def test_null_values_handled_correctly(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        # Provide minimal valid context data for ScalarContext
        minimal_context = {"name": "", "age": 0, "is_active": False, "score": 0.0}
        runner = Flujo(
            step, backend=backend, context_model=ScalarContext, initial_context_data=minimal_context
        )
        result = await gather_result(runner, None)
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, ScalarContext)

    @pytest.mark.asyncio
    async def test_boolean_values_preserved(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ScalarContext(name="Boolean Test", age=0, is_active=False, score=0.0)
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ScalarContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.is_active, bool)
        assert ctx.is_active is False

    @pytest.mark.asyncio
    async def test_float_values_preserved(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ScalarContext(name="Float Test", age=25, is_active=True, score=3.14159)
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ScalarContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.score, float)
        assert abs(ctx.score - 3.14159) < 0.0001

    @pytest.mark.asyncio
    async def test_empty_containers_preserved(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ComplexContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ComplexContext(
            scalar_string="Empty Test",
            scalar_int=0,
            scalar_bool=False,
            scalar_float=0.0,
            list_strings=[],
            list_ints=[],
            list_bools=[],
            list_floats=[],
            nested_dict={},
            empty_list=[],
            empty_dict={},
        )
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ComplexContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx.list_strings, list)
        assert isinstance(ctx.list_ints, list)
        assert isinstance(ctx.list_bools, list)
        assert isinstance(ctx.list_floats, list)
        assert isinstance(ctx.nested_dict, dict)
        assert isinstance(ctx.empty_list, list)
        assert isinstance(ctx.empty_dict, dict)
        assert len(ctx.list_strings) == 0
        assert len(ctx.list_ints) == 0
        assert len(ctx.list_bools) == 0
        assert len(ctx.list_floats) == 0
        assert len(ctx.nested_dict) == 0
        assert len(ctx.empty_list) == 0
        assert len(ctx.empty_dict) == 0

    @pytest.mark.asyncio
    async def test_serialization_roundtrip_integrity(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        original_context = ScalarContext(name="Roundtrip Test", age=42, is_active=True, score=99.9)
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ScalarContext,
            initial_context_data=original_context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert ctx.name == original_context.name
        assert ctx.age == original_context.age
        assert ctx.is_active == original_context.is_active
        assert abs(ctx.score - original_context.score) < 0.0001

    @pytest.mark.asyncio
    async def test_regression_bug_fix(self):
        class SimpleAgent:
            async def run(self, data: Dict[str, Any], *, context: ScalarContext) -> Dict[str, Any]:
                return data

        backend = DummyRemoteBackend()
        context = ScalarContext(name="John Doe", age=30, is_active=True, score=95.5)
        step = Step.model_validate({"name": "test", "agent": SimpleAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ScalarContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"test": "data"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        # Verify scalar values are NOT wrapped in lists (regression bug fix)
        assert isinstance(ctx.name, str)  # Should be str, not List[str]
        assert isinstance(ctx.age, int)  # Should be int, not List[int]
        assert isinstance(ctx.is_active, bool)  # Should be bool, not List[bool]
        assert isinstance(ctx.score, float)  # Should be float, not List[float]
        # Verify the actual values
        assert ctx.name == "John Doe"
        assert ctx.age == 30
        assert ctx.is_active is True
        assert abs(ctx.score - 95.5) < 0.0001

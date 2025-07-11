import pytest
import json
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel

from flujo.testing.utils import DummyRemoteBackend, gather_result
from flujo.domain.dsl.step import Step
from flujo.utils.serialization import safe_serialize
from flujo import Flujo


class EdgeCaseAgent:
    async def run(self, data: Any, **kwargs: Any) -> Any:
        return data


class EdgeCaseContext(BaseModel):
    empty_string: str = ""
    zero_int: int = 0
    zero_float: float = 0.0
    false_bool: bool = False
    empty_list: List[str] = []
    empty_dict: Dict[str, Any] = {}
    none_value: Optional[str] = None
    mixed_union: Union[str, int, bool] = "default"
    nested_empty: Dict[str, List[Dict[str, Any]]] = {}
    special_chars: str = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    unicode_string: str = "café résumé naïve"
    very_long_string: str = "x" * 1000
    very_large_number: int = 999999999999999999
    very_small_float: float = 0.000000000000001
    infinity_float: float = float("inf")
    negative_infinity_float: float = float("-inf")
    nan_float: float = float("nan")


class ComplexNestedContext(BaseModel):
    level1: Dict[str, Any]
    level2: Dict[str, Dict[str, Any]]
    level3: Dict[str, Dict[str, Dict[str, Any]]]
    mixed_levels: Dict[str, Union[str, int, List[Any], Dict[str, Any]]]


class TestSerializationEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_and_zero_values(self):
        backend = DummyRemoteBackend()
        context = EdgeCaseContext(
            empty_string="",
            zero_int=0,
            zero_float=0.0,
            false_bool=False,
            empty_list=[],
            empty_dict={},
            none_value=None,
        )
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=EdgeCaseContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"empty": ""})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, EdgeCaseContext)
        assert ctx.empty_string == ""
        assert ctx.zero_int == 0
        assert ctx.zero_float == 0.0
        assert ctx.false_bool is False
        assert ctx.empty_list == []
        assert ctx.empty_dict == {}
        assert ctx.none_value is None

    @pytest.mark.asyncio
    async def test_special_characters_and_unicode(self):
        backend = DummyRemoteBackend()
        context = EdgeCaseContext(
            special_chars="!@#$%^&*()_+-=[]{}|;':\",./<>?",
            unicode_string="café résumé naïve",
            very_long_string="x" * 1000,
        )
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=EdgeCaseContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"special": "!@#$%^&*()"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, EdgeCaseContext)
        assert ctx.special_chars == "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert ctx.unicode_string == "café résumé naïve"
        assert ctx.very_long_string == "x" * 1000

    @pytest.mark.asyncio
    async def test_extreme_numeric_values(self):
        backend = DummyRemoteBackend()
        context = EdgeCaseContext(
            very_large_number=999999999999999999,
            very_small_float=0.000000000000001,
            infinity_float=float("inf"),
            negative_infinity_float=float("-inf"),
            nan_float=float("nan"),
        )
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=EdgeCaseContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"extreme": 999999999999999999})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, EdgeCaseContext)
        assert ctx.very_large_number == 999999999999999999
        assert ctx.very_small_float == 0.000000000000001
        assert ctx.infinity_float == float("inf")
        assert ctx.negative_infinity_float == float("-inf")
        assert str(ctx.nan_float) == "nan"

    @pytest.mark.asyncio
    async def test_complex_nested_structures(self):
        backend = DummyRemoteBackend()
        context = ComplexNestedContext(
            level1={"key1": "value1", "key2": 42},
            level2={
                "nested1": {"inner1": "data1", "inner2": 123},
                "nested2": {"inner3": "data2", "inner4": 456},
            },
            level3={"deep1": {"deep2": {"deep3": "deep_value", "deep4": 789}}},
            mixed_levels={
                "string": "hello",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            },
        )
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=ComplexNestedContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"complex": {"nested": {"deep": "data"}}})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, ComplexNestedContext)
        assert ctx.level1 == {"key1": "value1", "key2": 42}
        assert ctx.level2["nested1"] == {"inner1": "data1", "inner2": 123}
        assert ctx.level2["nested2"] == {"inner3": "data2", "inner4": 456}
        assert ctx.level3["deep1"]["deep2"]["deep3"] == "deep_value"
        assert ctx.level3["deep1"]["deep2"]["deep4"] == 789
        assert ctx.mixed_levels["string"] == "hello"
        assert ctx.mixed_levels["number"] == 42
        assert ctx.mixed_levels["list"] == [1, 2, 3]
        assert ctx.mixed_levels["dict"] == {"nested": "value"}

    @pytest.mark.asyncio
    async def test_union_types(self):
        backend = DummyRemoteBackend()
        context = EdgeCaseContext(mixed_union="string_value")
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=EdgeCaseContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"union": "test"})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, EdgeCaseContext)
        assert ctx.mixed_union == "string_value"

    @pytest.mark.asyncio
    async def test_optional_types(self):
        backend = DummyRemoteBackend()
        context = EdgeCaseContext(none_value=None)
        step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
        runner = Flujo(
            step,
            backend=backend,
            context_model=EdgeCaseContext,
            initial_context_data=context.model_dump(),
        )
        result = await gather_result(runner, {"optional": None})
        ctx = result.final_pipeline_context
        assert ctx is not None
        assert isinstance(ctx, EdgeCaseContext)
        assert ctx.none_value is None

    @pytest.mark.asyncio
    async def test_string_encoded_lists_edge_cases(self):
        backend = DummyRemoteBackend()
        test_cases = [
            ("[]", []),
            ("[1, 2, 3]", [1, 2, 3]),
            ("['a', 'b', 'c']", ["a", "b", "c"]),
            ("[True, False, True]", [True, False, True]),
            ("[1.1, 2.2, 3.3]", [1.1, 2.2, 3.3]),
            ("[1, 'mixed', True, 3.14]", [1, "mixed", True, 3.14]),
        ]
        for string_list, expected_list in test_cases:
            context = EdgeCaseContext()
            step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
            runner = Flujo(
                step,
                backend=backend,
                context_model=EdgeCaseContext,
                initial_context_data=context.model_dump(),
            )
            result = await gather_result(runner, {"list_data": string_list})
            ctx = result.final_pipeline_context
            assert ctx is not None
            assert isinstance(ctx, EdgeCaseContext)

    @pytest.mark.asyncio
    async def test_malformed_string_encoded_lists(self):
        backend = DummyRemoteBackend()
        malformed_cases = [
            "[1, 2, 3",
            "1, 2, 3]",
            "[1, 2, 3,]",
            "[1, 2, 3, ]",
            "not_a_list",
            "",
        ]
        for malformed_list in malformed_cases:
            context = EdgeCaseContext()
            step = Step.model_validate({"name": "test", "agent": EdgeCaseAgent()})
            runner = Flujo(
                step,
                backend=backend,
                context_model=EdgeCaseContext,
                initial_context_data=context.model_dump(),
            )
            result = await gather_result(runner, {"malformed": malformed_list})
            ctx = result.final_pipeline_context
            assert ctx is not None
            assert isinstance(ctx, EdgeCaseContext)

    @pytest.mark.asyncio
    async def test_serialization_roundtrip_edge_cases(self):
        # This test is for serialization, not context, so keep as is
        edge_case_data = {
            "empty_string": "",
            "zero_values": {"int": 0, "float": 0.0, "bool": False},
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "unicode": "café résumé naïve",
            "extreme_numbers": {
                "large": 999999999999999999,
                "small": 0.000000000000001,
                "infinity": float("inf"),
                "negative_infinity": float("-inf"),
                "nan": float("nan"),
            },
            "nested_empty": {"level1": {"level2": {"level3": {}}}},
            "mixed_types": {
                "string": "hello",
                "number": 42,
                "boolean": True,
                "list": [1, "mixed", True, 3.14],
                "dict": {"nested": "value"},
            },
        }
        serialized = safe_serialize(edge_case_data)
        deserialized = json.loads(json.dumps(serialized))
        assert deserialized["empty_string"] == ""
        assert deserialized["zero_values"]["int"] == 0
        assert deserialized["zero_values"]["float"] == 0.0
        assert deserialized["zero_values"]["bool"] is False
        assert deserialized["special_chars"] == "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert deserialized["unicode"] == "café résumé naïve"
        assert deserialized["extreme_numbers"]["large"] == 999999999999999999
        assert deserialized["extreme_numbers"]["small"] == 0.000000000000001
        assert deserialized["extreme_numbers"]["infinity"] == float("inf")
        assert deserialized["extreme_numbers"]["negative_infinity"] == float("-inf")
        assert str(deserialized["extreme_numbers"]["nan"]) == "nan"
        assert deserialized["nested_empty"]["level1"]["level2"]["level3"] == {}
        assert deserialized["mixed_types"]["string"] == "hello"
        assert deserialized["mixed_types"]["number"] == 42
        assert deserialized["mixed_types"]["boolean"] is True
        assert deserialized["mixed_types"]["list"] == [1, "mixed", True, 3.14]
        assert deserialized["mixed_types"]["dict"] == {"nested": "value"}

    @pytest.mark.asyncio
    async def test_future_bug_prevention(self):
        backend = DummyRemoteBackend()
        test_scenarios = [
            {"name": "string_value", "age": 25, "is_active": True, "score": 95.5},
            {"tags": ["tag1", "tag2"], "scores": [85, 92, 78]},
            {"name": "John", "age": 30, "tags": ["tag1", "tag2"], "scores": [85, 92]},
            {"user": {"name": "John", "age": 30}, "preferences": ["pref1", "pref2"]},
            {"empty_list": [], "empty_dict": {}},
            {"optional_field": None},
            {"flag1": True, "flag2": False},
            {"price": 29.99, "rating": 4.5},
        ]
        for i, scenario in enumerate(test_scenarios):

            class TestContext(BaseModel):
                data: Dict[str, Any]

            context = TestContext(data=scenario)
            step = Step.model_validate({"name": f"test_{i}", "agent": EdgeCaseAgent()})
            runner = Flujo(
                step,
                backend=backend,
                context_model=TestContext,
                initial_context_data=context.model_dump(),
            )
            result = await gather_result(runner, {"scenario": i})
            ctx = result.final_pipeline_context
            assert ctx is not None
            assert isinstance(ctx, TestContext)
            assert ctx.data == scenario
            for key, value in scenario.items():
                if isinstance(value, (str, int, bool, float)) and not isinstance(value, list):
                    assert isinstance(ctx.data[key], type(value)), (
                        f"Scalar value {key}={value} was incorrectly wrapped in a list"
                    )
                elif isinstance(value, list):
                    assert isinstance(ctx.data[key], list), (
                        f"List value {key}={value} was not preserved as a list"
                    )

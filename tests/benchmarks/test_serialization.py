import json
import pytest
from flujo.domain.models import Candidate, Checklist, ChecklistItem

complex_candidate = Candidate(
    solution="This is a very long solution string...",
    score=0.85,
    checklist=Checklist(
        items=[
            ChecklistItem(description=f"Item {i}", passed=True, feedback="Looks good")
            for i in range(20)
        ]
    ),
)


@pytest.mark.benchmark(group="serialization")
def test_benchmark_pydantic_orjson_dumps(benchmark):
    benchmark(complex_candidate.model_dump_json)


@pytest.mark.benchmark(group="serialization")
def test_benchmark_stdlib_json_dumps(benchmark):
    data = complex_candidate.model_dump()
    benchmark(json.dumps, data)


@pytest.mark.benchmark(group="deserialization")
def test_benchmark_pydantic_orjson_loads(benchmark):
    json_str = complex_candidate.model_dump_json()
    benchmark(Candidate.model_validate_json, json_str)


@pytest.mark.benchmark(group="deserialization")
def test_benchmark_stdlib_json_loads(benchmark):
    json_str = complex_candidate.model_dump_json()
    benchmark(json.loads, json_str)

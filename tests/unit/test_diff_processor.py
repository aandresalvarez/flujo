import pytest

from flujo.processors.diff import DiffProcessor


@pytest.mark.asyncio
async def test_diff_processor_add_replace_and_list_extend() -> None:
    proc = DiffProcessor()
    result = await proc.process(
        {
            "before": {"a": 1, "b": {"c": 2}, "list": [1, 2]},
            "after": {"a": 1, "b": {"c": 3}, "list": [1, 2, 3], "d": "x"},
        }
    )

    assert result["patch"] == [
        {"op": "add", "path": "/d", "value": "x"},
        {"op": "replace", "path": "/b/c", "value": 3},
        {"op": "add", "path": "/list/2", "value": 3},
    ]


@pytest.mark.asyncio
async def test_diff_processor_list_remove() -> None:
    proc = DiffProcessor()
    result = await proc.process(([1, 2, 3], [1, 2]))
    assert result["patch"] == [{"op": "remove", "path": "/2"}]

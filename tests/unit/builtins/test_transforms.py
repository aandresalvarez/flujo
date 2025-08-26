from __future__ import annotations

import pytest


def _get_callable(skill_id: str):
    # Ensure builtins registered
    import importlib
    import flujo.builtins as builtins

    importlib.reload(builtins)

    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    entry = reg.get(skill_id)
    assert entry is not None, f"Skill not registered: {skill_id}"
    factory = entry.get("factory")
    assert callable(factory)
    fn = factory()
    assert callable(fn)
    return fn


@pytest.mark.asyncio
async def test_to_csv_basic_and_headers():
    to_csv = _get_callable("flujo.builtins.to_csv")

    rows = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4, "c": 5},  # extra column 'c' ignored when headers provided
    ]

    # Deterministic headers derived (sorted union) when not provided
    csv_text = await to_csv(rows)
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    # sorted union => a,b,c
    assert lines[0] == "a,b,c"
    assert lines[1] == "1,2,"
    assert lines[2] == "3,4,5"

    # Custom headers order is respected
    csv_text2 = await to_csv(rows, headers=["b", "a"])  # omit 'c'
    lines2 = [ln for ln in csv_text2.strip().splitlines() if ln]
    assert lines2[0] == "b,a"
    assert lines2[1] == "2,1"
    assert lines2[2] == "4,3"


@pytest.mark.asyncio
async def test_aggregate_sum_avg_count():
    aggregate = _get_callable("flujo.builtins.aggregate")

    data = [
        {"price": 1.5},
        {"price": 2},
        {"price": None},
        {"price": "n/a"},
        {"other": 10},
    ]

    s = await aggregate(data, operation="sum", field="price")
    assert s == pytest.approx(3.5)

    avg = await aggregate(data, operation="avg", field="price")
    assert avg == pytest.approx(1.75)

    cnt_field = await aggregate(data, operation="count", field="price")
    # count entries with field present and not None
    assert cnt_field == 3

    cnt_all = await aggregate(data, operation="count")
    assert cnt_all == len(data)


@pytest.mark.asyncio
async def test_select_fields_projection_and_rename():
    select_fields = _get_callable("flujo.builtins.select_fields")

    obj = {"a": 1, "b": 2, "c": 3}
    out = await select_fields(obj, include=["b", "a"], rename={"a": "alpha"})
    assert out == {"b": 2, "alpha": 1}

    rows = [
        {"x": 10, "y": 20},
        {"x": 30, "y": 40, "z": 50},
    ]
    out_list = await select_fields(rows, rename={"x": "X"})
    assert out_list == [{"X": 10, "y": 20}, {"X": 30, "y": 40, "z": 50}]


@pytest.mark.asyncio
async def test_flatten_one_level_and_passthrough():
    flatten = _get_callable("flujo.builtins.flatten")

    items = [[1, 2], [3], [], (4, 5), 6]
    out = await flatten(items)
    assert out == [1, 2, 3, 4, 5, 6]

    # Non-list input -> empty list per current semantics
    out2 = await flatten(123)
    assert out2 == []

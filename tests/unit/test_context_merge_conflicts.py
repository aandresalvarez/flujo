import pytest
from pydantic import BaseModel

from flujo.utils.context import safe_merge_context_updates
from flujo.domain.dsl.step import MergeStrategy
from flujo.exceptions import ConfigurationError


class _Ctx(BaseModel):
    value: str
    other: int = 0


def test_safe_merge_conflict_errors_on_context_update():
    target = _Ctx(value="base", other=1)
    source = _Ctx(value="branchA", other=2)

    with pytest.raises(ConfigurationError, match="Merge conflict for key 'value'"):
        safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.CONTEXT_UPDATE)


def test_safe_merge_conflict_errors_on_error_on_conflict():
    target = _Ctx(value="base", other=1)
    source = _Ctx(value="branchB", other=2)

    with pytest.raises(ConfigurationError, match="Merge conflict for key 'value'"):
        safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.ERROR_ON_CONFLICT)


def test_safe_merge_conflict_allowed_on_overwrite():
    target = _Ctx(value="base", other=1)
    source = _Ctx(value="branchC", other=2)

    # Should not raise for OVERWRITE (conflict detection is disabled)
    assert safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.OVERWRITE)


def test_safe_merge_no_error_when_values_equal():
    target = _Ctx(value="same", other=1)
    source = _Ctx(value="same", other=1)

    assert safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.CONTEXT_UPDATE)

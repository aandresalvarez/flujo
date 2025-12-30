import pytest

from flujo.domain.dsl.step import MergeStrategy
from flujo.exceptions import ConfigurationError
from flujo.utils.context import safe_merge_context_updates
from tests.test_types.fixtures import create_test_context


def test_safe_merge_conflict_errors_on_context_update():
    target = create_test_context(counter=1)
    source = create_test_context(counter=2)

    with pytest.raises(ConfigurationError, match="Merge conflict for key 'counter'"):
        safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.CONTEXT_UPDATE)


def test_safe_merge_conflict_errors_on_error_on_conflict():
    target = create_test_context(counter=1)
    source = create_test_context(counter=2)

    with pytest.raises(ConfigurationError, match="Merge conflict for key 'counter'"):
        safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.ERROR_ON_CONFLICT)


def test_safe_merge_conflict_allowed_on_overwrite():
    target = create_test_context(counter=1)
    source = create_test_context(counter=2)

    # Should not raise for OVERWRITE (conflict detection is disabled)
    assert safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.OVERWRITE)


def test_safe_merge_no_error_when_values_equal():
    target = create_test_context(counter=1)
    source = create_test_context(counter=1)

    assert safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.CONTEXT_UPDATE)


def test_safe_merge_overwrite_overwrites_nested_primitives():
    target = create_test_context(scratchpad={"payload": {"count": 5, "flag": True}})
    source = create_test_context(scratchpad={"payload": {"count": 2, "flag": False}})

    safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.OVERWRITE)

    assert target.scratchpad == {"payload": {"count": 2, "flag": False}}


def test_safe_merge_nested_conflict_errors_on_error_on_conflict():
    target = create_test_context(scratchpad={"payload": {"key": 1}})
    source = create_test_context(scratchpad={"payload": {"key": 2}})

    with pytest.raises(ConfigurationError, match=r"scratchpad\.payload\.key"):
        safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.ERROR_ON_CONFLICT)


def test_safe_merge_keep_first_preserves_existing_dict_keys():
    target = create_test_context(scratchpad={"a": {"b": 1}})
    source = create_test_context(scratchpad={"a": {"b": 2, "c": 3}, "d": 4})

    safe_merge_context_updates(target, source, merge_strategy=MergeStrategy.KEEP_FIRST)

    assert target.scratchpad == {"a": {"b": 1, "c": 3}, "d": 4}

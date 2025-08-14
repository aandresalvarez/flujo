from typing import List, Dict

from flujo.application.core.context_manager import _types_compatible


def test_list_generic_argument_incompatible() -> None:
    assert _types_compatible(List[str], List[int]) is False


def test_list_generic_argument_compatible() -> None:
    assert _types_compatible(List[str], List[str]) is True


def test_dict_generic_arguments_incompatible() -> None:
    assert _types_compatible(Dict[str, int], Dict[str, str]) is False


def test_dict_generic_arguments_compatible() -> None:
    assert _types_compatible(Dict[str, int], Dict[str, int]) is True


def test_plain_vs_parametric_considered_compatible() -> None:
    # By design we allow parametric vs raw container as compatible for now
    assert _types_compatible(List[str], list) is True
    assert _types_compatible(list, List[str]) is True

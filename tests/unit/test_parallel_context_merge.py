from pydantic import BaseModel

from flujo.application.core.context_manager import ContextManager


class _Ctx(BaseModel):
    data: dict[str, int] = {}


def test_parallel_context_merge_deep_dict() -> None:
    """Context merge should deep-merge dict fields rather than overwrite."""

    main = _Ctx(data={"key_a": 1})
    branch = _Ctx(data={"key_b": 2})

    merged = ContextManager.merge(main, branch)
    assert merged is not None
    assert merged.data == {"key_a": 1, "key_b": 2}

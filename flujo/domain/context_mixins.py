from __future__ import annotations

from typing import TypeVar, Type

from .models import PipelineContext

CtxT = TypeVar("CtxT", bound="BaseContext")


class BaseContext(PipelineContext):
    """Base context for typed pipeline contexts."""

    # model_config inherited from BaseModel


def typed_context(context_cls: Type[CtxT]) -> Type[CtxT]:
    """
    Declare a typed context class for pipelines.

    Usage:
        class MyContext(BaseContext):
            counter: int = 0
            result: str | None = None

        TypedCtx = typed_context(MyContext)
    """

    if not issubclass(context_cls, BaseContext):
        raise TypeError("typed_context expects a subclass of BaseContext")
    return context_cls


__all__ = ["BaseContext", "typed_context"]

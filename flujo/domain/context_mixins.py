from __future__ import annotations

from typing import TypeVar, Type

from .models import PipelineContext

CtxT = TypeVar("CtxT", bound="BaseContext")


class BaseContext(PipelineContext):
    """Base context that reserves scratchpad for framework metadata only."""

    class Config:
        arbitrary_types_allowed = True

    def forbid_scratchpad_user_data(self) -> None:
        if not isinstance(self.scratchpad, dict):
            return
        # Allow only reserved framework keys; others should be typed fields instead
        reserved = {"state", "status", "pause_message", "paused_step_input", "background_error"}
        extra_keys = set(self.scratchpad.keys()) - reserved
        if extra_keys:
            raise ValueError(
                f"scratchpad keys not allowed for user data: {sorted(extra_keys)}; "
                "declare typed fields on the context model instead."
            )


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

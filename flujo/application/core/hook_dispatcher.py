from __future__ import annotations

import inspect
from typing import Any, Sequence, get_type_hints, get_origin, get_args, Union, Literal, cast

from ...infra import telemetry
from ...domain.events import (
    HookPayload,
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)
from ...domain.types import HookCallable
from ...exceptions import PipelineAbortSignal

__all__ = ["_dispatch_hook", "_should_dispatch", "_get_hook_params"]


def _get_hook_params(hook: HookCallable) -> tuple[list[inspect.Parameter], dict[str, Any]]:
    try:
        sig = inspect.signature(hook)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        params = []
    try:
        hints = get_type_hints(hook)
    except Exception:
        hints = {}
    return params, hints


def _should_dispatch(annotation: Any, payload: HookPayload) -> bool:
    if annotation is inspect.Signature.empty:
        return True
    origin = get_origin(annotation)
    if origin is Union:
        return any(isinstance(payload, t) for t in get_args(annotation))
    if isinstance(annotation, type):
        return isinstance(payload, annotation)
    return True


async def _dispatch_hook(
    hooks: Sequence[HookCallable],
    event_name: Literal[
        "pre_run",
        "post_run",
        "pre_step",
        "post_step",
        "on_step_failure",
    ],
    **kwargs: Any,
) -> None:
    payload_map: dict[str, type[HookPayload]] = {
        "pre_run": PreRunPayload,
        "post_run": PostRunPayload,
        "pre_step": PreStepPayload,
        "post_step": PostStepPayload,
        "on_step_failure": OnStepFailurePayload,
    }
    PayloadCls = payload_map.get(event_name)
    if PayloadCls is None:
        return

    payload = PayloadCls(event_name=cast(Any, event_name), **kwargs)

    for hook in hooks:
        try:
            should_call = True
            try:
                params, hints = _get_hook_params(hook)
                if params:
                    ann = hints.get(params[0].name, params[0].annotation)
                    should_call = _should_dispatch(ann, payload)
            except Exception as e:
                name = getattr(hook, "__name__", str(hook))
                telemetry.logfire.error(f"Error in hook '{name}': {e}")
            if should_call:
                await hook(payload)
        except PipelineAbortSignal:
            raise
        except Exception as e:
            name = getattr(hook, "__name__", str(hook))
            telemetry.logfire.error(f"Error in hook '{name}': {e}")

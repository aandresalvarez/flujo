from __future__ import annotations

import inspect
from typing import (
    Any,
    Sequence,
    get_type_hints,
    get_origin,
    get_args,
    Union,
    Literal,
    cast,
)

import logging
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

# Get the flujo logger for proper test capture
_flujo_logger = logging.getLogger("flujo")
# Also get the root logger to ensure pytest caplog can capture our messages
_root_logger = logging.getLogger()


def _get_hook_params(
    hook: HookCallable,
) -> tuple[list[inspect.Parameter], dict[str, Any]]:
    """Extract parameter information from a hook function."""
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
    """Determine if a hook should be dispatched based on its annotation."""
    if annotation is inspect.Signature.empty:
        return True
    origin = get_origin(annotation)
    if origin is Union:
        return any(isinstance(payload, t) for t in get_args(annotation))
    if isinstance(annotation, type):
        return isinstance(payload, annotation)
    return True


def _log_hook_error(msg: str) -> None:
    """Log hook errors with fallback to ensure visibility in test environments.

    This function ensures that hook errors are always logged, even in test environments
    where the primary logging mechanism might not be configured.
    """
    # Primary logging through telemetry
    try:
        telemetry.logfire.error(msg)
    except Exception:
        pass  # Silently fail telemetry logging

    # Secondary logging to standard Python logging (captured by pytest caplog)
    try:
        _flujo_logger.error(msg)
        # Also log as warning to increase visibility
        _flujo_logger.warning(msg)
        # Log to root logger to ensure pytest caplog can capture it
        _root_logger.error(msg)
        _root_logger.warning(msg)
    except Exception:
        pass  # Silently fail if standard logging fails

    # Final fallback: print to stderr if all else fails
    # This ensures the error is visible even in minimal test environments
    try:
        import sys

        print(f"HOOK ERROR: {msg}", file=sys.stderr)
    except Exception:
        pass  # Absolute last resort


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
    """Dispatch hooks for the given event, handling errors gracefully."""
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
                msg = f"Error in hook '{name}': {e}"
                _log_hook_error(msg)

            if should_call:
                await hook(payload)
        except PipelineAbortSignal:
            raise
        except Exception as e:
            name = getattr(hook, "__name__", str(hook))
            msg = f"Error in hook '{name}': {e}"
            _log_hook_error(msg)

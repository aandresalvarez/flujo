from __future__ import annotations

import inspect
import os
from typing import (
    Sequence,
    get_type_hints,
    get_origin,
    get_args,
    Union,
    Literal,
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
IS_BACKGROUND_FALLBACK_RUN_ID_MARKER = "_bg_"

# Get the flujo logger for proper test capture
_flujo_logger = logging.getLogger("flujo")
# Also get the root logger to ensure pytest caplog can capture our messages
_root_logger = logging.getLogger()


def _get_hook_params(
    hook: HookCallable,
) -> tuple[list[inspect.Parameter], dict[str, object]]:
    """Extract parameter information from a hook function."""
    try:
        sig = inspect.signature(hook)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        params = []
    try:
        hints: dict[str, object] = dict(get_type_hints(hook))
    except Exception:
        hints = {}
    return params, hints


def _should_dispatch(annotation: object, payload: HookPayload) -> bool:
    """Determine if a hook should be dispatched based on its annotation."""
    if annotation is inspect.Signature.empty:
        return True
    origin = get_origin(annotation)
    if origin is Union:
        return any(isinstance(payload, t) for t in get_args(annotation))
    if isinstance(annotation, type):
        return isinstance(payload, annotation)
    return True


def _is_background_context(context: object) -> bool:
    """Best-effort determination of background execution based on context."""
    try:
        if bool(getattr(context, "is_background_task", False)):
            return True
    except Exception:
        pass

    try:
        run_id = getattr(context, "run_id", None)
        if isinstance(run_id, str) and IS_BACKGROUND_FALLBACK_RUN_ID_MARKER in run_id:
            return True
    except Exception:
        pass

    return False


def _is_background_step(step: object) -> bool:
    """Detect if a step is configured for background execution."""
    try:
        cfg = getattr(step, "config", None)
        if cfg is not None and getattr(cfg, "execution_mode", "sync") == "background":
            return True
    except Exception:
        return False
    return False


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


def _should_reraise_hook_errors() -> bool:
    """Return True only when explicitly requested to surface hook bugs."""
    try:
        flag = os.getenv("FLUJO_STRICT_HOOKS")
        return str(flag).lower() in {"1", "true", "yes"}
    except Exception:
        return False


async def _dispatch_hook(
    hooks: Sequence[HookCallable],
    event_name: Literal[
        "pre_run",
        "post_run",
        "pre_step",
        "post_step",
        "on_step_failure",
    ],
    **kwargs: object,
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

    # Derive is_background if not explicitly provided
    if "is_background" not in kwargs:
        ctx = kwargs.get("context")
        step_obj = kwargs.get("step")
        kwargs["is_background"] = _is_background_context(ctx) or _is_background_step(step_obj)

    payload = PayloadCls(event_name=event_name, **kwargs)

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
            if _should_reraise_hook_errors():
                raise

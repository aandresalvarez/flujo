"""Compatibility shim for `step_history_tracker` (moved to core/state/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.state.step_history_tracker import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.state.step_history_tracker")
_sys.modules[__name__] = _module

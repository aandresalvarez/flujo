"""Compatibility shim for `executor_helpers` (moved to core/execution/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.execution.executor_helpers import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.executor_helpers")
_sys.modules[__name__] = _module

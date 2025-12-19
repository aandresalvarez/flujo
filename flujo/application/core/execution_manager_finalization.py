"""Compatibility shim for `execution_manager_finalization` (moved to core/execution/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.execution.execution_manager_finalization import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.execution_manager_finalization")
_sys.modules[__name__] = _module

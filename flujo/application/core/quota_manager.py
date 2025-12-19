"""Compatibility shim for `quota_manager` (moved to core/runtime/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.runtime.quota_manager import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.runtime.quota_manager")
_sys.modules[__name__] = _module

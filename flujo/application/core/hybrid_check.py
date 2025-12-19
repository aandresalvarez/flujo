"""Compatibility shim for `hybrid_check` (moved to core/support/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.support.hybrid_check import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.support.hybrid_check")
_sys.modules[__name__] = _module

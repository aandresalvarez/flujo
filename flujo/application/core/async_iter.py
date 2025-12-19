"""Compatibility shim for `async_iter` (moved to core/support/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.support.async_iter import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.support.async_iter")
_sys.modules[__name__] = _module

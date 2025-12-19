"""Compatibility shim for `usage_messages` (moved to core/runtime/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.runtime.usage_messages import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.runtime.usage_messages")
_sys.modules[__name__] = _module

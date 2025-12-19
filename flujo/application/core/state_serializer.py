"""Compatibility shim for `state_serializer` (moved to core/state/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.state.state_serializer")
_sys.modules[__name__] = _module

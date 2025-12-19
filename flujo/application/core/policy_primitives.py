"""Compatibility shim for `policy_primitives` (moved to core/policy/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.policy.policy_primitives")
_sys.modules[__name__] = _module

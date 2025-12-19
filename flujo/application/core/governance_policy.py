"""Compatibility shim for `governance_policy` (moved to core/policy/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.policy.governance_policy")
_sys.modules[__name__] = _module

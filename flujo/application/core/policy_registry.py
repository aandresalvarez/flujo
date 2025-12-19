"""Compatibility shim for `policy_registry` (moved to core/policy/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.policy.policy_registry import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.policy.policy_registry")
_sys.modules[__name__] = _module

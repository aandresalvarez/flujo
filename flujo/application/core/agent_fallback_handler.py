"""Compatibility shim for `agent_fallback_handler` (moved to core/agents/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.agents.agent_fallback_handler import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.agents.agent_fallback_handler")
_sys.modules[__name__] = _module

"""Compatibility shim for `agent_execution_runner` (moved to core/agents/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.agents.agent_execution_runner")
_sys.modules[__name__] = _module

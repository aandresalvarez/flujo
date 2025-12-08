"""Shared context variables used across ExecutorCore modules.

This module defines context variables that need to be shared between modules
to avoid circular imports and ensure a single instance is used.
"""

from __future__ import annotations
from contextvars import ContextVar
from typing import Optional

# Cache enable override to avoid mutating shared core state (used by loop/state-machine policies).
_CACHE_OVERRIDE: ContextVar[Optional[bool]] = ContextVar("CACHE_OVERRIDE", default=None)

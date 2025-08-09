"""
Backward compatibility import for DeterministicRepairProcessor.

This module has been moved to flujo.agents.repair as part of FSD-005.3
to isolate specialized repair logic.
"""

import warnings

# Import from the new location
from ..agents.repair import DeterministicRepairProcessor, MAX_LITERAL_EVAL_SIZE

# Issue a deprecation warning
warnings.warn(
    "Importing DeterministicRepairProcessor from flujo.processors.repair is deprecated. "
    "Import from flujo.agents.repair instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DeterministicRepairProcessor", "MAX_LITERAL_EVAL_SIZE"]
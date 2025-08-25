"""
Stable CLI exit codes for Flujo commands.

These codes are intended for CI and scripts to reliably detect outcomes.

- EX_OK (0): success
- EX_RUNTIME_ERROR (1): unhandled runtime error or execution failure
- EX_CONFIG_ERROR (2): configuration/settings error
- EX_IMPORT_ERROR (3): import/module resolution error
- EX_VALIDATION_FAILED (4): validation completed but reported errors (strict mode)
- EX_SIGINT (130): interrupted by user (Ctrl+C)
"""

EX_OK = 0
EX_RUNTIME_ERROR = 1
EX_CONFIG_ERROR = 2
EX_IMPORT_ERROR = 3
EX_VALIDATION_FAILED = 4
EX_SIGINT = 130

__all__ = [
    "EX_OK",
    "EX_RUNTIME_ERROR",
    "EX_CONFIG_ERROR",
    "EX_IMPORT_ERROR",
    "EX_VALIDATION_FAILED",
    "EX_SIGINT",
]

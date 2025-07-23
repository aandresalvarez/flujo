# CLI tests in this file are now skipped.
# Reason: CLI integration tests have been moved to standalone scripts in tests/cli_integration/ for robustness and reliability.
# See those scripts for current CLI test coverage.
# This file is retained for reference but all tests are skipped to avoid confusion and noise in test output.

import pytest

pytest.skip(
    "CLI tests have moved to tests/cli_integration/. This file is skipped to avoid confusion.",
    allow_module_level=True,
)

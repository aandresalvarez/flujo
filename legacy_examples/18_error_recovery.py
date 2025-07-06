"""Example: Recovering from errors with a fallback step.

See docs/cookbook/error_recovery.md for details.
"""

from flujo import Flujo, Step
from flujo.testing.utils import StubAgent

# Primary step fails; fallback step provides a result
primary = Step("primary", StubAgent(["fail"]), max_retries=1)
backup = Step("backup", StubAgent(["ok"]))
primary.fallback(backup)

runner = Flujo(primary)

if __name__ == "__main__":
    result = runner.run("data")
    print(f"Pipeline output: {result.step_history[0].output}")
    print(
        f"Fallback triggered: {result.step_history[0].metadata_['fallback_triggered']}"
    )

"""Example: Initializing telemetry with custom settings.

See docs/cookbook/telemetry_setup.md for details.
"""

from flujo import Flujo, Step, init_telemetry
from flujo.infra.settings import Settings

init_telemetry(
    Settings(
        telemetry_export_enabled=True,
        otlp_export_enabled=True,
        otlp_endpoint="https://otlp.example.com",
    )
)

pipeline = Step.from_mapper(lambda x: x.upper(), name="upper")
runner = Flujo(pipeline)

if __name__ == "__main__":
    runner.run("hello")
    print("Telemetry initialized")

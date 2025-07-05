"""Example: Initializing telemetry with custom settings.

See docs/cookbook/telemetry_setup.md for details.
"""

from flujo import Flujo, Step, init_telemetry

init_telemetry(
    service_name="demo-app",
    environment="production",
    version="1.0.0",
    sampling_rate=0.5,
    export_telemetry=True,
)

pipeline = Step.from_mapper(lambda x: x.upper(), name="upper")
runner = Flujo(pipeline)

if __name__ == "__main__":
    runner.run("hello")
    print("Telemetry initialized")

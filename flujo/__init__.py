"""
Flujo: A powerful Python library for orchestrating AI workflows.
"""

# Get version
try:
    from importlib.metadata import version

    __version__ = version("flujo")
except Exception:
    __version__ = "0.0.0"

# 1. Import and expose sub-modules for a curated API
from . import recipes
from . import testing
from . import plugins
from . import processors
from . import models
from . import utils
from . import domain
from . import application
from . import infra
from . import caching

# 2. Expose the most essential core components at the top level for convenience.
# These are the symbols users will interact with 90% of the time.
from .application.runner import Flujo
from .registry import PipelineRegistry
from .domain.dsl.step import Step, step
from .domain.dsl.pipeline import Pipeline
from .domain.models import Task, Candidate
from .infra.agents import make_agent_async
from .infra.settings import settings
from .infra.telemetry import init_telemetry
from .telemetry import (
    OpenTelemetryHook,
    PrometheusCollector,
    start_prometheus_server,
)
from .agents import validated_agent, monitored_agent
from .monitor import global_monitor, FlujoMonitor, FailureType

# 3. Define __all__ to control `from flujo import *` behavior and document the public API.
__all__ = [
    # Core Components
    "Flujo",
    "PipelineRegistry",
    "Step",
    "step",
    "Pipeline",
    "Task",
    "Candidate",
    "make_agent_async",
    "validated_agent",
    "monitored_agent",
    "global_monitor",
    "FlujoMonitor",
    "FailureType",
    # Global Singletons & Initializers
    "settings",
    "init_telemetry",
    # Sub-modules
    "recipes",
    "testing",
    "plugins",
    "processors",
    "models",
    "utils",
    "domain",
    "application",
    "infra",
    "caching",
    "OpenTelemetryHook",
    "PrometheusCollector",
    "start_prometheus_server",
]

# ---------------------------------------------------------------------------
# Hypothesis configuration
# ---------------------------------------------------------------------------
# Some property-based tests (e.g. `test_random_linear_pipeline`) can exhibit
# wide latency variance on the first invocation due to JIT, disk caches, etc.
# To avoid flaky `DeadlineExceeded` failures we globally disable the timing
# deadline unless the user overrides it via the HYPOTHESIS_PROFILE env var.

try:
    from hypothesis import settings as _hyp_settings

    # Register and load a profile with no execution deadline.
    _hyp_settings.register_profile("flujo", deadline=None)
    _hyp_settings.load_profile("flujo")
except Exception:  # pragma: no cover – Hypothesis not installed in minimal envs
    pass

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

# 2. Expose the most essential core components at the top level for convenience.
# These are the symbols users will interact with 90% of the time.
from .application.flujo_engine import Flujo
from .domain.pipeline_dsl import Step, step, Pipeline
from .domain.models import Task, Candidate
from .infra.agents import make_agent_async
from .infra.settings import settings
from .infra.telemetry import init_telemetry

# 3. Define __all__ to control `from flujo import *` behavior and document the public API.
__all__ = [
    # Core Components
    "Flujo",
    "Step",
    "step",
    "Pipeline",
    "Task",
    "Candidate",
    "make_agent_async",
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
]

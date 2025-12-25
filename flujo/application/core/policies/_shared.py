from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from flujo.domain.models import (
    BaseModel,
    Failure,
    Paused,
    PipelineContext,
    PipelineResult,
    Quota,
    StepOutcome,
    StepResult,
    Success,
    UsageEstimate,
    UsageLimits,
)
from flujo.domain.outcomes import to_outcome
from ..policy_primitives import (
    LoopResumeState,
    PolicyRegistry,
    _check_hitl_nesting_safety,
    _load_template_config,
    _normalize_builtin_params,
    _normalize_plugin_feedback,
    _unpack_agent_result,
)
from ..types import ExecutionFrame
from flujo.exceptions import (
    ConfigurationError,
    InfiniteFallbackError,
    InfiniteRedirectError,
    MissingAgentError,
    NonRetryableError,
    PausedException,
    PipelineAbortSignal,
    PricingNotConfiguredError,
    UsageLimitExceededError,
)
from flujo.cost import extract_usage_metrics
from flujo.utils.performance import time_perf_ns, time_perf_ns_to_seconds
from flujo.infra import telemetry
from ..hybrid_check import run_hybrid_check
from ..context_adapter import _build_context_update, _inject_context
from ..context_manager import ContextManager
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import BranchFailureStrategy, HumanInTheLoopStep, MergeStrategy, Step
from flujo.domain.dsl.import_step import ImportStep
from flujo.application.conversation.history_manager import HistoryManager, HistoryStrategyConfig
from flujo.processors.conversation import ConversationHistoryPromptProcessor
from flujo.domain.dsl.cache_step import CacheStep, _generate_cache_key

__all__ = [
    "asyncio",
    "time",
    "Awaitable",
    "Callable",
    "Dict",
    "List",
    "Optional",
    "Protocol",
    "Tuple",
    "BaseModel",
    "Failure",
    "Paused",
    "PipelineContext",
    "PipelineResult",
    "Quota",
    "StepOutcome",
    "StepResult",
    "Success",
    "UsageEstimate",
    "UsageLimits",
    "to_outcome",
    "LoopResumeState",
    "PolicyRegistry",
    "_check_hitl_nesting_safety",
    "_load_template_config",
    "_normalize_builtin_params",
    "_normalize_plugin_feedback",
    "_unpack_agent_result",
    "ExecutionFrame",
    "ConfigurationError",
    "InfiniteFallbackError",
    "InfiniteRedirectError",
    "MissingAgentError",
    "NonRetryableError",
    "PausedException",
    "PipelineAbortSignal",
    "PricingNotConfiguredError",
    "UsageLimitExceededError",
    "extract_usage_metrics",
    "time_perf_ns",
    "time_perf_ns_to_seconds",
    "telemetry",
    "run_hybrid_check",
    "_build_context_update",
    "_inject_context",
    "ContextManager",
    "ParallelStep",
    "Pipeline",
    "BranchFailureStrategy",
    "HumanInTheLoopStep",
    "MergeStrategy",
    "Step",
    "ImportStep",
    "HistoryManager",
    "HistoryStrategyConfig",
    "ConversationHistoryPromptProcessor",
    "CacheStep",
    "_generate_cache_key",
]

"""Unit tests for GranularAgentStepExecutor policy.

Tests cover PRD v12 requirements:
- CAS guard logic (skip/gap detection)
- Fingerprint validation
- Quota management (no touch on skip)
- History truncation
"""

import pytest
from pydantic import Field

from unittest.mock import patch

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.policies.granular_policy import GranularAgentStepExecutor
from flujo.domain.dsl.granular import GranularStep, ResumeError
from flujo.domain.models import BaseModel, Success, Quota


class MockContext(BaseModel):
    """Typed mock context for strict-mode execution (must be a Pydantic model)."""

    granular_state: dict[str, object] | None = None
    executed_branches: list[str] = Field(default_factory=list)


class MockAgent:
    """Simple mock agent for testing."""

    def __init__(self, output: object = "test_output", is_complete: bool = False) -> None:
        self._output = output
        self._is_complete = is_complete

    async def run(
        self, payload: object, context: object = None, resources: object = None, **kwargs: object
    ) -> object:
        return self._output


class LegacyProviderAgent(MockAgent):
    """Agent with legacy runtime identity fields."""

    _provider: str = "legacy-provider"


@pytest.mark.asyncio
async def test_granular_cas_guard_skip() -> None:
    """CAS skip: is_complete=True → skip execution, return final_output."""
    core = ExecutorCore()

    # Create a granular step
    step = GranularStep(
        name="test_granular",
        agent=MockAgent(output="should_not_run"),
    )

    # Set up context with COMPLETED granular state
    context = MockContext()
    fingerprint = GranularStep.compute_fingerprint(
        input_data="test",
        system_prompt=None,
        model_id="",
        provider=None,
        tools=[],
        settings={
            "history_max_tokens": 128_000,
            "blob_threshold_bytes": 20_000,
            "enforce_idempotency": False,
        },
    )
    context.granular_state = {
        "turn_index": 1,
        "history": [{"turn_index": 0, "input": "test", "output": "previous_result"}],
        "is_complete": True,  # Agent signaled completion
        "final_output": "previous_result",
        "fingerprint": fingerprint,
    }

    frame = make_execution_frame(
        core,
        step,
        data="test",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    executor = GranularAgentStepExecutor()
    outcome = await executor.execute(core, frame)

    # Should skip and return success with cas_skipped flag
    assert isinstance(outcome, Success)
    assert outcome.step_result.metadata_.get("cas_skipped") is True
    assert outcome.step_result.metadata_.get("is_complete") is True
    # Output should be from stored final_output
    assert outcome.step_result.output == "previous_result"


@pytest.mark.asyncio
async def test_granular_resumes_from_stored_turn() -> None:
    """Resume: stored_index used as starting point for next turn."""
    core = ExecutorCore()

    step = GranularStep(
        name="test_granular",
        agent=MockAgent(output="new_output"),
    )
    executor = GranularAgentStepExecutor()

    context = MockContext()
    fingerprint = executor._compute_fingerprint(
        step=step,
        data="test",
        context=context,
        mode="strict",
    )
    # State exists but not complete - should continue execution
    context.granular_state = {
        "turn_index": 2,  # Already completed 2 turns
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": fingerprint,
    }

    frame = make_execution_frame(
        core,
        step,
        data="test",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    outcome = await executor.execute(core, frame)

    # Should execute and return success
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True
    # Turn index should be incremented from stored value
    assert outcome.step_result.metadata_.get("turn_index") == 4  # 2 + 1 + 1 (next + stored logic)


@pytest.mark.asyncio
async def test_granular_fingerprint_mismatch() -> None:
    """Fingerprint mismatch on resume → ResumeError(irrecoverable=True)."""
    core = ExecutorCore()

    step = GranularStep(
        name="test_granular",
        agent=MockAgent(),
    )

    context = MockContext()
    # Store a different fingerprint
    context.granular_state = {
        "turn_index": 0,
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": "different_fingerprint_hash",
    }

    frame = make_execution_frame(
        core,
        step,
        data="test",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    executor = GranularAgentStepExecutor()

    with pytest.raises(ResumeError) as exc_info:
        await executor.execute(core, frame)

    assert "Fingerprint" in str(exc_info.value) or "fingerprint" in str(exc_info.value).lower()
    assert exc_info.value.irrecoverable is True


@pytest.mark.asyncio
async def test_granular_resume_strict_allows_legacy_fingerprint_shape() -> None:
    """Strict resume should accept legacy strict fingerprints without new identity fields."""
    core = ExecutorCore()
    executor = GranularAgentStepExecutor()
    context = MockContext()

    step = GranularStep(
        name="test_granular",
        agent=LegacyProviderAgent(output="legacy_output"),
    )

    legacy_fingerprint = executor._compute_fingerprint(
        step=step,
        data="input",
        context=context,
        mode="strict",
        include_agent_type=False,
    )
    modern_fingerprint = executor._compute_fingerprint(
        step=step,
        data="input",
        context=context,
        mode="strict",
        include_agent_type=True,
    )
    assert legacy_fingerprint != modern_fingerprint

    context.granular_state = {
        "turn_index": 0,
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": legacy_fingerprint,
    }

    frame = make_execution_frame(
        core,
        step,
        data="input",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    outcome = await executor.execute(core, frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.output == "legacy_output"
    assert outcome.step_result.success is True


@pytest.mark.asyncio
async def test_granular_resume_compat_mode_per_step_override(monkeypatch) -> None:
    """Per-step compat mode should ignore runtime-only drift (e.g. blob threshold)."""
    core = ExecutorCore()
    executor = GranularAgentStepExecutor()
    context = MockContext()
    step_for_fingerprint = GranularStep(
        name="test_granular",
        agent=MockAgent(),
        resume_fingerprint_mode="compat",
        blob_threshold_bytes=20_000,
    )
    stored_strict = executor._compute_fingerprint(
        step_for_fingerprint,
        data="input",
        context=context,
        mode="strict",
    )
    stored_compat = executor._compute_fingerprint(
        step_for_fingerprint,
        data="input",
        context=context,
        mode="compat",
    )

    class GlobalStrictSettings:
        granular_resume_fingerprint_mode = "strict"

    # Ensure per-step override wins over a strict global default.
    monkeypatch.setattr(
        "flujo.application.core.policies.granular_policy.config_manager.load_settings",
        lambda: GlobalStrictSettings(),
    )

    step = GranularStep(
        name="test_granular",
        agent=MockAgent(),
        resume_fingerprint_mode="compat",
        blob_threshold_bytes=10_000,
    )
    context.granular_state = {
        "turn_index": 0,
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": stored_strict,
        "compat_fingerprint": stored_compat,
    }

    frame = make_execution_frame(
        core,
        step,
        data="input",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    outcome = await executor.execute(core, frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.output == "test_output"
    assert outcome.step_result.success is True


@pytest.mark.asyncio
async def test_granular_resume_compat_mode_global_default(monkeypatch) -> None:
    """Global compat mode should allow compatibility resume when no per-step override is set."""
    core = ExecutorCore()
    executor = GranularAgentStepExecutor()
    context = MockContext()
    step_for_fingerprint = GranularStep(
        name="test_granular",
        agent=MockAgent(),
        blob_threshold_bytes=20_000,
    )
    stored_strict = executor._compute_fingerprint(
        step_for_fingerprint,
        data="input",
        context=context,
        mode="strict",
    )
    stored_compat = executor._compute_fingerprint(
        step_for_fingerprint,
        data="input",
        context=context,
        mode="compat",
    )

    class GlobalCompatSettings:
        granular_resume_fingerprint_mode = "compat"

    monkeypatch.setattr(
        "flujo.application.core.policies.granular_policy.config_manager.load_settings",
        lambda: GlobalCompatSettings(),
    )

    step = GranularStep(
        name="test_granular",
        agent=MockAgent(),
        blob_threshold_bytes=10_000,
    )
    context.granular_state = {
        "turn_index": 0,
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": stored_strict,
        "compat_fingerprint": stored_compat,
    }

    frame = make_execution_frame(
        core,
        step,
        data="input",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    outcome = await executor.execute(core, frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.output == "test_output"
    assert outcome.step_result.success is True


@pytest.mark.asyncio
async def test_granular_resume_compat_rejects_input_drift() -> None:
    """Compatibility mode should still fail when input data changes."""
    core = ExecutorCore()
    executor = GranularAgentStepExecutor()
    context = MockContext()
    step = GranularStep(
        name="test_granular",
        agent=MockAgent(),
        resume_fingerprint_mode="compat",
    )
    context.granular_state = {
        "turn_index": 0,
        "history": [],
        "is_complete": False,
        "final_output": None,
        "fingerprint": executor._compute_fingerprint(
            step,
            data="original_input",
            context=context,
            mode="strict",
        ),
        "compat_fingerprint": executor._compute_fingerprint(
            step,
            data="original_input",
            context=context,
            mode="compat",
        ),
    }

    frame = make_execution_frame(
        core,
        step,
        data="drifted_input",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(ResumeError) as exc_info:
        await executor.execute(core, frame)
    assert "Fingerprint mismatch on resume" in str(exc_info.value)
    assert exc_info.value.irrecoverable is True


@pytest.mark.asyncio
async def test_cas_skip_no_quota_touch() -> None:
    """CAS skip path must not reserve/reclaim quota."""
    core = ExecutorCore()

    # Create a small quota that would be affected if touched
    quota = Quota(remaining_cost_usd=1.0, remaining_tokens=100)
    initial_cost = quota._remaining_cost_usd
    initial_tokens = quota._remaining_tokens

    core._set_current_quota(quota)

    step = GranularStep(
        name="test_granular",
        agent=MockAgent(),
    )

    context = MockContext()
    fingerprint = GranularStep.compute_fingerprint(
        input_data="test",
        system_prompt=None,
        model_id="",
        provider=None,
        tools=[],
        settings={
            "history_max_tokens": 128_000,
            "blob_threshold_bytes": 20_000,
            "enforce_idempotency": False,
        },
    )
    context.granular_state = {
        "turn_index": 1,
        "history": [{"turn_index": 0, "input": "test", "output": "cached_out"}],
        "is_complete": True,  # Must be complete to trigger skip
        "final_output": "cached_out",
        "fingerprint": fingerprint,
    }

    frame = make_execution_frame(
        core,
        step,
        data="test",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    executor = GranularAgentStepExecutor()
    outcome = await executor.execute(core, frame)

    assert isinstance(outcome, Success)
    assert outcome.step_result.metadata_.get("cas_skipped") is True
    assert outcome.step_result.output == "cached_out"

    # Quota should be unchanged
    assert quota._remaining_cost_usd == initial_cost
    assert quota._remaining_tokens == initial_tokens


@pytest.mark.asyncio
async def test_history_truncation_middle_out() -> None:
    """Verify deterministic head/tail truncation with placeholder."""
    executor = GranularAgentStepExecutor()

    # Create a long history
    history = [
        {"turn_index": i, "input": f"input_{i}", "output": f"output_{i}" * 100} for i in range(20)
    ]

    # Truncate to a small token limit
    truncated = executor._truncate_history(history, max_tokens=500)

    # Should have first item, placeholder, and tail
    assert len(truncated) < len(history)
    assert truncated[0] == history[0]  # First preserved

    # Check for placeholder
    has_placeholder = any(
        "[Context Truncated:" in str(item.get("content", "")) for item in truncated
    )
    assert has_placeholder, "Truncation should include placeholder"


def test_granular_step_fingerprint_deterministic() -> None:
    """Fingerprint computation should be deterministic."""
    fp1 = GranularStep.compute_fingerprint(
        input_data={"key": "value", "nested": {"a": 1, "b": 2}},
        system_prompt="You are helpful",
        model_id="gpt-4o",
        provider="openai",
        tools=[{"name": "tool1", "sig_hash": "abc123"}],
        settings={"temperature": 0.7, "max_tokens": 1000},
    )

    fp2 = GranularStep.compute_fingerprint(
        input_data={"nested": {"b": 2, "a": 1}, "key": "value"},  # Different order
        system_prompt="You are helpful",
        model_id="gpt-4o",
        provider="openai",
        tools=[{"name": "tool1", "sig_hash": "abc123"}],
        settings={"max_tokens": 1000, "temperature": 0.7},  # Different order
    )

    assert fp1 == fp2, "Fingerprints should be deterministic regardless of dict order"


def test_granular_step_idempotency_key_generation() -> None:
    """Idempotency key should be deterministic."""
    key1 = GranularStep.generate_idempotency_key("run_123", "my_step", 5)
    key2 = GranularStep.generate_idempotency_key("run_123", "my_step", 5)
    key3 = GranularStep.generate_idempotency_key("run_123", "my_step", 6)

    assert key1 == key2, "Same inputs should produce same key"
    assert key1 != key3, "Different turn index should produce different key"
    assert len(key1) == 64, "Should be SHA-256 hex digest"


@pytest.mark.asyncio
async def test_granular_idempotency_wrap_failure_fails_fast() -> None:
    """Verify that if agent wrapping for idempotency fails, we fail fast with ConfigurationError."""
    core = ExecutorCore()
    executor = GranularAgentStepExecutor()

    # Step with idempotency enabled
    step = GranularStep(
        name="test_idempotency_fail",
        agent=MockAgent(),
        enforce_idempotency=True,
    )

    context = MockContext()
    frame = make_execution_frame(
        core,
        step,
        data="test",
        context=context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    # Patch _enforce_idempotency_on_agent to raise an error
    from flujo.exceptions import ConfigurationError

    with patch.object(
        executor, "_enforce_idempotency_on_agent", side_effect=ValueError("Wrap error")
    ):
        with pytest.raises(ConfigurationError) as exc_info:
            await executor.execute(core, frame)

    assert "Failed to wrap agent for idempotency" in str(exc_info.value)
    assert "Wrap error" in str(exc_info.value)
    assert "test_idempotency_fail" in str(exc_info.value)

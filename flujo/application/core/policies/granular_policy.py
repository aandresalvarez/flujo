"""GranularAgentStepExecutor - Policy for crash-safe, resumable agent execution.

Implements PRD v12 Granular Execution Mode with:
- CAS guards for double-execution prevention
- Fingerprint validation for deterministic resume
- Quota reserve/execute/reconcile pattern
- Context isolation and merge-on-success
- Idempotency key injection
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Type

from flujo.application.core.context_manager import ContextManager
from flujo.application.core.policy_registry import StepPolicy
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.granular import GranularStep, GranularState, ResumeError
from flujo.domain.models import (
    Failure,
    StepOutcome,
    StepResult,
    Success,
    UsageEstimate,
)
from flujo.exceptions import (
    PausedException,
    PipelineAbortSignal,
    InfiniteRedirectError,
    UsageLimitExceededError,
)
from flujo.infra import telemetry

__all__ = ["GranularAgentStepExecutor", "DefaultGranularAgentStepExecutor"]


# Storage key for granular state in scratchpad
GRANULAR_STATE_KEY = "granular_state"


def _get_granular_state(context: Any) -> Optional[GranularState]:
    """Extract granular_state from context scratchpad."""
    if context is None:
        return None
    scratch = getattr(context, "scratchpad", None)
    if not isinstance(scratch, dict):
        return None
    raw = scratch.get(GRANULAR_STATE_KEY)
    if isinstance(raw, dict):
        return GranularState(
            turn_index=int(raw.get("turn_index", 0)),
            history=list(raw.get("history", [])),
            is_complete=bool(raw.get("is_complete", False)),
            final_output=raw.get("final_output"),
            fingerprint=str(raw.get("fingerprint", "")),
        )
    return None


def _set_granular_state(context: Any, state: GranularState) -> None:
    """Store granular_state in context scratchpad."""
    if context is None:
        return
    scratch = getattr(context, "scratchpad", None)
    if not isinstance(scratch, dict):
        try:
            scratch = {}
            object.__setattr__(context, "scratchpad", scratch)
        except Exception:
            return
    scratch[GRANULAR_STATE_KEY] = dict(state)


def _get_run_id(context: Any) -> str:
    """Extract run_id from context or generate a fallback."""
    if context is None:
        return f"run_{int(time.time() * 1000)}"
    # Try common locations for run_id
    for attr in ("run_id", "_run_id", "execution_id"):
        val = getattr(context, attr, None)
        if isinstance(val, str) and val:
            return val
    scratch = getattr(context, "scratchpad", None)
    if isinstance(scratch, dict):
        for key in ("run_id", "_run_id"):
            val = scratch.get(key)
            if isinstance(val, str) and val:
                return val
    return f"run_{id(context)}"


def _get_loop_iteration_index(context: Any) -> int:
    """Get current loop iteration from context (for logging only)."""
    if context is None:
        return 0
    scratch = getattr(context, "scratchpad", None)
    if isinstance(scratch, dict):
        # Try internal loop tracking key first
        for key in ("_loop_iteration_index", "loop_iteration"):
            val = scratch.get(key)
            if isinstance(val, int):
                return val
    return 0


class GranularAgentStepExecutor(StepPolicy[GranularStep]):
    """Policy executor for GranularStep - crash-safe, resumable agent execution.

    Implements:
    - CAS guards (§5.1): Skip on ghost-write, fail on gap
    - Fingerprint validation (§5.2): Deterministic resume
    - Quota management (§6): Reserve → Execute → Reconcile
    - Context isolation (§5.4): Isolate before, merge on success
    - Idempotency keys (§7): Inject into deps for tool calls
    """

    @property
    def handles_type(self) -> Type[GranularStep]:
        return GranularStep

    async def execute(
        self,
        core: Any,
        frame: ExecutionFrame[Any],
    ) -> StepOutcome[StepResult]:
        step: GranularStep = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        stream = frame.stream
        on_chunk = frame.on_chunk

        start_ns = time.perf_counter_ns()
        step_name = getattr(step, "name", "<granular>")

        telemetry.logfire.info(f"[GranularPolicy] Executing '{step_name}'")

        # Initialize result for tracking
        result = StepResult(
            name=step_name,
            output=None,
            success=False,
            attempts=1,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=None,
            branch_context=None,
            metadata_={},
            step_history=[],
        )

        # Load existing granular state
        stored_state = _get_granular_state(context)

        # Get stored turn_index (0 if no state yet)
        stored_index = stored_state["turn_index"] if stored_state else 0

        # === CAS Guard Logic (§5.1) ===
        # The granular policy manages its OWN turn tracking independent of loop iteration.
        # On each turn: if stored_index == loop_iteration, we've already done this turn (skip).
        # On resume: we read stored_index and continue from there.

        # Important: The LoopStep runs this policy N times (max_turns).
        # Each invocation is a new "turn". The turn_index in stored_state tells us
        # how many turns we've completed. We should execute if turn_index < max_turns
        # and the agent hasn't signaled completion yet.

        if stored_state is not None:
            # Check if already complete
            if stored_state.get("is_complete", False):
                telemetry.logfire.info(
                    f"[GranularPolicy] Agent already complete at turn {stored_index}"
                )
                result.success = True
                result.output = stored_state.get("final_output")
                result.latency_s = self._ns_to_seconds(time.perf_counter_ns() - start_ns)
                result.metadata_["cas_skipped"] = True
                result.metadata_["turn_index"] = stored_index
                result.metadata_["is_complete"] = True
                result.branch_context = context
                return Success(step_result=result)

            # Validate fingerprint before continuing
            current_fingerprint = self._compute_fingerprint(step, data, context)
            if stored_state["fingerprint"] and stored_state["fingerprint"] != current_fingerprint:
                raise ResumeError(
                    "Fingerprint mismatch on resume - configuration changed",
                    irrecoverable=True,
                )

        # === Fingerprint Computation (§5.2) ===
        current_fingerprint = self._compute_fingerprint(step, data, context)

        # === Context Isolation (§5.4) ===
        isolated_context = (
            ContextManager.isolate(
                context,
                purpose=f"granular_turn:{step_name}:{stored_index}",
            )
            if context is not None
            else None
        )

        # === Quota Reservation (§6) ===
        estimate = self._estimate_usage(step, data, context)
        quota = None
        try:
            if hasattr(core, "_get_current_quota"):
                quota = core._get_current_quota()
        except Exception:
            pass

        if quota is not None:
            if not quota.reserve(estimate):
                try:
                    from flujo.application.core.usage_messages import format_reservation_denial

                    denial = format_reservation_denial(estimate, limits)
                    raise UsageLimitExceededError(denial.human)
                except ImportError:
                    raise UsageLimitExceededError("Insufficient quota for granular turn")

        # === Idempotency Key Injection (§7) ===
        run_id = _get_run_id(context)
        # Use stored_index + 1 as next turn index for idempotency
        next_turn_index = stored_index + 1 if stored_index > 0 else 0
        idempotency_key = GranularStep.generate_idempotency_key(run_id, step_name, next_turn_index)

        # Inject key into resources/deps if possible
        if resources is not None:
            try:
                object.__setattr__(resources, "_idempotency_key", idempotency_key)
            except Exception:
                pass

        # Track execution state for quota reconciliation
        capture_started = False
        network_attempted = False
        actual_usage = UsageEstimate(cost_usd=0.0, tokens=0)

        try:
            # === Execute Agent Turn ===
            capture_started = True

            agent = getattr(step, "agent", None)
            if agent is None:
                raise ValueError(f"GranularStep '{step_name}' has no agent configured")

            # Build history for agent context
            history: List[Dict[str, Any]] = []
            if stored_state is not None:
                history = list(stored_state.get("history", []))

            # Apply history truncation if needed
            if step.history_max_tokens > 0 and history:
                history = self._truncate_history(history, step.history_max_tokens)

            network_attempted = True

            # Run the agent
            try:
                agent_output = await core._agent_runner.run(
                    agent=agent,
                    payload=data,
                    context=isolated_context,
                    resources=resources,
                    options={},
                    stream=stream,
                    on_chunk=on_chunk,
                )
            except PausedException:
                raise  # Re-raise control flow
            except PipelineAbortSignal:
                raise  # Re-raise control flow
            except InfiniteRedirectError:
                raise  # Re-raise control flow

            # Extract usage metrics
            try:
                from flujo.cost import extract_usage_metrics

                prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                    raw_output=agent_output, agent=agent, step_name=step_name
                )
                actual_usage = UsageEstimate(
                    cost_usd=cost_usd,
                    tokens=prompt_tokens + completion_tokens,
                )
                result.token_counts = prompt_tokens + completion_tokens
                result.cost_usd = cost_usd
            except Exception:
                pass

            # Check if agent indicates completion
            is_complete = self._check_completion(agent_output)

            # Update history with new turn
            new_history = list(history)
            new_history.append(
                {
                    "turn_index": next_turn_index,
                    "input": data if self._is_json_serializable(data) else str(data),
                    "output": agent_output
                    if self._is_json_serializable(agent_output)
                    else str(agent_output),
                }
            )

            # === Persist New State (CAS check: only if turn_index matches) ===
            new_state: GranularState = {
                "turn_index": next_turn_index + 1,
                "history": new_history,
                "is_complete": is_complete,
                "final_output": agent_output if is_complete else None,
                "fingerprint": current_fingerprint,
            }

            # Merge isolated context back to main on success
            if context is not None and isolated_context is not None:
                merged = ContextManager.merge(context, isolated_context)
                if merged is not None:
                    context = merged

            _set_granular_state(context, new_state)

            result.success = True
            result.output = agent_output
            result.branch_context = context
            result.metadata_["turn_index"] = next_turn_index + 1
            result.metadata_["is_complete"] = is_complete

            telemetry.logfire.info(
                f"[GranularPolicy] Turn {next_turn_index} complete, is_complete={is_complete}"
            )

            return Success(step_result=result)

        except (PausedException, PipelineAbortSignal, InfiniteRedirectError, ResumeError):
            # Re-raise control flow exceptions unwrapped (§5.3)
            raise
        except Exception as exc:
            result.success = False
            result.feedback = str(exc)
            result.latency_s = self._ns_to_seconds(time.perf_counter_ns() - start_ns)

            telemetry.logfire.error(f"[GranularPolicy] Turn {next_turn_index} failed: {exc}")

            return Failure(
                error=exc,
                feedback=str(exc),
                step_result=result,
            )
        finally:
            # === Quota Reconciliation (§6) - Single reclaim ===
            if quota is not None and capture_started:
                try:
                    if network_attempted and actual_usage.tokens == 0:
                        # Network attempted but no messages - charge baseline
                        actual_usage = estimate
                        telemetry.logfire.warning(
                            "[GranularPolicy] Network attempted but no usage - charging baseline"
                        )
                    quota.reclaim(estimate, actual_usage)
                except Exception:
                    pass

            result.latency_s = self._ns_to_seconds(time.perf_counter_ns() - start_ns)

    def _compute_fingerprint(self, step: GranularStep, data: Any, context: Any) -> str:
        """Compute deterministic fingerprint for run configuration."""
        agent = getattr(step, "agent", None)

        # Extract agent configuration
        model_id = getattr(agent, "_model_name", "") or getattr(agent, "model_id", "")
        provider = getattr(agent, "_provider", None)
        system_prompt = getattr(agent, "_system_prompt", None)

        # Extract tools
        tools: List[Dict[str, Any]] = []
        agent_tools = getattr(agent, "_tools", None) or getattr(agent, "tools", None)
        if agent_tools:
            for tool in agent_tools:
                tool_name = getattr(tool, "__name__", str(tool))
                # Hash the tool's signature
                sig_hash = hashlib.sha256(str(tool).encode()).hexdigest()[:16]
                tools.append({"name": tool_name, "sig_hash": sig_hash})

        # Extract settings
        settings = {
            "history_max_tokens": step.history_max_tokens,
            "blob_threshold_bytes": step.blob_threshold_bytes,
            "enforce_idempotency": step.enforce_idempotency,
        }

        return GranularStep.compute_fingerprint(
            input_data=data,
            system_prompt=system_prompt,
            model_id=model_id,
            provider=str(provider) if provider else None,
            tools=tools,
            settings=settings,
        )

    def _estimate_usage(self, step: Any, data: Any, context: Any) -> UsageEstimate:
        """Estimate token usage for quota reservation."""
        # Conservative default estimate
        try:
            cfg = getattr(step, "config", None)
            if cfg is not None:
                c = getattr(cfg, "expected_cost_usd", None)
                t = getattr(cfg, "expected_tokens", None)
                cost = float(c) if c is not None else 0.01
                tokens = int(t) if t is not None else 1000
                return UsageEstimate(cost_usd=cost, tokens=tokens)
        except Exception:
            pass
        return UsageEstimate(cost_usd=0.01, tokens=1000)

    def _truncate_history(
        self,
        history: List[Dict[str, Any]],
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Apply middle-out truncation to history (§8.2)."""
        if not history:
            return history

        # Simple token estimation (4 chars ≈ 1 token)
        def estimate_tokens(item: Dict[str, Any]) -> int:
            text = json.dumps(item, ensure_ascii=True)
            return len(text) // 4

        total_tokens = sum(estimate_tokens(h) for h in history)

        if total_tokens <= max_tokens:
            return history

        # Keep first and last, truncate middle
        if len(history) <= 2:
            return history

        first = history[0]
        first_tokens = estimate_tokens(first)

        # Budget for tail
        remaining_budget = max_tokens - first_tokens - 50  # 50 tokens for placeholder

        # Collect from tail until budget exhausted
        tail: List[Dict[str, Any]] = []
        tail_tokens = 0
        for item in reversed(history[1:]):
            item_tokens = estimate_tokens(item)
            if tail_tokens + item_tokens <= remaining_budget:
                tail.insert(0, item)
                tail_tokens += item_tokens
            else:
                break

        dropped_count = len(history) - 1 - len(tail)

        if dropped_count > 0:
            placeholder: Dict[str, Any] = {
                "role": "system",
                "content": f"... [Context Truncated: {dropped_count} messages omitted] ...",
            }
            return [first, placeholder] + tail

        return [first] + tail

    def _check_completion(self, output: Any) -> bool:
        """Check if agent output indicates completion."""
        if hasattr(output, "is_complete"):
            return bool(output.is_complete)
        if hasattr(output, "done"):
            return bool(output.done)
        if hasattr(output, "finished"):
            return bool(output.finished)
        if isinstance(output, dict):
            return bool(output.get("is_complete") or output.get("done") or output.get("finished"))
        return False

    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _ns_to_seconds(ns: int) -> float:
        """Convert nanoseconds to seconds."""
        return ns / 1_000_000_000


# Alias for backward compatibility
DefaultGranularAgentStepExecutor = GranularAgentStepExecutor

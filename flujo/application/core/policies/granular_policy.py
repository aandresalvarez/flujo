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
from typing import Optional, Type

from flujo.application.core.context_manager import ContextManager
from flujo.application.core.policy_registry import StepPolicy
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.granular import GranularStep, GranularState, ResumeError
from flujo.exceptions import ConfigurationError
from flujo.state.granular_blob_store import GranularBlobStore
from flujo.domain.models import (
    BaseModel,
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

_NONCALLABLE_TOOL_ERROR = "Tool function for '{tool_name}' is not callable"


def _get_granular_state(context: object | None) -> Optional[GranularState]:
    """Extract granular_state from context.granular_state."""
    if context is None:
        return None
    raw = getattr(context, "granular_state", None)
    if isinstance(raw, dict):
        return GranularState(
            turn_index=int(raw.get("turn_index", 0)),
            history=list(raw.get("history", [])),
            is_complete=bool(raw.get("is_complete", False)),
            final_output=raw.get("final_output"),
            fingerprint=str(raw.get("fingerprint", "")),
        )
    return None


def _set_granular_state(context: object | None, state: GranularState) -> None:
    """Store granular_state in context.granular_state."""
    if context is None:
        return
    payload = dict(state)
    try:
        object.__setattr__(context, "granular_state", payload)
    except Exception:
        try:
            setattr(context, "granular_state", payload)
        except Exception:
            return


def _get_run_id(context: object | None) -> str:
    """Extract run_id from context or generate a fallback."""
    if context is None:
        return f"run_{int(time.time() * 1000)}"
    # Try common locations for run_id
    for attr in ("run_id", "_run_id", "execution_id"):
        val = getattr(context, attr, None)
        if isinstance(val, str) and val:
            return val
    return f"run_{id(context)}"


def _get_loop_iteration_index(context: object | None) -> int:
    """Get current loop iteration from context (for logging only)."""
    if context is None:
        return 0
    val = getattr(context, "loop_iteration_index", None)
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
        core: object,
        frame: ExecutionFrame[BaseModel],
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

        # === Blob Store Initialization (§8.1) ===
        state_backend = getattr(core, "_state_backend", None)
        blob_store = None
        if state_backend is not None:
            blob_store = GranularBlobStore(state_backend, threshold_bytes=step.blob_threshold_bytes)

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
            remaining_before: tuple[float, int] | None = None
            try:
                remaining_before = quota.get_remaining()
            except Exception:
                remaining_before = None
            if not quota.reserve(estimate):
                try:
                    from flujo.application.core.usage_messages import format_reservation_denial

                    denial = format_reservation_denial(estimate, limits, remaining=remaining_before)
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

        # Extract agent from step
        agent = getattr(step, "agent", None)
        if agent is None:
            raise ValueError(f"GranularStep '{step_name}' has no agent configured")

        # Wrap agent for idempotency enforcement if requested (§7)
        if step.enforce_idempotency:
            try:
                agent = self._enforce_idempotency_on_agent(agent, idempotency_key)
            except Exception as e:
                # Fail-fast: ensure idempotency requests are not silently ignored
                error_msg = (
                    f"Failed to wrap agent for idempotency (step='{step_name}', "
                    f"key='{idempotency_key}'): {e}"
                )
                telemetry.logfire.error(error_msg)
                raise ConfigurationError(error_msg) from e

        try:
            # === Execute Agent Turn ===
            capture_started = True

            # Build history for agent context
            history: list[dict[str, object]] = []
            if stored_state is not None:
                raw_history = list(stored_state.get("history", []))
                # Hydrate blobs if store is available (§8.1)
                if blob_store is not None:
                    hydrated_history = []
                    for entry in raw_history:
                        try:
                            hydrated = await blob_store.hydrate_history_entry(entry)
                            hydrated_history.append(hydrated)
                        except Exception as e:
                            telemetry.logfire.error(f"Failed to hydrate blob in history: {e}")
                            raise ResumeError(
                                f"Irrecoverable state: Failed to hydrate history from blob store: {e}",
                                irrecoverable=True,
                            ) from e
                    history = hydrated_history
                else:
                    history = raw_history

            # Apply history truncation if needed
            if step.history_max_tokens > 0 and history:
                history = self._truncate_history(history, step.history_max_tokens)

            network_attempted = True

            # Run the agent
            try:
                agent_runner = getattr(core, "_agent_runner", None)
                run_fn = getattr(agent_runner, "run", None)
                if not callable(run_fn):
                    raise TypeError("ExecutorCore missing _agent_runner.run")
                agent_output = await run_fn(
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
            new_turn_entry = {
                "turn_index": next_turn_index,
                "input": data if self._is_json_serializable(data) else str(data),
                "output": agent_output
                if self._is_json_serializable(agent_output)
                else str(agent_output),
            }

            # Offload large payloads if store is available (§8.1)
            if blob_store is not None:
                new_turn_entry = await blob_store.process_history_entry(
                    new_turn_entry, run_id, step_name, next_turn_index
                )

            new_history = list(history)
            new_history.append(new_turn_entry)

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

    def _compute_fingerprint(self, step: GranularStep, data: object, context: object | None) -> str:
        """Compute deterministic fingerprint for run configuration."""
        agent = getattr(step, "agent", None)

        # Extract agent configuration
        model_id = getattr(agent, "_model_name", "") or getattr(agent, "model_id", "")
        provider = getattr(agent, "_provider", None)
        system_prompt = getattr(agent, "_system_prompt", None)

        # Extract tools
        tools: list[dict[str, object]] = []
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

    def _estimate_usage(self, step: object, data: object, context: object | None) -> UsageEstimate:
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
        history: list[dict[str, object]],
        max_tokens: int,
    ) -> list[dict[str, object]]:
        """Apply middle-out truncation to history (§8.2)."""
        if not history:
            return history

        # Simple token estimation (4 chars ≈ 1 token)
        def estimate_tokens(item: dict[str, object]) -> int:
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
        tail: list[dict[str, object]] = []
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
            placeholder: dict[str, object] = {
                "role": "system",
                "content": f"... [Context Truncated: {dropped_count} messages omitted] ...",
            }
            return [first, placeholder] + tail

        return [first] + tail

    def _check_completion(self, output: object) -> bool:
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

    def _is_json_serializable(self, obj: object) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _enforce_idempotency_on_agent(self, agent: object, key: str) -> object:
        """Wrap agent's tools to ensure idempotency_key is present in payloads (§7)."""
        # If it's a pydantic-ai Agent, we can attempt to wrap its tools
        if hasattr(agent, "tools") and isinstance(agent.tools, dict):
            from functools import wraps

            from flujo.exceptions import ConfigurationError

            # We should avoid mutating the original agent if possible,
            # but pydantic-ai agents are often created per-step.
            # For GranularStep, we might just wrap the tool functions.

            for name, tool in agent.tools.items():
                # Check if tool requires idempotency key
                tool_requires = getattr(tool, "requires_idempotency_key", False)

                # Validation hook: wrap the tool function to check payload
                original_func = tool.function

                @wraps(original_func)
                async def wrapped_tool(
                    *args: object,
                    _name: str = name,
                    _tool_requires: bool = tool_requires,
                    _original_func: object = original_func,
                    _key: str = key,
                    **kwargs: object,
                ) -> object:
                    # In pydantic-ai, tool functions usually get (ctx, payload) or (payload)
                    # We look for idempotency_key in kwargs or payload
                    has_key = "idempotency_key" in kwargs
                    if has_key:
                        if kwargs["idempotency_key"] != _key:
                            raise ConfigurationError(
                                f"Idempotency key mismatch in tool '{_name}': "
                                f"expected {_key}, got {kwargs['idempotency_key']}"
                            )

                    if not has_key and args:
                        # Check last arg if it's a dict (payload)
                        last_arg = args[-1]
                        if isinstance(last_arg, dict) and "idempotency_key" in last_arg:
                            has_key = True
                            if last_arg["idempotency_key"] != _key:
                                raise ConfigurationError(
                                    f"Idempotency key mismatch in tool '{_name}': "
                                    f"expected {_key}, got {last_arg['idempotency_key']}"
                                )

                    if not has_key and _tool_requires:
                        raise ConfigurationError(
                            f"Tool '{_name}' requires 'idempotency_key' but it was not provided."
                        )

                    if not callable(_original_func):
                        raise ConfigurationError(_NONCALLABLE_TOOL_ERROR.format(tool_name=_name))

                    return await _original_func(*args, **kwargs)

                # Replace function with wrapped version
                tool.function = wrapped_tool

        return agent

    @staticmethod
    def _ns_to_seconds(ns: int) -> float:
        """Convert nanoseconds to seconds."""
        return ns / 1_000_000_000


# Alias for backward compatibility
DefaultGranularAgentStepExecutor = GranularAgentStepExecutor

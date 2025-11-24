from __future__ import annotations
# mypy: ignore-errors

from ._shared import (
    Any,
    Awaitable,
    BaseModel,
    InfiniteRedirectError,
    Optional,
    Protocol,
    Tuple,
    asyncio,
    telemetry,
)


# --- Timeout runner policy ---
class TimeoutRunner(Protocol):
    async def run_with_timeout(self, coro: Awaitable[Any], timeout_s: Optional[float]) -> Any: ...


class DefaultTimeoutRunner:
    async def run_with_timeout(self, coro: Awaitable[Any], timeout_s: Optional[float]) -> Any:
        if timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout_s)


# --- Agent result unpacker policy ---
class AgentResultUnpacker(Protocol):
    def unpack(self, output: Any) -> Any: ...


class DefaultAgentResultUnpacker:
    def unpack(self, output: Any) -> Any:
        if isinstance(output, BaseModel):
            return output
        for attr in ("output", "content", "result", "data", "text", "message", "value"):
            if hasattr(output, attr):
                return getattr(output, attr)
        return output


# --- Plugin redirector policy ---
class PluginRedirector(Protocol):
    async def run(
        self,
        initial: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        timeout_s: Optional[float],
    ) -> Any: ...


class DefaultPluginRedirector:
    def __init__(self, plugin_runner: Any, agent_runner: Any):
        self._plugin_runner = plugin_runner
        self._agent_runner = agent_runner

    def _hash_text_streaming(self, text: str, chunk_size: int = 65536) -> str:
        """Hash large text inputs in chunks to reduce peak memory usage.

        Uses SHA-256 with UTF-8 encoding in streaming updates.
        """
        try:
            from hashlib import sha256  # Lazy import
        except Exception:
            return f"len:{len(text)}"
        hasher = sha256()
        for i in range(0, len(text), chunk_size):
            hasher.update(text[i : i + chunk_size].encode("utf-8"))
        return hasher.hexdigest()

    def _get_agent_signature(self, agent: Any) -> Tuple[Any, Optional[str], str]:
        """Generate a stable logical signature for an agent to detect redirect loops.

        The signature combines:
          - The agent's concrete type (class)
          - A model identifier when available (e.g., provider:model)
          - A SHA-256 hash of the system prompt (stringified), if present
        """
        if agent is None:
            return (None, None, "")

        try:
            # Prefer explicit public attribute, then common fallbacks
            model_id: Optional[str] = None
            try:
                model_id = getattr(agent, "model_id", None)
                if model_id is None:
                    model_id = getattr(agent, "_model_name", None)
                if model_id is None:
                    model_id = getattr(agent, "model", None)
            except Exception:
                model_id = None

            # System prompt may be stored in different attributes
            try:
                system_prompt_val = getattr(agent, "system_prompt", None)
                if system_prompt_val is None and hasattr(agent, "_system_prompt"):
                    system_prompt_val = getattr(agent, "_system_prompt", None)
            except Exception:
                system_prompt_val = None

            # Normalize and hash system prompt to avoid large tuples and ensure stability
            if system_prompt_val is not None:
                sp_hash = self._hash_text_streaming(str(system_prompt_val))
            else:
                sp_hash = ""

            return (agent.__class__, str(model_id) if model_id is not None else None, sp_hash)
        except Exception:
            # Defensive fallback: use class only; avoids crashing loop detection
            return (agent.__class__, None, "")

    async def run(
        self,
        initial: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        timeout_s: Optional[float],
    ) -> Any:
        telemetry.logfire.info("[Redirector] Start plugin redirect loop")
        redirect_chain: list[Any] = []
        redirect_chain_signatures: list[Tuple[Any, Optional[str], str]] = []
        processed = initial
        unpacker = DefaultAgentResultUnpacker()
        while True:
            # Normalize plugin input to expected dict shape
            plugin_input = processed
            if not isinstance(plugin_input, dict):
                try:
                    plugin_input = {"output": unpacker.unpack(plugin_input)}
                except Exception:
                    plugin_input = {"output": plugin_input}
            outcome = await asyncio.wait_for(
                self._plugin_runner.run_plugins(
                    step.plugins,
                    plugin_input,
                    context=context,
                    resources=resources,
                ),
                timeout_s,
            )
            try:
                rt = getattr(outcome, "redirect_to", None)
                telemetry.logfire.info(
                    f"[Redirector] Plugin outcome: redirect_to={rt}, success={getattr(outcome, 'success', None)}"
                )
            except Exception:
                pass
            # Handle redirect_to
            if hasattr(outcome, "redirect_to") and outcome.redirect_to is not None:
                # Compute logical identity-based signature for loop detection
                redirect_agent = outcome.redirect_to
                agent_sig = self._get_agent_signature(redirect_agent)

                # Check against previously seen agent signatures in this redirect chain
                if agent_sig in redirect_chain_signatures:
                    telemetry.logfire.warning(
                        f"[Redirector] Loop detected for agent signature {agent_sig}"
                    )
                    raise InfiniteRedirectError(
                        f"Redirect loop detected for agent signature {agent_sig}"
                    )

                redirect_chain.append(redirect_agent)
                redirect_chain_signatures.append(agent_sig)
                telemetry.logfire.info(f"[Redirector] Redirecting to agent {outcome.redirect_to}")
                raw = await asyncio.wait_for(
                    self._agent_runner.run(
                        agent=outcome.redirect_to,
                        payload=data,
                        context=context,
                        resources=resources,
                        options={},
                        stream=False,
                    ),
                    timeout_s,
                )
                processed = unpacker.unpack(raw)
                continue
            # Failure
            if hasattr(outcome, "success") and not outcome.success:
                # Core will wrap generic exceptions as its own PluginError and add retry semantics
                fb = outcome.feedback or "Plugin failed without feedback"
                raise Exception(f"Plugin validation failed: {fb}")
            # New solution
            if hasattr(outcome, "new_solution") and outcome.new_solution is not None:
                processed = outcome.new_solution
                continue
            # Dict-based contract with 'output' overrides processed value
            if isinstance(outcome, dict) and "output" in outcome:
                processed = outcome["output"]
                # No redirect or failure case; return the processed value
                return processed
            # Success without changes → keep processed as-is
            return processed


# --- Validator invocation policy ---
class ValidatorInvoker(Protocol):
    async def validate(
        self, output: Any, step: Any, context: Any, timeout_s: Optional[float]
    ) -> None: ...


class DefaultValidatorInvoker:
    def __init__(self, validator_runner: Any):
        self._validator_runner = validator_runner

    async def validate(
        self, output: Any, step: Any, context: Any, timeout_s: Optional[float]
    ) -> None:
        # No validators
        if not getattr(step, "validators", []):
            return
        results = await asyncio.wait_for(
            self._validator_runner.validate(step.validators, output, context=context),
            timeout_s,
        )
        # ✅ FLUJO BEST PRACTICE: Robust NoneType and iterable validation
        # Critical fix: Handle cases where validator results might be None or not iterable
        if results is None:
            return

        # Ensure results is iterable before iterating
        if not hasattr(results, "__iter__"):
            return

        for r in results:
            if not getattr(r, "is_valid", False):
                # Raise a generic exception; core wraps/handles uniformly for retries/fallback
                raise Exception(r.feedback)

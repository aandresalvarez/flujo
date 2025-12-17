from __future__ import annotations

from ._shared import (
    Awaitable,
    BaseModel,
    InfiniteRedirectError,
    Optional,
    Protocol,
    asyncio,
    telemetry,
)
from flujo.exceptions import PluginError, ValidationError
from typing import Sequence, TypeVar

from ..executor_protocols import IAgentRunner, IPluginRunner, IValidatorRunner


# --- Timeout runner policy ---
T = TypeVar("T")


class TimeoutRunner(Protocol):
    async def run_with_timeout(self, coro: Awaitable[T], timeout_s: Optional[float]) -> T: ...


class DefaultTimeoutRunner:
    async def run_with_timeout(self, coro: Awaitable[T], timeout_s: Optional[float]) -> T:
        if timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout_s)


# --- Agent result unpacker policy ---
class AgentResultUnpacker(Protocol):
    def unpack(self, output: object) -> object: ...


class DefaultAgentResultUnpacker:
    def unpack(self, output: object) -> object:
        if isinstance(output, BaseModel):
            return output
        # Avoid unwrapping structured Pydantic models (including user-defined ones)
        # just because they have common attribute names like `.text`.
        try:
            from pydantic import (
                BaseModel as PydanticBaseModel,
            )  # local import to avoid hard dep at import time

            if isinstance(output, PydanticBaseModel):
                return output
        except Exception:
            pass
        for attr in ("output", "content", "result", "data", "text", "message", "value"):
            if hasattr(output, attr):
                return getattr(output, attr)
        return output


# --- Plugin redirector policy ---
class SupportsPlugins(Protocol):
    @property
    def plugins(self) -> Sequence[tuple[object, int]]: ...


class PluginRedirector(Protocol):
    async def run(
        self,
        initial: object,
        step: SupportsPlugins,
        data: object,
        context: object,
        resources: object,
        timeout_s: Optional[float],
    ) -> object: ...


class DefaultPluginRedirector:
    def __init__(self, plugin_runner: IPluginRunner, agent_runner: IAgentRunner):
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

    def _get_agent_signature(self, agent: object) -> tuple[type[object] | None, str | None, str]:
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
        initial: object,
        step: SupportsPlugins,
        data: object,
        context: object,
        resources: object,
        timeout_s: Optional[float],
    ) -> object:
        telemetry.logfire.info("[Redirector] Start plugin redirect loop")
        redirect_chain: list[object] = []
        redirect_chain_signatures: list[tuple[type[object] | None, str | None, str]] = []
        processed: object = initial
        unpacker = DefaultAgentResultUnpacker()
        while True:
            # Normalize plugin input to expected dict shape
            plugin_input: object
            if isinstance(processed, dict):
                plugin_input = processed
            else:
                try:
                    plugin_input = {"output": unpacker.unpack(processed)}
                except Exception:
                    plugin_input = {"output": processed}
            outcome = await asyncio.wait_for(
                self._plugin_runner.run_plugins(
                    list(step.plugins),
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
            redirect_to = getattr(outcome, "redirect_to", None)
            if redirect_to is not None:
                # Compute logical identity-based signature for loop detection
                redirect_agent = redirect_to
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
                telemetry.logfire.info(f"[Redirector] Redirecting to agent {redirect_agent}")
                raw = await asyncio.wait_for(
                    self._agent_runner.run(
                        agent=redirect_agent,
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
            success = getattr(outcome, "success", None)
            if success is False:
                # Core will wrap generic exceptions as its own PluginError and add retry semantics
                feedback = getattr(outcome, "feedback", None)
                fb = feedback or "Plugin failed without feedback"
                raise PluginError(f"Plugin validation failed: {fb}")
            # New solution
            new_solution = getattr(outcome, "new_solution", None)
            if new_solution is not None:
                processed = new_solution
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
        self, output: object, step: object, context: object | None, timeout_s: Optional[float]
    ) -> None: ...


class DefaultValidatorInvoker:
    def __init__(self, validator_runner: IValidatorRunner):
        self._validator_runner = validator_runner

    async def validate(
        self, output: object, step: object, context: object | None, timeout_s: Optional[float]
    ) -> None:
        # No validators
        validators = getattr(step, "validators", None)
        if not validators:
            return
        validators_list: list[object] = (
            validators if isinstance(validators, list) else list(validators)
        )
        results = await asyncio.wait_for(
            self._validator_runner.validate(validators_list, output, context=context),
            timeout_s,
        )
        for r in results:
            if not r.is_valid:
                # Raise a specific validation exception
                raise ValidationError(r.feedback or "Validation failed", code="VALIDATION_FAILED")

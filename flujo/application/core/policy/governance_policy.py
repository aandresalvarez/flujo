from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from typing import Protocol, Sequence

from ....exceptions import ConfigurationError
from ....infra import telemetry


@dataclass(frozen=True)
class GovernanceDecision:
    allow: bool
    reason: str | None = None
    # Optional data replacement (e.g., PII-scrubbed input) to use for downstream execution.
    replacement_data: object | None = None


class GovernancePolicy(Protocol):
    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> GovernanceDecision: ...


class AllowAllGovernancePolicy:
    """Default policy that allows all agent executions."""

    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> GovernanceDecision:
        return GovernanceDecision(allow=True)


class DenyAllGovernancePolicy:
    """Policy that denies all agent executions."""

    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> GovernanceDecision:
        return GovernanceDecision(allow=False, reason="governance_mode=deny_all")


@dataclass(frozen=True)
class PIIScrubbingPolicy:
    """Best-effort PII scrubbing policy.

    This policy replaces common PII patterns in the step input payload before agent execution.
    """

    replacement: str = "[REDACTED]"
    strong: bool = False

    # Conservative patterns to reduce accidental leaks to LLM providers.
    _email_re: re.Pattern[str] = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    )
    _ssn_re: re.Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _phone_re: re.Pattern[str] = re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    )
    _cc_re: re.Pattern[str] = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

    def _redact_text(self, text: str) -> tuple[str, bool]:
        changed = False
        for pat in (self._email_re, self._ssn_re, self._phone_re, self._cc_re):
            new = pat.sub(self.replacement, text)
            if new != text:
                changed = True
                text = new
        return text, changed

    def _scrub(self, obj: object) -> tuple[object, bool]:
        if isinstance(obj, str):
            if self.strong:
                strong = self._redact_presidio(obj)
                if strong is not None and strong != obj:
                    return strong, True
            return self._redact_text(obj)
        if isinstance(obj, list):
            out_list: list[object] = []
            changed_any = False
            for item in obj:
                scrubbed, changed = self._scrub(item)
                out_list.append(scrubbed)
                changed_any = changed_any or changed
            return out_list, changed_any
        if isinstance(obj, tuple):
            scrubbed_list, changed = self._scrub(list(obj))
            if isinstance(scrubbed_list, list):
                return tuple(scrubbed_list), changed
            return obj, changed
        if isinstance(obj, dict):
            out_dict: dict[object, object] = {}
            changed_any = False
            for k, v in obj.items():
                scrubbed_v, changed = self._scrub(v)
                out_dict[k] = scrubbed_v
                changed_any = changed_any or changed
            return out_dict, changed_any
        return obj, False

    def _redact_presidio(self, text: str) -> str | None:
        """Attempt Presidio-based redaction when available.

        Returns:
        - redacted string on success
        - None when presidio is unavailable or fails to initialize
        """
        try:
            presidio_analyzer = importlib.import_module("presidio_analyzer")
            presidio_anonymizer = importlib.import_module("presidio_anonymizer")
            presidio_entities = importlib.import_module("presidio_anonymizer.entities")
        except Exception:
            return None

        AnalyzerEngine = getattr(presidio_analyzer, "AnalyzerEngine", None)
        AnonymizerEngine = getattr(presidio_anonymizer, "AnonymizerEngine", None)
        OperatorConfig = getattr(presidio_entities, "OperatorConfig", None)
        if (
            not callable(AnalyzerEngine)
            or not callable(AnonymizerEngine)
            or not callable(OperatorConfig)
        ):
            return None

        try:
            analyzer = AnalyzerEngine()
            results = analyzer.analyze(text=text, language="en")
            if not results:
                return text
            anonymizer = AnonymizerEngine()
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": self.replacement})}
            anonymized = anonymizer.anonymize(
                text=text, analyzer_results=results, operators=operators
            )
            out = getattr(anonymized, "text", None)
            return out if isinstance(out, str) else None
        except Exception:
            return None

    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> GovernanceDecision:
        del core, context, resources
        scrubbed, changed = self._scrub(data)
        if not changed:
            return GovernanceDecision(allow=True)
        try:
            telemetry.logfire.debug(
                "[Governance] PII scrubbed from step input",
                extra={"step": getattr(step, "name", "<unnamed>")},
            )
        except Exception:
            pass
        return GovernanceDecision(allow=True, replacement_data=scrubbed, reason="pii_scrubbed")


@dataclass(frozen=True)
class ToolAllowlistPolicy:
    """Deny agent execution when a step exposes disallowed tools to the model."""

    allowed: frozenset[str]
    deny_unknown: bool = True

    def _iter_tool_ids(self, step: object) -> list[str]:
        agent_obj = getattr(step, "agent", None)
        if agent_obj is None:
            return []
        # Unwrap AsyncAgentWrapper when present.
        inner = getattr(agent_obj, "_agent", agent_obj)
        tools = getattr(inner, "tools", None)
        if not isinstance(tools, list):
            return []
        ids: list[str] = []
        for tool in tools:
            if tool is None:
                continue
            skill_id = getattr(tool, "__flujo_skill_id__", None)
            if isinstance(skill_id, str) and skill_id.strip():
                ids.append(skill_id)
                continue
            name = getattr(tool, "__name__", None)
            if isinstance(name, str) and name.strip():
                ids.append(name)
                continue
            tname = getattr(tool, "name", None)
            if isinstance(tname, str) and tname.strip():
                ids.append(tname)
        return ids

    async def evaluate(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> GovernanceDecision:
        del core, data, context, resources
        tool_ids = self._iter_tool_ids(step)
        if not tool_ids:
            return GovernanceDecision(allow=True)
        for tool_id in tool_ids:
            if tool_id in self.allowed:
                continue
            if not self.deny_unknown and tool_id.startswith("builtins."):
                continue
            return GovernanceDecision(allow=False, reason=f"tool_not_allowed:{tool_id}")
        return GovernanceDecision(allow=True)


class GovernanceEngine:
    """Evaluates governance policies before agent execution."""

    def __init__(self, policies: Sequence[GovernancePolicy] | None = None) -> None:
        self._policies: Sequence[GovernancePolicy] = (
            policies
            if policies is not None and len(policies) > 0
            else (AllowAllGovernancePolicy(),)
        )
        self._allow_count: int = 0
        self._deny_count: int = 0

    async def enforce(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
    ) -> object:
        current_data: object = data
        for policy in self._policies:
            decision = await policy.evaluate(
                core=core, step=step, data=current_data, context=context, resources=resources
            )
            if not decision.allow:
                self._deny_count += 1
                telemetry.logfire.warning(
                    f"[Governance] Deny agent execution "
                    f"(step={getattr(step, 'name', '<unnamed>')}, "
                    f"reason={decision.reason or 'unspecified'}) "
                    f"counts(allow={self._allow_count}, deny={self._deny_count})"
                )
                raise ConfigurationError(
                    decision.reason or "Agent execution blocked by governance policy"
                )
            if decision.replacement_data is not None:
                current_data = decision.replacement_data
            self._allow_count += 1
            telemetry.logfire.info(
                f"[Governance] Allow agent execution "
                f"(step={getattr(step, 'name', '<unnamed>')}) "
                f"counts(allow={self._allow_count}, deny={self._deny_count})"
            )
        return current_data

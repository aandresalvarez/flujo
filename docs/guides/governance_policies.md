# Governance Policies (Pre-Execution Guardrails)

Flujo supports a governance policy hook that can allow/deny agent execution before the agent is called.

## What It Does Today

- Runs before agent orchestration.
- Allows you to block execution (deny) with a reason.
- Default behavior is allow-all unless you configure a policy.

## Configure a Policy (Custom)

Set a module path (`pkg.mod:Class`) via config or environment:

```toml
[settings]
governance_policy_module = "my_project.policies:MyPolicy"
```

Or:

```bash
export FLUJO_GOVERNANCE_POLICY_MODULE="my_project.policies:MyPolicy"
```

## Policy Interface (current)

Policies implement an async `evaluate(...)` method and return a `GovernanceDecision`.

```python
from flujo.application.core.policy.governance_policy import GovernanceDecision

class MyPolicy:
    async def evaluate(self, *, core, step, data, context, resources) -> GovernanceDecision:
        # Example: deny if input contains a marker
        if isinstance(data, str) and "DO_NOT_RUN" in data:
            return GovernanceDecision(allow=False, reason="blocked_by_policy")
        return GovernanceDecision(allow=True)
```

## Built-in Governance Controls (Env)

These controls require no custom policy class.

Environment variables:
- `FLUJO_GOVERNANCE_MODE=allow_all|deny_all` (default `allow_all`)
- `FLUJO_GOVERNANCE_PII_SCRUB=1` (scrub common PII patterns from step input before agent execution)
- `FLUJO_GOVERNANCE_PII_STRONG=1` (use Presidio scrubbing when installed; falls back to regex scrubber)
- `FLUJO_GOVERNANCE_TOOL_ALLOWLIST=tool_a,tool_b` (allow only these tools; enforced at call time)

Optional dependency:

```bash
pip install "flujo[pii]"
```

### Tool Allowlisting Semantics

Tool allowlisting is enforced in two places:
- Pre-execution: blocks a step if it exposes disallowed tools.
- Tool-call-time: blocks invocation if `FLUJO_GOVERNANCE_TOOL_ALLOWLIST` is set and the tool id is not allowed.

This dual enforcement is intentional: it prevents both accidental exposure and accidental invocation.

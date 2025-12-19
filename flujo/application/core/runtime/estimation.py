from __future__ import annotations

from typing import get_type_hints, get_origin, get_args
from typing import Protocol, Callable

from flujo.domain.models import UsageEstimate
from flujo.infra import telemetry
from flujo.infra.config_manager import get_config_manager


class UsageEstimator(Protocol):
    def estimate(self, step: object, data: object, context: object | None) -> UsageEstimate: ...


class HeuristicUsageEstimator:
    """Phase 1: Simple, conservative estimator.

    Strategy:
    - Prefer explicit hints from `step.config.expected_cost_usd` and `step.config.expected_tokens`.
    - For known agent model names, apply small conservative upper-bounds to avoid false preemption.
    - Otherwise return minimal estimate (0 cost, 0 tokens) and rely on post-usage reconciliation.
    """

    def estimate(self, step: object, data: object, context: object | None) -> UsageEstimate:
        # 1) Explicit config hints
        try:
            cfg = getattr(step, "config", None)
            if cfg is not None:
                c = getattr(cfg, "expected_cost_usd", None)
                t = getattr(cfg, "expected_tokens", None)
                if c is not None or t is not None:
                    return UsageEstimate(
                        cost_usd=float(c) if c is not None else 0.0,
                        tokens=int(t) if t is not None else 0,
                    )
        except Exception:
            pass

        # 2) Infer fixed usage from agent return type defaults (common in tests where
        # the output model encodes constant `cost_usd`/`token_counts` fields).
        try:
            agent = getattr(step, "agent", None)
            run_fn = getattr(agent, "run", None) if agent is not None else None
            if callable(run_fn):
                hints = get_type_hints(run_fn)
                ret = hints.get("return")
                # Handle `-> Coroutine[..., ..., T]`-style shapes
                origin = get_origin(ret)
                if origin is not None and origin.__name__ in ("Coroutine", "Awaitable"):
                    args = get_args(ret)
                    if args:
                        ret = args[-1]
                # Handle unions by selecting the first concrete class
                if get_origin(ret) is None and isinstance(ret, type):
                    out_type: type[object] = ret
                else:
                    out_type = None  # type: ignore[assignment]
                    args = get_args(ret) if ret is not None else ()
                    for a in args:
                        if isinstance(a, type):
                            out_type = a
                            break
                if out_type is not None and isinstance(out_type, type):
                    fields = getattr(out_type, "model_fields", None)
                    if isinstance(fields, dict):
                        cost_default = None
                        tok_default = None
                        try:
                            fi = fields.get("cost_usd")
                            if fi is not None:
                                d = getattr(fi, "default", None)
                                if isinstance(d, (int, float)):
                                    cost_default = float(d)
                        except Exception:
                            pass
                        try:
                            fi = fields.get("token_counts")
                            if fi is not None:
                                d = getattr(fi, "default", None)
                                if isinstance(d, int):
                                    tok_default = int(d)
                                elif isinstance(d, float) and d.is_integer():
                                    tok_default = int(d)
                        except Exception:
                            pass
                        if cost_default is not None or tok_default is not None:
                            return UsageEstimate(
                                cost_usd=float(cost_default or 0.0),
                                tokens=int(tok_default or 0),
                            )
        except Exception:
            pass

        # 3) TOML overrides for provider/model specific hints (if present)
        try:
            cfg = get_config_manager().load_config()
            cost_cfg = getattr(cfg, "cost", None)
            if isinstance(cost_cfg, dict):
                agent = getattr(step, "agent", None)
                provider = getattr(agent, "_provider", None)
                model_name = getattr(agent, "_model_name", None)
                if isinstance(model_name, str):
                    prov_key = str(provider) if provider is not None else "default"
                    model_key = model_name
                    prov_section = cost_cfg.get("estimators", {}).get(prov_key, {})
                    model_section = prov_section.get(model_key, {})
                    c = model_section.get("expected_cost_usd")
                    t = model_section.get("expected_tokens")
                    if c is not None or t is not None:
                        return UsageEstimate(cost_usd=float(c or 0.0), tokens=int(t or 0))
        except Exception:
            pass

        # 4) Conservative upper-bounds for common model identifiers
        try:
            agent = getattr(step, "agent", None)
            model_name = getattr(agent, "_model_name", None)
            if isinstance(model_name, str):
                name = model_name.lower()
                if "gpt-4" in name or "o4" in name:
                    return UsageEstimate(cost_usd=0.10, tokens=500)
                if "gpt-3.5" in name or "gpt-35" in name:
                    return UsageEstimate(cost_usd=0.02, tokens=300)
        except Exception:
            pass

        # 5) Minimal default
        return UsageEstimate(cost_usd=0.0, tokens=0)


class EstimatorRule:
    """A single registry rule mapping a matcher to an estimator.

    The matcher receives the step object and returns True if the rule applies.
    The first matching rule wins.
    """

    def __init__(self, matcher: Callable[[object], bool], estimator: UsageEstimator) -> None:
        self.matcher = matcher
        self.estimator = estimator


class EstimatorRegistry:
    """Registry of estimator selection rules.

    Rules are evaluated in registration order; the first match is selected.
    """

    def __init__(self) -> None:
        self._rules: list[EstimatorRule] = []

    def register(self, matcher: Callable[[object], bool], estimator: UsageEstimator) -> None:
        self._rules.append(EstimatorRule(matcher, estimator))

    def resolve(self, step: object) -> UsageEstimator | None:
        for rule in self._rules:
            try:
                if rule.matcher(step):
                    return rule.estimator
            except Exception:
                # Ignore matcher errors and continue
                continue
        return None


class UsageEstimatorFactory:
    """Factory that selects an estimator using a registry with a default fallback."""

    def __init__(
        self, registry: EstimatorRegistry, default_estimator: UsageEstimator | None = None
    ) -> None:
        self._registry = registry
        self._default = default_estimator or HeuristicUsageEstimator()

    def select(self, step: object) -> UsageEstimator:
        selected = self._registry.resolve(step)
        est = selected or self._default
        # Telemetry hook: record estimator selection
        try:
            telemetry.logfire.debug(
                f"[Estimator] Selected {est.__class__.__name__} for step '{getattr(step, 'name', '<unnamed>')}'"
            )
        except Exception:
            pass
        return est


def build_default_estimator_factory() -> UsageEstimatorFactory:
    """Create a conservative default factory with a small set of useful rules.

    - Validation/adapters: minimal default estimate
    - Known heavy models: heuristic (already conservative)
    - Fallback to heuristic estimator
    """

    registry = EstimatorRegistry()

    class _MinimalEstimator:
        def estimate(self, step: object, data: object, context: object | None) -> UsageEstimate:
            return UsageEstimate(cost_usd=0.0, tokens=0)

    # Rule: adapter/output mapper or validation step → minimal
    def _is_adapter_or_validation(step: object) -> bool:
        try:
            meta = getattr(step, "meta", None)
            if isinstance(meta, dict) and meta.get("is_validation_step", False):
                return True
        except Exception:
            pass
        try:
            # Heuristic: adapter steps often carry meta.is_adapter flag
            meta = getattr(step, "meta", None)
            if isinstance(meta, dict) and meta.get("is_adapter", False):
                return True
        except Exception:
            pass
        # Otherwise, not adapter/validation
        return False

    registry.register(_is_adapter_or_validation, _MinimalEstimator())

    # Optional: Learnable/historical strategy gating via TOML config
    class LearnableUsageEstimator:
        """Phase 2: Learnable/historical estimator.

        Reads historical averages from flujo.toml under:
          [cost.historical.<provider>.<model>]
            avg_cost_usd = <float>
            avg_tokens = <int>
        Falls back to heuristic when no entry is found.
        """

        def __init__(self, fallback: UsageEstimator | None = None) -> None:
            self._fallback = fallback or HeuristicUsageEstimator()

        def estimate(self, step: object, data: object, context: object | None) -> UsageEstimate:
            try:
                cfg = get_config_manager().load_config()
                cost_cfg = getattr(cfg, "cost", None)
                if not isinstance(cost_cfg, dict):
                    return self._fallback.estimate(step, data, context)
                agent = getattr(step, "agent", None)
                provider = getattr(agent, "_provider", None)
                model_name = getattr(agent, "_model_name", None)
                if not isinstance(model_name, str):
                    return self._fallback.estimate(step, data, context)
                prov_key = str(provider) if provider is not None else "default"
                hist = cost_cfg.get("historical", {}).get(prov_key, {}).get(model_name, {})
                ac = hist.get("avg_cost_usd")
                at = hist.get("avg_tokens")
                if ac is not None or at is not None:
                    return UsageEstimate(cost_usd=float(ac or 0.0), tokens=int(at or 0))
            except Exception:
                pass
            return self._fallback.estimate(step, data, context)

    # Gate learnable strategy via config
    try:
        cfg = get_config_manager().load_config()
        cost_cfg = getattr(cfg, "cost", None)
        strategy = None
        if isinstance(cost_cfg, dict):
            # Prefer explicit strategy value; fallback to cost.learnable.enabled flag
            strategy = cost_cfg.get("estimation_strategy")
            learnable_enabled = (
                bool(cost_cfg.get("learnable", {}).get("enabled", False))
                if isinstance(cost_cfg.get("learnable"), dict)
                else False
            )
            use_learnable = (
                isinstance(strategy, str) and strategy.lower() == "learnable"
            ) or learnable_enabled
            if use_learnable:
                learnable = LearnableUsageEstimator()

                def _has_agent(step: object) -> bool:
                    return hasattr(step, "agent") and getattr(step, "agent") is not None

                # Register general rule after minimal-rule so adapters/validation keep minimal
                registry.register(_has_agent, learnable)
                try:
                    telemetry.logfire.info("[Estimator] Learnable strategy enabled via config")
                except Exception:
                    pass
    except Exception:
        pass

    return UsageEstimatorFactory(registry=registry, default_estimator=HeuristicUsageEstimator())

from flujo.application.core.estimation import (
    HeuristicUsageEstimator,
    build_default_estimator_factory,
)
from flujo.domain.models import UsageEstimate


def test_heuristic_returns_minimal_when_no_hints():
    est = HeuristicUsageEstimator()

    class _Step:
        config = type("_Cfg", (), {})()
        agent = type("_Agent", (), {"_provider": None, "_model_name": None})()

    r = est.estimate(_Step(), None, None)
    assert isinstance(r, UsageEstimate)
    assert r.cost_usd >= 0.0
    assert r.tokens >= 0


def test_heuristic_estimator_prefers_config_hints():
    class _Cfg:
        expected_cost_usd = 0.05
        expected_tokens = 123

    class _Step:
        config = _Cfg()
        agent = object()

    est = HeuristicUsageEstimator()
    e = est.estimate(_Step(), None, None)
    assert isinstance(e, UsageEstimate)
    assert abs(e.cost_usd - 0.05) < 1e-9
    assert e.tokens == 123


def test_heuristic_estimator_conservative_bounds_by_model():
    class _Agent:
        _model_name = "gpt-4o"

    class _Step:
        config = type("_Cfg", (), {})()
        agent = _Agent()

    est = HeuristicUsageEstimator()
    e = est.estimate(_Step(), None, None)
    assert e.cost_usd >= 0.10
    assert e.tokens >= 500


def test_heuristic_estimator_defaults_minimal():
    class _Step:
        config = type("_Cfg", (), {})()
        agent = object()

    est = HeuristicUsageEstimator()
    e = est.estimate(_Step(), None, None)
    assert e.cost_usd == 0.0
    assert e.tokens == 0


def test_registry_and_factory_selects_minimal_for_adapter_or_validation():
    # Build default factory with minimal rule for adapter/validation
    factory = build_default_estimator_factory()

    class _Step:
        meta = {"is_adapter": True, "adapter_id": "generic-adapter", "adapter_allow": "generic"}
        config = type("_Cfg", (), {})()
        agent = object()

    est = factory.select(_Step())
    e = est.estimate(_Step(), None, None)
    assert e.cost_usd == 0.0 and e.tokens == 0

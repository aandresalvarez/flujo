from pathlib import Path

import pytest

from flujo.application.core.estimation import build_default_estimator_factory
from flujo.domain.models import UsageEstimate


def test_learnable_estimator_gated_by_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "flujo.toml"
    cfg.write_text(
        """
        [cost]
        estimation_strategy = "learnable"

        [cost.historical.openai."gpt-4o"]
        avg_cost_usd = 0.33
        avg_tokens = 333
        """
    )
    monkeypatch.setenv("FLUJO_CONFIG_PATH", str(cfg))

    class _Agent:
        _provider = "openai"
        _model_name = "gpt-4o"

    class _Step:
        config = type("_Cfg", (), {})()
        agent = _Agent()

    factory = build_default_estimator_factory()
    est = factory.select(_Step())
    result = est.estimate(_Step(), None, None)
    assert isinstance(result, UsageEstimate)
    assert abs(result.cost_usd - 0.33) < 1e-9
    assert result.tokens == 333

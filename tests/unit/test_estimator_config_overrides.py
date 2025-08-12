from pathlib import Path

import pytest

from flujo.application.core.estimation import HeuristicUsageEstimator
from flujo.domain.models import UsageEstimate


@pytest.mark.asyncio
async def test_toml_config_overrides_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Write a temporary flujo.toml with provider/model-specific overrides
    cfg = tmp_path / "flujo.toml"
    cfg.write_text(
        """
        [cost.estimators.openai."gpt-4o"]
        expected_cost_usd = 0.77
        expected_tokens = 777
        """
    )

    # Point config manager to this file
    monkeypatch.setenv("FLUJO_CONFIG_PATH", str(cfg))

    class _Agent:
        _provider = "openai"
        _model_name = "gpt-4o"

    class _Step:
        config = type("_Cfg", (), {})()
        agent = _Agent()

    est = HeuristicUsageEstimator()
    result = est.estimate(_Step(), None, None)
    assert isinstance(result, UsageEstimate)
    assert abs(result.cost_usd - 0.77) < 1e-9
    assert result.tokens == 777

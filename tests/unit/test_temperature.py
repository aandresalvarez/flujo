from flujo.application.temperature import temp_for_round
import sys
import importlib


def test_temp_schedule_from_settings(monkeypatch) -> None:
    # It should use the default schedule from settings
    assert temp_for_round(0) == 1.0
    assert temp_for_round(3) == 0.2
    assert temp_for_round(99) == 0.2  # Last value is sticky

    # It should respect an override from settings
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module.settings, "t_schedule", [0.9, 0.1])
    # Reload the temperature module to pick up the new settings
    importlib.reload(sys.modules["flujo.application.temperature"])
    from flujo.application.temperature import temp_for_round as temp_for_round_new

    assert temp_for_round_new(0) == 0.9
    assert temp_for_round_new(1) == 0.1
    assert temp_for_round_new(2) == 0.1


def test_temperature(monkeypatch) -> None:
    # Set t_schedule explicitly for this test
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module.settings, "t_schedule", [0.1, 0.1, 0.1])
    # Reload the temperature module to pick up the new settings
    importlib.reload(sys.modules["flujo.application.temperature"])
    from flujo.application.temperature import temp_for_round as temp_for_round_new

    assert temp_for_round_new(0) == 0.1
    assert temp_for_round_new(1) == 0.1
    assert temp_for_round_new(2) == 0.1

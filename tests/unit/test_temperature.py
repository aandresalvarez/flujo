from flujo.application.temperature import temp_for_round
from flujo.infra.settings import settings


def test_temp_schedule_from_settings(monkeypatch) -> None:
    # It should use the default schedule from settings
    assert temp_for_round(0) == 1.0
    assert temp_for_round(3) == 0.2
    assert temp_for_round(99) == 0.2  # Last value is sticky

    # It should respect an override from settings
    monkeypatch.setattr(settings, "t_schedule", [0.9, 0.1])
    assert temp_for_round(0) == 0.9
    assert temp_for_round(1) == 0.1
    assert temp_for_round(2) == 0.1


def test_temperature(monkeypatch) -> None:
    # Set t_schedule explicitly for this test
    monkeypatch.setattr(settings, "t_schedule", [0.1, 0.1, 0.1])
    assert temp_for_round(0) == 0.1
    assert temp_for_round(1) == 0.1
    assert temp_for_round(2) == 0.1

from __future__ import annotations


from flujo.utils.config import get_settings


def test_get_settings_reflects_env_flags(monkeypatch) -> None:
    monkeypatch.setenv("FLUJO_TEST_MODE", "1")
    monkeypatch.setenv("FLUJO_WARN_LEGACY", "1")
    s = get_settings()
    assert s.test_mode is True
    assert s.warn_legacy is True
    # pure_quota_mode is always True by design
    assert s.pure_quota_mode is True

    # Toggle flags and ensure subsequent calls reflect changes (no caching)
    monkeypatch.setenv("FLUJO_TEST_MODE", "0")
    monkeypatch.delenv("FLUJO_WARN_LEGACY", raising=False)
    s2 = get_settings()
    assert s2.test_mode is False
    assert s2.warn_legacy is False
    assert s2.pure_quota_mode is True

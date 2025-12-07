import pytest

from flujo.domain.blueprint.loader_resolution import _import_object
from flujo.domain.blueprint.loader_models import BlueprintError


def test_blueprint_imports_default_deny(monkeypatch):
    """YAML blueprint imports should be denied unless explicitly allowed."""

    # Ensure no allow-list configured
    monkeypatch.delenv("FLUJO_CONFIG_PATH", raising=False)

    with pytest.raises(BlueprintError):
        _import_object("os:system")

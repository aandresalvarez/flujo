from unittest import mock
import pytest
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        allowed = ["tests", "flujo", "skills", "imports"]
        mock_config.blueprint_allowed_imports = allowed
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = allowed

        mock_get.return_value.load_config.return_value = mock_config
        yield

from __future__ import annotations

import pytest

import flujo.utils.prompting as prompting_mod
from flujo.utils.template_vars import render_template


def test_allowlist_permits_only_enabled_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Allow only 'upper'
    monkeypatch.setattr(prompting_mod, "_get_enabled_filters", lambda: {"upper"}, raising=True)
    assert render_template("{{ 'x' | upper }}") == "X"
    with pytest.raises(ValueError):
        _ = render_template("{{ 'X' | lower }}")

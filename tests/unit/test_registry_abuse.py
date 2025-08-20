from __future__ import annotations

import pytest


def test_register_non_step_raises() -> None:
    from flujo.framework import registry

    class NotAStep:  # type: ignore[too-few-public-methods]
        kind = "Nope"

    with pytest.raises(Exception):
        registry.register_step_type(NotAStep)  # type: ignore[arg-type]


def test_unknown_yaml_kind_raises() -> None:
    from flujo.domain.dsl import Pipeline
    from flujo.domain.blueprint.loader import BlueprintError

    bad = 'version: "0.1"\nsteps:\n  - kind: NotRegistered\n    name: X\n'
    with pytest.raises(BlueprintError):
        Pipeline.from_yaml_text(bad)

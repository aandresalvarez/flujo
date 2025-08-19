import pytest


def test_welcome_agent_is_registered() -> None:
    # Explicitly register builtins to avoid relying on import side effects
    from flujo.builtins import _register_builtins

    _register_builtins()

    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    entry = reg.get("flujo.builtins.welcome_agent")
    assert entry is not None, "welcome_agent should be registered in the skill registry"

    # Validate basic metadata
    assert entry.get("side_effects") is False
    schema = entry.get("input_schema") or {}
    props = schema.get("properties") or {}
    assert "name" in props
    assert props["name"].get("type") == "string"


@pytest.mark.asyncio
async def test_welcome_agent_factory_executes_without_optional_dependency() -> None:
    # Import after potentially unsetting PYFIGLET to simulate missing optional dep
    import importlib
    import flujo.builtins as builtins
    from flujo.infra.skill_registry import get_skill_registry

    # Force re-import to ensure optional import path runs
    importlib.reload(builtins)

    reg = get_skill_registry()
    entry = reg.get("flujo.builtins.welcome_agent")
    assert entry is not None

    factory = entry["factory"]
    func = factory()
    result = await func()  # default name="Developer"
    assert isinstance(result, str)
    assert "Welcome, Developer!" in result


def test_template_pipeline_yaml_loads() -> None:
    from importlib.resources import files
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    template_path = files("flujo.templates.project") / "pipeline.yaml"
    yaml_text = template_path.read_text(encoding="utf-8")

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert pipeline is not None
    assert len(pipeline.steps) == 1
    step = pipeline.steps[0]
    assert step.name == "welcome_message"

    # Validate static graph; should be valid
    report = pipeline.validate_graph()
    assert report.is_valid, f"Unexpected validation errors: {report.errors}"

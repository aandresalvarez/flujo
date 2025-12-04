"""Tests for ExceptionLinter (V-EX1) - Control flow exception handling validation."""

import pytest

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
from flujo.validation.linters import run_linters


class TestExceptionLinterDetectsCustomSkills:
    """Test that V-EX1 warns about custom skills that might catch control flow exceptions."""

    def test_vex1_detects_custom_skill_with_colon_syntax(self):
        """Test that V-EX1 detects custom skills using module:function syntax."""
        yaml_content = """
version: "0.1"
name: "test_custom_skill"
steps:
  - kind: step
    name: "custom_processor"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        # Should warn about control flow exception handling
        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) > 0, "V-EX1 should detect custom skill"

        finding = vex1_findings[0]
        assert "custom_processor" in finding.message
        assert "test_skills_for_vex1" in finding.message
        assert "PausedException" in finding.message
        assert "must re-raise" in finding.suggestion or "CRITICAL" in finding.suggestion

    def test_vex1_ignores_builtin_skills(self):
        """Test that V-EX1 does not warn for flujo built-in skills."""
        yaml_content = """
version: "0.1"
name: "test_builtin"
steps:
  - kind: step
    name: "builtin_processor"
    agent:
      id: "flujo.builtins.passthrough"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        # Should NOT warn for built-in skills
        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) == 0, "V-EX1 should not detect built-in skills"

    def test_vex1_ignores_declarative_agents(self):
        """Test that V-EX1 does not warn for declarative agents (agents.*)."""
        yaml_content = """
version: "0.1"
name: "test_declarative"
agents:
  my_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: "You are helpful"
    output_schema:
      type: string
steps:
  - kind: step
    name: "declarative_step"
    uses: agents.my_agent
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        # Should NOT warn for declarative agents
        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) == 0, "V-EX1 should not detect declarative agents"


class TestExceptionLinterMessageQuality:
    """Test that V-EX1 provides helpful messages and suggestions."""

    def test_vex1_message_includes_all_control_flow_exceptions(self):
        """Test that the warning message lists all control flow exceptions."""
        yaml_content = """
version: "0.1"
name: "test_message_quality"
steps:
  - kind: step
    name: "custom_step"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) > 0

        finding = vex1_findings[0]
        message = finding.message.lower()

        # Should mention all three control flow exceptions
        assert "pausedexception" in message
        assert "pipelineabortsignal" in message or "abort" in message
        assert "infiniteredirecterror" in message or "redirect" in message

    def test_vex1_suggestion_includes_code_example(self):
        """Test that the suggestion includes a helpful code example."""
        yaml_content = """
version: "0.1"
name: "test_suggestion"
steps:
  - kind: step
    name: "custom_step"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) > 0

        finding = vex1_findings[0]
        suggestion = finding.suggestion

        # Suggestion should include exception handling pattern
        assert "except" in suggestion
        assert "raise" in suggestion
        assert "PausedException" in suggestion or "control flow" in suggestion.lower()

        # Should reference documentation
        assert "FLUJO_TEAM_GUIDE" in suggestion or "Fatal Anti-Pattern" in suggestion

    def test_vex1_includes_step_name_and_location(self):
        """Test that V-EX1 includes step name and location information."""
        yaml_content = """
version: "0.1"
name: "test_location"
steps:
  - kind: step
    name: "my_custom_step"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) > 0

        finding = vex1_findings[0]

        # Should include step name
        assert finding.step_name == "my_custom_step"

        # Should include location information
        assert finding.location_path is not None
        assert "steps" in finding.location_path


class TestExceptionLinterCanBeSuppressed:
    """Test that V-EX1 can be suppressed via standard mechanisms."""

    @pytest.mark.skip(
        reason="Feature request: inline suppression via YAML comments not yet implemented"
    )
    def test_vex1_respects_inline_suppression(self):
        """Test that V-EX1 can be suppressed with inline comments.

        This test is currently skipped because the inline suppression mechanism
        is not yet implemented. When implemented, this test should verify that:
        1. The suppression comment is parsed from YAML
        2. The rule is added to step.meta['suppress_rules']
        3. The linter skips V-EX1 for that step
        """
        yaml_content = """
version: "0.1"
name: "test_suppression"
steps:
  - kind: step  # flujo: ignore V-EX1
    name: "custom_step_suppressed"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        # V-EX1 should be suppressed via inline comment
        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) == 0, "V-EX1 should be suppressed with inline comment"


class TestExceptionLinterMultipleSteps:
    """Test V-EX1 behavior with multiple steps."""

    def test_vex1_detects_multiple_custom_skills(self):
        """Test that V-EX1 detects multiple custom skills in a pipeline."""
        yaml_content = """
version: "0.1"
name: "test_multiple"
steps:
  - kind: step
    name: "custom_1"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
  - kind: step
    name: "builtin"
    agent:
      id: "flujo.builtins.passthrough"
  - kind: step
    name: "custom_2"
    uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        report = run_linters(pipeline)

        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]

        # Should detect both custom skills, not the builtin
        assert len(vex1_findings) == 2, "Should detect both custom skills"

        step_names = {f.step_name for f in vex1_findings}
        assert "custom_1" in step_names
        assert "custom_2" in step_names
        assert "builtin" not in step_names


class TestExceptionLinterEdgeCases:
    """Test V-EX1 edge cases and error handling."""

    def test_vex1_handles_steps_without_meta(self):
        """Test that V-EX1 gracefully handles steps without metadata."""
        # This tests the internal error handling - if a step doesn't have meta,
        # the linter should continue without crashing
        from flujo.domain.dsl import Step, Pipeline

        # Create a step without proper meta
        step = Step(name="test_step")
        pipeline = Pipeline(steps=[step])

        # Should not crash
        report = run_linters(pipeline)

        # Verify it completes successfully (no crash)
        assert report is not None

    def test_vex1_handles_empty_pipeline(self):
        """Test that V-EX1 handles empty pipelines gracefully."""
        from flujo.domain.dsl import Pipeline

        pipeline = Pipeline(steps=[])

        # Should not crash
        report = run_linters(pipeline)

        # No findings for empty pipeline
        vex1_findings = [f for f in report.warnings if f.rule_id == "V-EX1"]
        assert len(vex1_findings) == 0

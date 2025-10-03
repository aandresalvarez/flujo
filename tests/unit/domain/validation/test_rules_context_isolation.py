"""Tests for V-CTX1: Context isolation validation in loop/parallel steps.

This tests the OrchestrationLinter's ability to detect custom skills in loop and
parallel steps that might mutate shared context without isolation.
"""

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.validation.linters import run_linters


def test_vctx1_detects_custom_skill_in_loop():
    """V-CTX1 should warn when loop contains custom skill."""
    yaml_content = """
version: "0.1"
name: "loop_with_custom_skill"

steps:
  - kind: loop
    name: test_loop
    loop:
      max_loops: 5
      body:
        - kind: step
          name: process_item
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "{{ loop.iteration }}"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    # Should have a V-CTX1 warning for the loop
    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 1, f"Expected 1 V-CTX1 warning, got {len(v_ctx1_findings)}"
    assert "test_loop" in v_ctx1_findings[0].message
    assert "LoopStep" in v_ctx1_findings[0].message
    assert "ContextManager.isolate()" in v_ctx1_findings[0].suggestion


def test_vctx1_passes_for_builtin_in_loop():
    """V-CTX1 should NOT warn when loop only contains built-in skills."""
    yaml_content = """
version: "0.1"
name: "loop_with_builtin"

steps:
  - kind: loop
    name: test_loop
    loop:
      max_loops: 3
      body:
        - kind: step
          name: passthrough
          agent: { id: "flujo.builtins.passthrough" }
          input: "{{ loop.iteration }}"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    # Should NOT have V-CTX1 warnings for built-ins
    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 0, (
        f"Expected 0 V-CTX1 warnings for built-ins, got {len(v_ctx1_findings)}"
    )


def test_vctx1_detects_custom_skill_in_parallel():
    """V-CTX1 should warn when parallel step contains custom skills."""
    yaml_content = """
version: "0.1"
name: "parallel_with_custom_skill"

steps:
  - kind: parallel
    name: test_parallel
    branches:
      branch_a:
        - kind: step
          name: custom_a
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "data_a"
      branch_b:
        - kind: step
          name: custom_b
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
          input: "data_b"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    # Should have V-CTX1 warning for parallel
    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 1, f"Expected 1 V-CTX1 warning, got {len(v_ctx1_findings)}"
    assert "test_parallel" in v_ctx1_findings[0].message
    assert "ParallelStep" in v_ctx1_findings[0].message
    assert "branch_a" in v_ctx1_findings[0].message or "branch_b" in v_ctx1_findings[0].message


def test_vctx1_passes_for_agents_in_parallel():
    """V-CTX1 should NOT warn when parallel contains only declarative agents."""
    yaml_content = """
version: "0.1"
name: "parallel_with_agents"

agents:
  my_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: "You are helpful"
    output_schema:
      type: string

steps:
  - kind: parallel
    name: test_parallel
    branches:
      branch_a:
        - kind: step
          name: agent_a
          uses: agents.my_agent
          input: "question_a"
      branch_b:
        - kind: step
          name: agent_b
          uses: agents.my_agent
          input: "question_b"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    # Should NOT have V-CTX1 warnings for declarative agents
    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 0, (
        f"Expected 0 V-CTX1 warnings for agents, got {len(v_ctx1_findings)}"
    )


def test_vctx1_message_shows_skill_references():
    """V-CTX1 message should show which custom skills were detected."""
    yaml_content = """
version: "0.1"
name: "loop_with_multiple_skills"

steps:
  - kind: loop
    name: multi_skill_loop
    loop:
      max_loops: 3
      body:
        - kind: step
          name: skill_1
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "{{ loop.iteration }}"
        - kind: step
          name: skill_2
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
          input: "{{ loop.iteration }}"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 1
    # Message should list detected custom skills
    assert (
        "custom_skill_example" in v_ctx1_findings[0].message
        or "custom_sync_skill" in v_ctx1_findings[0].message
    )


def test_vctx1_mixed_skills_in_parallel_branches():
    """V-CTX1 should detect custom skills across multiple branches."""
    yaml_content = """
version: "0.1"
name: "parallel_mixed"

agents:
  my_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: "Helpful assistant"
    output_schema: { type: string }

steps:
  - kind: parallel
    name: mixed_parallel
    branches:
      builtin_branch:
        - kind: step
          name: passthrough
          agent: { id: "flujo.builtins.passthrough" }
          input: "data"
      agent_branch:
        - kind: step
          name: agent_step
          uses: agents.my_agent
          input: "question"
      custom_branch:
        - kind: step
          name: custom_step
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "data"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    # Should detect the custom skill in custom_branch
    assert len(v_ctx1_findings) == 1
    assert "mixed_parallel" in v_ctx1_findings[0].message
    assert "custom_branch" in v_ctx1_findings[0].message


def test_vctx1_suggestion_references_team_guide():
    """V-CTX1 suggestion should reference FLUJO_TEAM_GUIDE.md Section 3.5."""
    yaml_content = """
version: "0.1"
name: "loop_with_custom"

steps:
  - kind: loop
    name: test_loop
    loop:
      max_loops: 2
      body:
        - kind: step
          name: custom
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
          input: "data"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 1
    # Suggestion should reference the team guide
    assert "FLUJO_TEAM_GUIDE.md" in v_ctx1_findings[0].suggestion
    assert "Section 3.5" in v_ctx1_findings[0].suggestion
    assert "Idempotency" in v_ctx1_findings[0].suggestion


def test_vctx1_no_warnings_for_simple_loop():
    """V-CTX1 should NOT warn when loop has no custom skills."""
    yaml_content = """
version: "0.1"
name: "simple_loop"

agents:
  simple_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: "Count"
    output_schema: { type: number }

steps:
  - kind: loop
    name: counter_loop
    loop:
      max_loops: 5
      body:
        - kind: step
          name: count
          uses: agents.simple_agent
          input: "{{ loop.iteration }}"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 0


def test_vctx1_multiple_custom_skills_in_loop():
    """V-CTX1 should show up to 3 skill references in the message."""
    yaml_content = """
version: "0.1"
name: "loop_with_many_skills"

steps:
  - kind: loop
    name: multi_skill_loop
    loop:
      max_loops: 3
      body:
        - kind: step
          name: skill_1
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "data"
        - kind: step
          name: skill_2
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_sync_skill"
          input: "data"
        - kind: step
          name: skill_3
          uses: "tests.unit.domain.validation.test_skills_for_vex1:custom_skill_example"
          input: "data"
    """

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    report = run_linters(pipeline)

    v_ctx1_findings = [f for f in report.warnings if f.rule_id == "V-CTX1"]
    assert len(v_ctx1_findings) == 1
    # Message should list detected custom skills (up to 3)
    message = v_ctx1_findings[0].message
    # Should contain at least one of the custom skills
    assert "custom_skill_example" in message or "custom_sync_skill" in message

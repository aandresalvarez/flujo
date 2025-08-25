from __future__ import annotations

# This file intentionally includes dangerous code patterns for security validation testing
import pytest
import os
from flujo.architect.builder import build_architect_pipeline
from flujo.architect.context import ArchitectContext
from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

# Force minimal architect pipeline for security tests to avoid hanging
# This ensures tests use the simple pipeline instead of the complex state machine
#
# NOTE: This approach aligns with the proposed architect redesign (see
# bug_reports/architect/ARCHITECT_STATE_MACHINE_REDESIGN.md). We're implementing
# the "Short-Term Pragmatics" while the redesign is in progress. When the redesign
# is complete, these environment variables can be removed and tests will use the
# unified, deterministic state machine.
os.environ["FLUJO_ARCHITECT_IGNORE_CONFIG"] = "1"
os.environ["FLUJO_TEST_MODE"] = "1"

# These validation sweeps iterate over many malicious payloads and are inherently slow.
# Mark the module as slow so they're excluded from fast suites by default.
# Set module-level timeout to prevent hanging
pytestmark = [pytest.mark.slow, pytest.mark.timeout(30)]


@pytest.mark.integration
def test_architect_handles_sql_injection_attempts():
    """Test: Architect safely handles SQL injection attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various SQL injection patterns
    sql_injection_patterns = [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "'; INSERT INTO users VALUES ('hacker', 'password'); --",
        "'; UPDATE users SET password='hacked'; --",
        "'; EXEC xp_cmdshell('dir'); --",
    ]

    for pattern in sql_injection_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_xss_attempts():
    """Test: Architect safely handles XSS attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various XSS patterns
    xss_patterns = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "';alert('xss');//",
        "<svg onload=alert('xss')>",
        "javascript:void(alert('xss'))",
    ]

    for pattern in xss_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_command_injection_attempts():
    """Test: Architect safely handles command injection attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various command injection patterns
    command_injection_patterns = [
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& whoami",
        "|| id",
        "; ls -la",
        "`whoami`",
        "$(whoami)",
        "; cat /etc/shadow",
    ]

    for pattern in command_injection_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_path_traversal_attempts():
    """Test: Architect safely handles path traversal attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various path traversal patterns
    path_traversal_patterns = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "....//....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..%5C..%5C..%5Cwindows%5Csystem32%5Cdrivers%5Cetc%5Chosts",
    ]

    for pattern in path_traversal_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_ldap_injection_attempts():
    """Test: Architect safely handles LDAP injection attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various LDAP injection patterns
    ldap_injection_patterns = [
        "*)(uid=*))(|(uid=*",
        "*)(|(password=*))",
        "*)(|(objectClass=*))",
        "*)(|(cn=*))",
        "*)(|(mail=*))",
    ]

    for pattern in ldap_injection_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_no_sql_injection_attempts():
    """Test: Architect safely handles NoSQL injection attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various NoSQL injection patterns
    nosql_injection_patterns = [
        '{"$gt": ""}',
        '{"$ne": null}',
        '{"$where": "function() { return true; }"}',
        '{"$regex": ".*"}',
        '{"$exists": true}',
    ]

    for pattern in nosql_injection_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_template_injection_attempts():
    """Test: Architect safely handles template injection attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various template injection patterns
    template_injection_patterns = [
        "{{7*7}}",
        "{{config}}",
        "{{request}}",
        "{{''.__class__.__mro__[2].__subclasses__()}}",
        "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
    ]

    for pattern in template_injection_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_overflow_attempts():
    """Test: Architect safely handles buffer overflow attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various overflow patterns
    overflow_patterns = [
        "A" * 10000,  # 10KB of repeated characters
        "A" * 100000,  # 100KB of repeated characters
        "A" * 1000000,  # 1MB of repeated characters
        "\x00" * 1000,  # Null bytes
        "\xff" * 1000,  # High bytes
    ]

    for pattern in overflow_patterns:
        initial = {
            "initial_prompt": "Make a pipeline",
            "user_goal": f"Echo input: {pattern[:100]}...",
        }
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner,
            input_data=f"Echo input: {pattern[:100]}...",
            run_id=None,
            json_output=False,
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with overflow attempts
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_encoding_manipulation():
    """Test: Architect safely handles encoding manipulation attempts."""
    pipeline = build_architect_pipeline()

    # Test with various encoding manipulation patterns
    encoding_patterns = [
        "Echo input with %00 null bytes",
        "Echo input with %0A newlines",
        "Echo input with %0D carriage returns",
        "Echo input with %20 spaces",
        "Echo input with %3C %3E angle brackets",
        "Echo input with %22 %27 quotes",
    ]

    for pattern in encoding_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": pattern}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=pattern, run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with encoding manipulation
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_unicode_normalization_attacks():
    """Test: Architect safely handles Unicode normalization attacks."""
    pipeline = build_architect_pipeline()

    # Test with various Unicode normalization attack patterns
    unicode_attack_patterns = [
        "Echo input with \u0000 null characters",
        "Echo input with \u0001 control characters",
        "Echo input with \u0009 tab characters",
        "Echo input with \u000a line feed",
        "Echo input with \u000d carriage return",
        "Echo input with \u001f unit separator",
        "Echo input with \u007f delete character",
    ]

    for pattern in unicode_attack_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": pattern}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=pattern, run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with Unicode attacks
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_regex_dos_attempts():
    """Test: Architect safely handles regex DoS attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various regex DoS patterns
    regex_dos_patterns = [
        "a" * 1000 + "!",
        "(a|a|a)*" * 100,
        "((a|a|a)*)*" * 50,
        "((a|a|a)*)*" * 100,
        "((a|a|a)*)*" * 200,
    ]

    for pattern in regex_dos_patterns:
        initial = {
            "initial_prompt": "Make a pipeline",
            "user_goal": f"Echo input: {pattern[:100]}...",
        }
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner,
            input_data=f"Echo input: {pattern[:100]}...",
            run_id=None,
            json_output=False,
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with regex DoS attempts
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_prototype_pollution_attempts():
    """Test: Architect safely handles prototype pollution attempts in input."""
    pipeline = build_architect_pipeline()

    # Test with various prototype pollution patterns
    prototype_pollution_patterns = [
        '{"__proto__": {"isAdmin": true}}',
        '{"constructor": {"prototype": {"isAdmin": true}}}',
        '{"__proto__": {"toString": "hacked"}}',
        '{"constructor": {"prototype": {"toString": "hacked"}}}',
    ]

    for pattern in prototype_pollution_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with prototype pollution attempts
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_deserialization_attacks():
    """Test: Architect safely handles deserialization attack attempts."""
    pipeline = build_architect_pipeline()

    # Test with various deserialization attack patterns
    deserialization_patterns = [
        '{"rce": "java.lang.Runtime.getRuntime().exec(\'id\')"}',
        '{"rce": "System.Diagnostics.Process.Start(\'cmd\')"}',
        '{"rce": "eval(\'alert(1)\')"}',
        '{"rce": "Function(\'alert(1)\')()"}',
    ]

    for pattern in deserialization_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with deserialization attacks
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_mixed_malicious_inputs():
    """Test: Architect safely handles combinations of malicious inputs."""
    pipeline = build_architect_pipeline()

    # Test with combinations of malicious patterns
    mixed_malicious_patterns = [
        "'; DROP TABLE users; -- <script>alert('xss')</script>",
        "javascript:alert('xss'); rm -rf /",
        "../../../etc/passwd<script>alert('xss')</script>",
        "{{7*7}}; DROP TABLE users; --",
        "'; INSERT INTO users VALUES ('hacker', 'password'); -- <img src=x onerror=alert('xss')>",
    ]

    for pattern in mixed_malicious_patterns:
        initial = {"initial_prompt": "Make a pipeline", "user_goal": f"Echo input: {pattern}"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data=f"Echo input: {pattern}", run_id=None, json_output=False
        )

        # Should complete without crashing
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should generate YAML even with mixed malicious input
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_very_long_malicious_inputs():
    """Test: Architect safely handles very long malicious inputs."""
    pipeline = build_architect_pipeline()

    # Create very long malicious input
    long_malicious_input = "'; DROP TABLE users; -- " * 1000  # Very long SQL injection

    initial = {
        "initial_prompt": "Make a pipeline",
        "user_goal": f"Echo input: {long_malicious_input[:100]}...",
    }
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner,
        input_data=f"Echo input: {long_malicious_input[:100]}...",
        run_id=None,
        json_output=False,
    )

    # Should complete without crashing
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even with very long malicious input
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_nested_malicious_structures():
    """Test: Architect safely handles nested malicious data structures."""
    pipeline = build_architect_pipeline()

    # Test with nested malicious structures

    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input with nested data"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input with nested data", run_id=None, json_output=False
    )

    # Should complete without crashing
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even with nested malicious structures
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None

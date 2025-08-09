"""Tests for flujo.validation module."""

import pytest
from typing import Any, Optional, Tuple
from pydantic import BaseModel

from flujo.domain.validation import BaseValidator, validator
from flujo.domain.validation import ValidationResult


class MockContext(BaseModel):
    """Mock context for testing."""

    value: str = "test"


# Validator functions for testing (not test functions)
def validator_function_valid(value: Any) -> Tuple[bool, Optional[str]]:
    """Validator function that returns valid."""
    return True, "Valid"


def validator_function_invalid(value: Any) -> Tuple[bool, Optional[str]]:
    """Validator function that returns invalid."""
    return False, "Invalid"


def validator_function_exception(value: Any) -> Tuple[bool, Optional[str]]:
    """Validator function that raises an exception."""
    raise ValueError("Test exception")


@pytest.fixture
def simple_validator():
    """Fixture providing a simple validator class for testing."""

    class SimpleValidator(BaseValidator):
        async def validate(
            self,
            output_to_check: Any,
            *,
            context: Optional[BaseModel] = None,
        ) -> ValidationResult:
            if isinstance(output_to_check, str) and output_to_check == "valid":
                return ValidationResult(
                    is_valid=True,
                    feedback="Valid output",
                    validator_name=self.name,
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    feedback="Invalid output",
                    validator_name=self.name,
                )

    return SimpleValidator


@pytest.mark.asyncio
async def test_base_validator_initialization(simple_validator):
    """Test BaseValidator initialization."""

    validator = simple_validator()
    assert validator.name == "SimpleValidator"

    validator_with_name = simple_validator("CustomName")
    assert validator_with_name.name == "CustomName"


@pytest.mark.asyncio
async def test_base_validator_validate_valid(simple_validator):
    """Test BaseValidator validate method with valid input."""

    validator = simple_validator()
    result = await validator.validate("valid")

    assert result.is_valid is True
    assert result.feedback == "Valid output"
    assert result.validator_name == "SimpleValidator"


@pytest.mark.asyncio
async def test_base_validator_validate_invalid(simple_validator):
    """Test BaseValidator validate method with invalid input."""

    validator = simple_validator()
    result = await validator.validate("invalid")

    assert result.is_valid is False
    assert result.feedback == "Invalid output"
    assert result.validator_name == "SimpleValidator"


@pytest.mark.asyncio
async def test_base_validator_validate_with_context(simple_validator):
    """Test BaseValidator validate method with context."""

    validator = simple_validator()
    context = MockContext(value="test")
    result = await validator.validate("valid", context=context)

    assert result.is_valid is True
    assert result.feedback == "Valid output"
    assert result.validator_name == "SimpleValidator"


@pytest.mark.asyncio
async def test_validator_decorator_valid():
    """Test validator decorator with valid function."""
    decorated_validator = validator(validator_function_valid)

    result = await decorated_validator.validate("test")
    assert result.is_valid is True
    assert result.feedback == "Valid"
    assert result.validator_name == "validator_function_valid"


@pytest.mark.asyncio
async def test_validator_decorator_invalid():
    """Test validator decorator with invalid function."""
    decorated_validator = validator(validator_function_invalid)

    result = await decorated_validator.validate("test")
    assert result.is_valid is False
    assert result.feedback == "Invalid"
    assert result.validator_name == "validator_function_invalid"


@pytest.mark.asyncio
async def test_validator_decorator_exception():
    """Test validator decorator with function that raises exception."""
    decorated_validator = validator(validator_function_exception)

    result = await decorated_validator.validate("test")
    assert result.is_valid is False
    assert "Validator function raised an exception" in result.feedback
    assert "Test exception" in result.feedback
    assert result.validator_name == "validator_function_exception"


@pytest.mark.asyncio
async def test_validator_decorator_with_context():
    """Test validator decorator with context."""
    decorated_validator = validator(validator_function_valid)
    context = MockContext(value="test")

    result = await decorated_validator.validate("test", context=context)
    assert result.is_valid is True
    assert result.feedback == "Valid"
    assert result.validator_name == "validator_function_valid"


def test_validator_decorator_creates_functional_validator():
    """Test that validator decorator creates FunctionalValidator."""
    decorated_validator = validator(validator_function_valid)

    assert hasattr(decorated_validator, "validate")
    assert callable(decorated_validator.validate)
    assert decorated_validator.name == "validator_function_valid"

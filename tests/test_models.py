
import pytest
from pydantic import ValidationError
from models import GeneratorResult


def valid_test_case():
    return {
        "id": "TC-001",
        "title": "Verify successful login",
        "priority": "P1",
        "steps": [
            "Open login page",
            "Enter username",
            "Enter password"
        ],
        "expected_result": "User successfully lands on home page."
    }


def test_ready_status_with_test_cases():
    """READY with test cases should be accepted."""

    result = GeneratorResult(
        status="READY",
        missing_information=[],
        test_cases=[valid_test_case()]
    )

    assert isinstance(result, GeneratorResult)
    assert len(result.test_cases) == 1


def test_ready_status_with_empty_test_cases():
    """READY without test cases should raise ValidationError."""

    with pytest.raises(ValidationError, match="READY requires"):
        GeneratorResult(
            status="READY",
            missing_information=[],
            test_cases=[]
        )


def test_ready_status_with_missing_information():
    """READY with missing information should be rejected."""

    with pytest.raises(ValidationError, match="READY cannot contain"):
        GeneratorResult(
            status="READY",
            missing_information=["Authentication details unavailable."],
            test_cases=[valid_test_case()]
        )


def test_blocked_status_without_clarification():
    """BLOCKED without clarification questions should be rejected."""

    with pytest.raises(ValidationError, match="BLOCKED requires"):
        GeneratorResult(
            status="BLOCKED",
            missing_information=[],
            test_cases=[]
        )


def test_blocked_status_with_test_cases():
    """BLOCKED containing test cases should be rejected."""

    with pytest.raises(ValidationError, match="BLOCKED cannot contain"):
        GeneratorResult(
            status="BLOCKED",
            missing_information=["Authentication mechanism is unknown."],
            test_cases=[valid_test_case()]
        )


def test_blocked_status_with_clarification_and_no_tests():
    """BLOCKED with clarification and no test cases should be accepted."""

    result = GeneratorResult(
        status="BLOCKED",
        missing_information=["Forgot Password mechanism is unknown."],
        test_cases=[]
    )

    assert isinstance(result, GeneratorResult)
    assert result.status == "BLOCKED"
    assert result.test_cases == []


def test_ready_status_with_invalid_priority():
    """Invalid priority should raise ValidationError."""

    invalid_test_case = valid_test_case()
    invalid_test_case["priority"] = "high"

    with pytest.raises(ValidationError, match="priority"):
        GeneratorResult(
            status="READY",
            missing_information=[],
            test_cases=[invalid_test_case]
        )



import sys
import json

import pytest
from unittest.mock import Mock

import generate as cli

from models import GeneratorResult, TestCase as TestCaseModel

from pathlib import Path


# --------------------------------------------------
# Helper: Configure CLI arguments
# --------------------------------------------------

def set_cli_arguments(monkeypatch, requirement_file):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(requirement_file)]
    )


# --------------------------------------------------
# Test 1: READY response -> Exit code 0
# --------------------------------------------------

def test_cli_returns_zero_for_ready_result(tmp_path, monkeypatch, capsys):

    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text(
        "User should be able to log in.",
        encoding="utf-8"
    )

    ready_result = GeneratorResult(
        status="READY",
        missing_information=[],
        test_cases=[
            TestCaseModel(
                id="TC-0001",
                title="Login with valid credentials",
                priority="P1",
                steps=["Enter valid credentials", "Click Login"],
                expected_result="User logs in successfully."
            )
        ]
    )

    mock_generate = Mock(return_value=ready_result)

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["status"] == "READY"
    assert len(output["test_cases"]) == 1
    assert captured.err == ""

    mock_generate.assert_called_once_with(
        "User should be able to log in."
    )


# --------------------------------------------------
# Test 2: BLOCKED response -> Exit code 3
# --------------------------------------------------

def test_cli_returns_three_for_blocked_result(tmp_path, monkeypatch, capsys):

    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text(
        "User should be able to log in.",
        encoding="utf-8"
    )

    blocked_result = GeneratorResult(
        status="BLOCKED",
        missing_information=[
            "What authentication method should be supported?",
            "What should happen after three failed login attempts?"
        ],
        test_cases=[]
    )

    mock_generate = Mock(return_value=blocked_result)

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 3

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert output["status"] == "BLOCKED"
    assert len(output["missing_information"]) == 2
    assert output["test_cases"] == []
    assert captured.err == ""

    mock_generate.assert_called_once_with(
        "User should be able to log in."
    )


# --------------------------------------------------
# Test 3: Missing requirement file -> Exit code 2
# --------------------------------------------------

def test_cli_returns_two_for_missing_file(tmp_path, monkeypatch, capsys):

    requirement_file = tmp_path / "missing.txt"

    mock_generate = Mock()

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 2

    captured = capsys.readouterr()

    assert "Requirement file not found" in captured.err
    assert captured.out == ""

    mock_generate.assert_not_called()


# --------------------------------------------------
# Test 4: Empty requirement file -> Exit code 2
# --------------------------------------------------

@pytest.mark.parametrize(
    "file_content",
    ["", "   \n\t  "]
)
def test_cli_returns_two_for_empty_file(
    tmp_path, monkeypatch, capsys, file_content
):

    requirement_file = tmp_path / "empty.txt"
    requirement_file.write_text(file_content, encoding="utf-8")

    mock_generate = Mock()

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 2

    captured = capsys.readouterr()

    assert "Requirement file cannot be empty" in captured.err
    assert captured.out == ""

    mock_generate.assert_not_called()


# --------------------------------------------------
# Test 5: API failure -> Exit code 4
# --------------------------------------------------

def test_cli_returns_four_for_api_error(tmp_path, monkeypatch, capsys):

    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text(
        "User should be able to log in.",
        encoding="utf-8"
    )

    mock_generate = Mock(
        side_effect=cli.generator.GeneratorAPIError(
            "LLM API unavailable."
        )
    )

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 4

    captured = capsys.readouterr()

    assert "API error:" in captured.err
    assert "LLM API unavailable." in captured.err
    assert captured.out == ""

    mock_generate.assert_called_once()


# --------------------------------------------------
# Test 6: Invalid model output -> Exit code 5
# --------------------------------------------------

def test_cli_returns_five_for_output_error(
    tmp_path, monkeypatch, capsys
):

    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text(
        "User should be able to log in.",
        encoding="utf-8"
    )

    mock_generate = Mock(
        side_effect=cli.generator.GeneratorOutputError(
            "LLM returned unusable output."
        )
    )

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 5

    captured = capsys.readouterr()

    assert "Output error:" in captured.err
    assert "LLM returned unusable output." in captured.err
    assert captured.out == ""

    mock_generate.assert_called_once()


# --------------------------------------------------
# Test 7: Configuration failure -> Exit code 6
# --------------------------------------------------

def test_cli_returns_six_for_config_error(
    tmp_path, monkeypatch, capsys
):

    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text(
        "User should be able to log in.",
        encoding="utf-8"
    )

    mock_generate = Mock(
        side_effect=cli.generator.GeneratorConfigError(
            "OPENAI_API_KEY is not configured."
        )
    )

    monkeypatch.setattr(
        cli.generator,
        "generate",
        mock_generate
    )

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    assert exit_code == 6

    captured = capsys.readouterr()

    assert "Configuration error:" in captured.err
    assert "OPENAI_API_KEY is not configured." in captured.err
    assert captured.out == ""

    mock_generate.assert_called_once()

def test_cli_returns_two_for_directory_path(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "requirements"
    directory.mkdir()

    set_cli_arguments(monkeypatch, directory)

    exit_code = cli.main()

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Path is not a file" in captured.err
    assert captured.out == ""

def test_cli_returns_two_for_unreadable_file(
    tmp_path, monkeypatch, capsys
):
    requirement_file = tmp_path / "login.txt"
    requirement_file.write_text("Login requirement", encoding="utf-8")

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    set_cli_arguments(monkeypatch, requirement_file)

    exit_code = cli.main()

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Unable to read requirement file" in captured.err
    assert captured.out == ""
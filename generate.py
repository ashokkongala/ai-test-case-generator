import argparse
import sys
from pathlib import Path
import generator
from generator import RequirementInputError


def read_requirement(file_path: str) -> str:
    file = Path(file_path)

    if not file.exists():
        raise RequirementInputError(
            f"Requirement file not found: {file_path}"
        )

    if not file.is_file():
        raise RequirementInputError(
            f"Path is not a file: {file_path}"
        )

    try:
        requirement = file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RequirementInputError(
            f"Unable to read requirement file: {file_path}"
        ) from exc

    if not requirement.strip():
        raise RequirementInputError(
            "Requirement file cannot be empty."
        )

    return requirement


def run_generator(requirement_file: str) -> int:
    requirement = read_requirement(requirement_file)

    result = generator.generate(requirement)

    print(result.model_dump_json(indent=2))

    if result.status == "BLOCKED":
        return 3

    return 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="AI-powered test case generator")

    parser.add_argument("requirement_file", help="Path to the requirement text file.")

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        return run_generator(args.requirement_file)

    except RequirementInputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    except generator.GeneratorAPIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 4

    except generator.GeneratorOutputError as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return 5

    except generator.GeneratorConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())

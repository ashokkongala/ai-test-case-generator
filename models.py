from pydantic import BaseModel, model_validator
from typing import Literal


class TestCase(BaseModel):
    id: str
    title: str
    priority: Literal["P0", "P1", "P2", "P3"]
    steps: list[str]
    expected_result: str


class GeneratorResult(BaseModel):
    status: Literal["READY", "BLOCKED"]
    missing_information: list[str]
    test_cases: list[TestCase]

    @model_validator(mode="after")
    def validate_result(self):

        if self.status == "READY":
            if not self.test_cases:
                raise ValueError("READY requires at least one test case.")

            if self.missing_information:
                raise ValueError("READY cannot contain missing information.")

        elif self.status == "BLOCKED":
            if self.test_cases:
                raise ValueError("BLOCKED cannot contain test cases.")

            if not self.missing_information:
                raise ValueError(
                    "BLOCKED requires at least one clarification question."
                )

        return self


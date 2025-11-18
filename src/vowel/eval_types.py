import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IsInstanceCase(BaseModel):
    """Type checking evaluation case. Validates that output matches the specified Python type."""

    type: str = Field(
        description="Python type as string to check against. Can use union types with '|'.",
        examples=["int", "str", "bool", "list", "dict", "int | float", "str | None"],
    )

    strict: Optional[bool] = Field(
        default=None,
        description="Whether to use strict mode for type validation. When True, performs stricter type checking.",
    )

    def evaluate(self, output: Any) -> bool:
        return isinstance(output, eval(self.type))


class AssertionCase(BaseModel):
    """Custom assertion evaluation case. Runs Python expression with 'input' and 'output' variables."""

    assertion: str = Field(
        description=(
            "Python expression that returns boolean. Must evaluate to True for test to pass.\n\n"
            "Available variables in assertion context:\n"
            "  - input: The input value(s) passed to the function\n"
            "  - output: The actual output returned by the function\n"
            "  - expected: The expected output value from test case (if provided)\n"
            "  - duration: Actual execution time in seconds (float)\n"
            "  - metadata: Additional metadata dict (if provided)\n\n"
            "Common patterns:\n"
            "  - Compare output: output > 0, output == input * 2\n"
            "  - Type checks: isinstance(output, int), type(output).__name__ == 'str'\n"
            "  - String operations: output.isupper(), len(output) > 0\n"
            "  - Collection operations: all(x > 0 for x in output), len(output) == len(input)\n"
            "  - Dict access: output['key'] == expected, input['x'] + input['y'] == output\n"
            "  - Containment: output in input, 'substring' in output\n"
            "  - Performance: duration < 0.1\n"
            "  - Logic: (output and input > 0) or (not output and input <= 0)"
        ),
        examples=[
            "output > 0",
            "output == input * 2",
            "output == input ** 2",
            "len(output) > 0",
            "output.isupper()",
            "output.islower()",
            "all(x > 0 for x in output)",
            "len(output) <= len(input)",
            "output in input",
            "input in str(output)",
            "input['x'] + input['y'] == output",
            "output == expected",
            "abs(output - expected) < 0.001",
            "duration < 1.0",
            "isinstance(output, (int, float))",
            "output % 2 == 0",
            "(output and input % 2 == 0) or (not output and input % 2 != 0)",
        ],
    )

    def evaluate(self, input: Any, output: Any) -> bool:
        env = {"input": input, "output": output}
        return eval(self.assertion, env, env)


class DurationCase(BaseModel):
    """Performance evaluation case. Validates execution time is within specified duration."""

    duration: float = Field(
        description="Maximum allowed duration in seconds. Test fails if execution takes longer.",
        examples=[0.1, 1.0, 5.0, 0.001],
        gt=0,
    )

    def evaluate(self, actual_duration: float) -> bool:
        return actual_duration <= self.duration


class ContainsInputCase(BaseModel):
    """Input containment evaluation case. Validates that output contains the input value."""

    contains_input: bool = Field(
        default=True, description="Whether output should contain the input value."
    )
    case_sensitive: bool = Field(
        default=True, description="Whether string comparison should be case-sensitive."
    )
    as_strings: bool = Field(
        default=False,
        description="Whether to convert both input and output to strings before comparison.",
    )


class PatternMatchCase(BaseModel):
    """Regex pattern matching evaluation case. Validates that output matches a regex pattern."""

    pattern: str = Field(
        description="Regular expression pattern to match against the output (converted to string).",
        examples=[r"^\d+$", r"^[A-Z]+$", r".*@.*\.com$", r"id: \d+"],
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the regex matching should be case-sensitive.",
    )


class RaisesCase(BaseModel):
    """Exception raising evaluation case. Validates that the function raises a specific exception."""

    raises: str = Field(
        description="Expected exception type as string (e.g., 'ValueError', 'KeyError', 'TypeError').",
        examples=["ValueError", "TypeError", "KeyError", "ZeroDivisionError", "IndexError"],
    )
    match: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to match against the exception message.",
        examples=["invalid input", "must be positive", "not found"],
    )


class LLMJudgeCase(BaseModel):
    """LLM Judge evaluation case. Uses an LLM to evaluate the output based on a rubric."""

    rubric: str = Field(
        description="The rubric/criteria that the LLM should use to evaluate the output.",
        examples=[
            "Does the output equivalent to the expected output?",
            "Is the output grammatically correct?",
            "Does the response answer the question accurately?",
        ],
    )
    include: List[str] = Field(
        default_factory=list,
        description="List of context variables to include in the evaluation. Valid options: 'input', 'expected_output'.",
        examples=[["input"], ["expected_output"], ["input", "expected_output"]],
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the LLM model. 'model' is required unless JUDGE_MODEL env var is set. Other parameters are optional.",
    )

    @field_validator("include")
    @classmethod
    def validate_include(cls, v: Optional[List[str]]) -> List[str]:
        """Validate that include only contains valid options."""
        if v is None:
            return []

        valid_options = {"input", "expected_output"}
        invalid_options = set(v) - valid_options

        if invalid_options:
            raise ValueError(
                f"Invalid options in 'include': {invalid_options}. "
                f"Valid options are: {valid_options}"
            )

        return v

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that model is specified either in config or JUDGE_MODEL env var."""
        if v is None:
            v = {}

        if "model" not in v and not os.getenv("JUDGE_MODEL"):
            raise ValueError(
                "'model' must be specified in config or set JUDGE_MODEL environment variable"
            )

        config = {k: v for k, v in v.items() if v is not None}

        return config


class MatchCase(BaseModel):
    """Test case with input, expected output, and optional constraints."""

    id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for this test case.",
        examples=["test_positive_numbers", "edge_case_empty_list", "error_invalid_input"],
    )
    input: Optional[Any] = Field(
        default=None,
        description=(
            "Single input value to pass to the function as the only argument. "
            "Use this when the function takes a single argument. "
            "Cannot be used together with 'inputs'."
        ),
        examples=[5, "hello", [1, 2, 3], {"x": 10, "y": 20}, {"name": "test", "value": 42}],
    )
    inputs: Optional[list[Any]] = Field(
        default=None,
        description=(
            "Multiple input values to pass to the function as separate arguments (*args). "
            "Use this when the function takes multiple arguments. "
            "Cannot be used together with 'input'."
        ),
        examples=[[1, 2], [10, 20, 30], ["hello", "world"], [{"x": 1}, {"y": 2}]],
    )
    expected: Optional[Any] = Field(
        default=None,
        description="Expected output value. If provided, output will be compared for equality.",
        examples=[25, "HELLO", [1, 3, 5], True, {"result": 30}],
    )
    duration: Optional[float] = Field(
        default=None,
        description="Maximum allowed execution time in milliseconds for this specific case.",
        examples=[100, 500, 1000, 50],
        gt=0,
    )
    contains: Optional[Any] = Field(
        default=None,
        description="Value that should be contained in the output.",
        examples=["substring", 42, "expected_key"],
    )
    assertion: Optional[str] = Field(
        default=None,
        description=(
            "Optional case-specific Python assertion expression. Same as global assertions but only for this case.\n"
            "Available variables: input, output, expected, duration, metadata.\n"
            "Examples: 'output > 0', 'len(output) == 3', 'output == input * 2'"
        ),
        examples=["output > 0", "len(output) == 3", "output % 2 == 0", "output in input"],
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to match against the output (converted to string) for this specific case.",
        examples=[r"^\d+$", r"^[A-Z]+$", r".*@.*\.com$"],
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the regex pattern matching should be case-sensitive (only used if pattern is specified).",
    )
    raises: Optional[str] = Field(
        default=None,
        description="Expected exception type for this case. If specified, the test expects the function to raise this exception.",
        examples=["ValueError", "TypeError", "KeyError", "ZeroDivisionError"],
    )
    match: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to match against the exception message (only used if raises is specified).",
        examples=["invalid input", "must be positive", "not found"],
    )
    
    @field_validator("match")
    @classmethod
    def validate_match_requires_raises(cls, v, info):
        if v is not None and info.data.get("raises") is None:
            raise ValueError("'match' can only be used together with 'raises'")
        return v

    @field_validator("inputs")
    @classmethod
    def validate_input_xor_inputs(cls, v, info):
        """Ensure only one of input or inputs is provided."""
        if v is not None and info.data.get("input") is not None:
            raise ValueError("Cannot specify both 'input' and 'inputs'. Use only one.")
        return v

    def model_post_init(self, __context):
        """Validate that at least one of input or inputs is provided."""
        if self.input is None and self.inputs is None:
            raise ValueError("Must specify either 'input' or 'inputs'")

    @property
    def has_expected(self) -> bool:
        return self.expected is not None

    @property
    def has_duration(self) -> bool:
        return self.duration is not None

    @property
    def has_contains(self) -> bool:
        return self.contains is not None

    @property
    def has_assertion(self) -> bool:
        return self.assertion is not None

    @property
    def has_pattern(self) -> bool:
        return self.pattern is not None

    @property
    def has_raises(self) -> bool:
        return self.raises is not None


class EvalCase(BaseModel):
    """Internal representation of an evaluation case with its data."""

    id: str = Field(
        description="Unique identifier for this evaluation case.",
        examples=["IsInteger", "IsPositive", "TypeCheck", "CorrectLogic"],
    )
    case_data: Union[
        IsInstanceCase,
        AssertionCase,
        DurationCase,
        ContainsInputCase,
        PatternMatchCase,
        LLMJudgeCase,
    ] = Field(
        description="The actual evaluation logic - can be type check, assertion, duration, contains check, pattern match, or LLM judge."
    )

    @property
    def has_assertion(self) -> bool:
        return isinstance(self.case_data, AssertionCase)

    @property
    def has_typecheck(self) -> bool:
        return isinstance(self.case_data, IsInstanceCase)

    @property
    def has_duration(self) -> bool:
        return isinstance(self.case_data, DurationCase)

    @property
    def has_contains_input(self) -> bool:
        return isinstance(self.case_data, ContainsInputCase)

    @property
    def has_pattern_match(self) -> bool:
        return isinstance(self.case_data, PatternMatchCase)

    @property
    def has_llm_judge(self) -> bool:
        return isinstance(self.case_data, LLMJudgeCase)


class DatasetCase(BaseModel):
    """Wrapper for a single test case in the dataset."""

    case: MatchCase = Field(
        description="The test case containing input, expected output, and constraints."
    )

    @property
    def id(self) -> Optional[str]:
        return self.case.id


class Evals(BaseModel):
    """
    Complete evaluation specification for a single function.

    This is the main model for defining tests. It includes:
    - Function identifier
    - Global evaluation rules (type checks, assertions, etc.)
    - Test dataset with input/output pairs
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(
        description="Function name to evaluate. Must match the actual function name.",
        examples=["is_prime", "calculate_sum", "process_data", "validate_email"],
    )

    evals: Dict[
        str,
        Union[
            IsInstanceCase,
            AssertionCase,
            DurationCase,
            ContainsInputCase,
            PatternMatchCase,
            LLMJudgeCase,
        ],
    ] = Field(
        default_factory=dict,
        description=(
            "Dictionary of evaluation rules that apply to ALL test cases. "
            "Each key is a descriptive name, value is the evaluation case. "
            "Use IsInstanceCase for type checks, AssertionCase for custom logic, "
            "DurationCase for performance constraints, ContainsInputCase for input containment, "
            "PatternMatchCase for regex pattern matching."
        ),
        examples=[
            {"IsInteger": {"type": "int"}, "IsPositive": {"assertion": "output > 0"}},
            {
                "TypeCheck": {"type": "str"},
                "NotEmpty": {"assertion": "len(output) > 0"},
                "IsUppercase": {"assertion": "output.isupper()"},
            },
            {
                "IsBoolean": {"type": "bool"},
                "CorrectLogic": {
                    "assertion": "(output and input > 0) or (not output and input <= 0)"
                },
            },
        ],
    )

    dataset: list[DatasetCase] = Field(
        description=(
            "List of test cases. Each case has input, expected output, and optional constraints. "
            "Should cover normal cases, edge cases, and corner cases."
        ),
        examples=[
            [
                {"case": {"input": 2, "expected": 4}},
                {"case": {"input": 0, "expected": 0}},
                {"case": {"input": -3, "expected": 9}},
            ],
            [
                {"case": {"input": "hello", "expected": "HELLO"}},
                {"case": {"input": "world", "expected": "WORLD"}},
            ],
            [
                {"case": {"input": {"x": 2, "y": 3}, "expected": 5}},
                {"case": {"input": {"x": 10, "y": 20}, "expected": 30}},
            ],
        ],
        min_length=1,
    )

    @property
    def eval_cases(self) -> list[EvalCase]:
        return [
            EvalCase(id=eval_id, case_data=case_data) for eval_id, case_data in self.evals.items()
        ]


class EvalsFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def model_validate(cls, obj, **kwargs):
        instance = cls.model_construct(**obj)
        return instance

    def get_evals(self) -> Dict[str, Evals]:
        result = {}
        for key in dir(self):
            if key.startswith("_"):
                continue
            try:
                value = getattr(self, key)
                if isinstance(value, dict) and "dataset" in value:
                    result[key] = Evals(id=key, **value)
            except:
                continue

        if hasattr(self, "__pydantic_extra__"):
            for key, value in self.__pydantic_extra__.items():
                if isinstance(value, dict) and "dataset" in value:
                    result[key] = Evals(id=key, **value)

        return result

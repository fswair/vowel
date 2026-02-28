"""Evaluator implementations for the vowel framework.

This module contains the concrete evaluator classes that implement
the evaluation logic defined in eval_types.py. Each evaluator
integrates with pydantic-evals to provide result reporting.

Evaluators:
    AssertionEvaluator: Runs Python assertion expressions
    TypeAdapterEvaluator: Validates output types using Pydantic
    ContainsInputEvaluator: Checks if output contains input value
    PatternMatchingEvaluator: Validates output against regex patterns
    RaisesEvaluator: Validates expected exception raising

Factory functions:
    create_llm_judge: Creates an LLM-based judge evaluator
    prepare_env_and_condition: Prepares evaluation context
"""

import importlib.util
import logging
import os
import re
import typing
from contextlib import suppress
from dataclasses import dataclass

from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

logger = logging.getLogger(__name__)

MONTY_AVAILABLE = bool(importlib.util.find_spec("pydantic-monty"))


def prepare_env_and_condition(ctx: EvaluatorContext, condition: str) -> tuple[dict, str]:
    """
    Prepare environment variables and format condition string for evaluation.

    Args:
        ctx: EvaluatorContext containing inputs, output, expected, etc.
        condition: The condition string to format

    Returns:
        Tuple of (environment dict, formatted condition string)
    """
    actual_input = ctx.inputs
    if isinstance(ctx.inputs, dict):
        if "input" in ctx.inputs:
            actual_input = ctx.inputs["input"]
        elif "inputs" in ctx.inputs:
            actual_input = ctx.inputs["inputs"]

    env = {
        "input": actual_input,
        "output": ctx.output,
        "expected": ctx.expected_output,
        "metrics": ctx.metrics,
        "metadata": ctx.metadata,
        "duration": ctx.duration,
    }

    formatted_condition = condition
    for key, val in env.items():
        formatted_val = f"`{val}`" if isinstance(val, str) else repr(val)
        formatted_condition = re.sub(rf"\b{re.escape(key)}\b", formatted_val, formatted_condition)

    return env, formatted_condition


@dataclass
class AssertionEvaluator(Evaluator):
    """Evaluator that runs a Python assertion expression.

    The condition is evaluated with access to input, output, expected,
    metrics, metadata, and duration variables.
    """

    def __init__(self, condition: str, *, evaluation_name: str = "Assertion"):
        self.condition = condition
        self.evaluation_name = evaluation_name
        self.interpreter = None
        if MONTY_AVAILABLE:
            import pydantic_monty

            self.interpreter = pydantic_monty.Monty(  # pyright: ignore[reportOptionalMemberAccess]
                condition,
                script_name="assertion.py",
                inputs=["input", "output", "expected", "metrics", "metadata", "duration"],
            )

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        """Evaluate the assertion condition against the context."""
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        if "__import__" in self.condition:
            raise ValueError(f"__import__ is not allowed in assertions: {self.condition}")
        env, condition = prepare_env_and_condition(ctx, self.condition)

        # TL;DR
        # BETA API

        return self.eval_python(condition, env)

        # CURRENT API
        # if eval(self.condition, env, env):
        #     return EvaluationReason(
        #         value=True, reason=f"Assertion passed for condition: {condition}"
        #     )
        # else:
        #     return EvaluationReason(
        #         value=False, reason=f"Assertion failed for condition: {condition}"
        #     )

    def eval_python(self, condition: str, inputs: dict) -> EvaluationReason:
        try:
            if self.interpreter:
                if self.interpreter.run(inputs=inputs):
                    return EvaluationReason(
                        value=True, reason=f"Assertion passed for condition: {condition}"
                    )
            else:
                raise ImportError(
                    "Monty runtime is unavailable, install via `pip install vowel[monty]`."
                )

        except Exception:
            with suppress(Exception):
                if eval(self.condition, inputs, inputs):
                    return EvaluationReason(
                        value=True, reason=f"Assertion passed for condition: {condition}"
                    )

        return EvaluationReason(value=False, reason=f"Assertion failed for condition: {condition}")


@dataclass
class TypeAdapterEvaluator(Evaluator):
    """Evaluator that validates output type using Pydantic TypeAdapter.

    Supports union types (e.g., 'int | float') and optional strict mode.
    """

    type: str
    evaluation_name: str = "Exact Type"
    strict: bool | None = None

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        """Validate that output matches the expected type."""
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        type_env = {
            "typing": typing,
            "__import__": None,
            "eval": None,
            "exec": None,
            "compile": None,
        }
        try:
            expected_type = eval(self.type, type_env, type_env)
            ta = TypeAdapter(expected_type)
        except Exception:
            return EvaluationReason(
                value=False,
                reason=f"Invalid type expression: Failed to determine type {self.type!r}.",
            )
        try:
            ta.validate_python(ctx.output, strict=self.strict)
            return EvaluationReason(value=True, reason=f"Output is of type {self.type!r}")
        except (ValidationError, TypeError, ValueError) as e:
            return EvaluationReason(value=False, reason=f"Output is not of type {self.type!r}: {e}")


@dataclass
class ContainsInputEvaluator(Evaluator):
    """Evaluator that checks if output contains the input value.

    Supports case-sensitive/insensitive comparison and string conversion.
    """

    evaluation_name: str = "ContainsInput"
    case_sensitive: bool = True
    as_strings: bool = False

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        """Check if output contains the input value."""
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        input_value = ctx.inputs
        if isinstance(ctx.inputs, dict):
            if "input" in ctx.inputs:
                input_value = ctx.inputs["input"]
            elif "inputs" in ctx.inputs:
                input_value = ctx.inputs["inputs"]

        output_value = ctx.output

        try:
            should_use_strings = self.as_strings or (
                isinstance(input_value, str) and isinstance(output_value, str)
            )

            if should_use_strings:
                input_str = str(input_value)
                output_str = str(output_value)

                if not self.case_sensitive:
                    input_str = input_str.lower()
                    output_str = output_str.lower()

                if input_str in output_str:
                    return EvaluationReason(
                        value=True,
                        reason=f"Output string contains input value {input_value!r}",
                    )
                else:
                    return EvaluationReason(
                        value=False,
                        reason=f"Output string does not contain input value {input_value!r}",
                    )

            if input_value in output_value:
                return EvaluationReason(
                    value=True, reason=f"Output contains input value {input_value!r}"
                )
            else:
                return EvaluationReason(
                    value=False,
                    reason=f"Output {output_value!r} does not contain input value {input_value!r}",
                )
        except (TypeError, ValueError) as e:
            return EvaluationReason(value=False, reason=f"Containment check failed: {e}")


@dataclass
class PatternMatchingEvaluator(Evaluator):
    """Evaluator that validates output matches a regex pattern.

    Converts output to string before matching.
    """

    pattern: str
    evaluation_name: str = "RegEx Match"
    case_sensitive: bool = True

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        """Check if output matches the regex pattern."""
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        flags = 0 if self.case_sensitive else re.IGNORECASE
        output_str = str(ctx.output)

        if re.search(self.pattern, output_str, flags):
            return EvaluationReason(
                value=True,
                reason=f"Output matches the regex pattern {self.pattern!r}",
            )
        else:
            return EvaluationReason(
                value=False,
                reason=f"Output does not match the regex pattern {self.pattern!r}",
            )


@dataclass
class RaisesEvaluator(Evaluator):
    """Evaluator for checking if function raised expected exception.

    When optional=True (from 'raises: TypeError?' syntax), the evaluator
    passes if the function returns normally OR raises the expected exception.
    It only fails if a DIFFERENT exception type is raised.

    Special values:
        'any'  — function must raise any exception (type doesn't matter)
        'any?' — function may raise any exception or return normally
    """

    expected_exception_type: str
    expected_exception_match: str | None = None
    optional: bool = False
    evaluation_name: str = "Raises"

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        is_exception = isinstance(output, dict) and "_exception" in output

        # raises: any / raises: any?
        if self.expected_exception_type == "any":
            if not is_exception:
                if self.optional:
                    return EvaluationReason(
                        value=True,
                        reason="Function returned normally (raises: any? — OK)",
                    )
                return EvaluationReason(
                    value=False,
                    reason=f"Expected any exception to be raised, but function returned normally with output: {output!r}",
                )
            actual_type = output["_exception_type"]
            return EvaluationReason(value=True, reason=f"Correctly raised {actual_type}")

        if not is_exception:
            if self.optional:
                return EvaluationReason(
                    value=True,
                    reason=(
                        f"Function returned normally (optional raises: "
                        f"{self.expected_exception_type}? — OK)"
                    ),
                )
            return EvaluationReason(
                value=False,
                reason=(
                    f"Expected {self.expected_exception_type} to be raised, "
                    f"but function returned normally with output: {output!r}"
                ),
            )
        actual_exception = output["_exception"]
        actual_type = output["_exception_type"]
        expected_type_short = self.expected_exception_type.split(".")[-1]
        if actual_type != expected_type_short:
            return EvaluationReason(
                value=False,
                reason=(
                    f"Expected {expected_type_short}, but got {actual_type}: {actual_exception}"
                ),
            )
        if self.expected_exception_match:
            exception_message = str(actual_exception)
            if not re.search(self.expected_exception_match, exception_message, re.I):
                return EvaluationReason(
                    value=False,
                    reason=(
                        f"Exception type matches ({actual_type}), but message doesn't "
                        f"match pattern {self.expected_exception_match!r}. "
                        f"Message: {exception_message}"
                    ),
                )
        reason = f"Correctly raised {actual_type}"
        if self.expected_exception_match:
            reason += f" with message matching pattern {self.expected_exception_match!r}"
        return EvaluationReason(value=True, reason=reason)


def create_llm_judge(
    rubric: str,
    include: list[str] | None = None,
    config: dict | None = None,
) -> LLMJudge:
    if config is None:
        config = {}

    model = config.pop("model", None) or os.getenv("JUDGE_MODEL")
    if not model:
        raise ValueError(
            "'model' must be specified in config or set JUDGE_MODEL environment variable"
        )

    if model.strip().startswith("$"):
        env_var = model.strip().lstrip("$")
        model = os.getenv(env_var)
        if not model:
            raise ValueError(
                f"Environment variable {env_var} is not set for judge model, "
                f"set {env_var} to a valid model name."
            )

    include_input = False
    include_expected_output = False

    if include:
        include_input = "input" in include
        include_expected_output = "expected_output" in include

    model_settings_kwargs = {}
    for key, value in config.items():
        if value is not None:
            model_settings_kwargs[key] = value

    model_settings = ModelSettings(**model_settings_kwargs) if model_settings_kwargs else None

    return LLMJudge(
        model=model,
        rubric=rubric,
        include_input=include_input,
        include_expected_output=include_expected_output,
        model_settings=model_settings,
    )

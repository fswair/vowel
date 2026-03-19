"""Concrete evaluator implementations used by the vowel runtime."""

import importlib.util
import os
import re
import typing
from contextlib import suppress
from dataclasses import dataclass

import logfire
from pydantic import ValidationError
from pydantic.type_adapter import TypeAdapter
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    LLMJudge,
    OutputConfig,  # noqa: F401
)

MONTY_AVAILABLE = bool(importlib.util.find_spec("pydantic-monty"))

SAFE_ASSERTION_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

SAFE_TYPE_NAMES = {
    "Any": typing.Any,
    "None": None,
    "bool": bool,
    "bytes": bytes,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "object": object,
    "set": set,
    "str": str,
    "tuple": tuple,
    "typing": typing,
}
SAFE_TYPE_NAMES.update(
    {name: getattr(typing, name) for name in dir(typing) if not name.startswith("_")}
)


def _eval_assertion_restricted(condition: str, inputs: dict[str, typing.Any]) -> bool:
    env = {**inputs, "__builtins__": SAFE_ASSERTION_BUILTINS}
    return bool(eval(condition, env, env))


def _eval_type_restricted(type_expr: str) -> typing.Any:
    env = {"__builtins__": {}}
    env.update(SAFE_TYPE_NAMES)
    return eval(type_expr, env, env)


def _apply_serializer_for_assertion(
    value: typing.Any,
    serializer: type | typing.Callable | dict[str, type | typing.Callable] | None,
    *,
    param_name: str | None = None,
) -> typing.Any:
    """Apply serializer in assertion path to mirror function call conversions."""
    if serializer is None:
        return value

    if isinstance(serializer, dict):
        if param_name and param_name in serializer:
            return _apply_serializer_for_assertion(value, serializer[param_name])
        if isinstance(value, dict):
            converted: dict[str, typing.Any] = {}
            for key, item in value.items():
                if key in serializer:
                    converted[key] = _apply_serializer_for_assertion(item, serializer[key])
                else:
                    converted[key] = item
            return converted
        return value

    if isinstance(value, dict):
        try:
            return serializer(**value)
        except TypeError:
            return serializer(value)

    return serializer(value)


def _normalize_input_for_assertion(
    raw_inputs: typing.Any,
    serializer: type | typing.Callable | dict[str, type | typing.Callable] | None,
    serializer_fn: typing.Callable[[dict], typing.Any] | None,
) -> typing.Any:
    """Compute assertion `input` value from raw case inputs using active serializer config."""
    if not isinstance(raw_inputs, dict):
        return _apply_serializer_for_assertion(raw_inputs, serializer)

    if serializer_fn is not None:
        serialized = serializer_fn(raw_inputs)
        if isinstance(serialized, tuple):
            return serialized[0] if len(serialized) == 1 else serialized
        return serialized

    if "input" in raw_inputs:
        return _apply_serializer_for_assertion(raw_inputs["input"], serializer)

    if "inputs" in raw_inputs:
        values = raw_inputs["inputs"]
        if values is None:
            return None
        if isinstance(values, dict):
            if serializer is not None and not isinstance(serializer, dict):
                return _apply_serializer_for_assertion(values, serializer)
            if isinstance(serializer, dict):
                return {
                    key: _apply_serializer_for_assertion(item, serializer, param_name=key)
                    for key, item in values.items()
                }
            return values
        if serializer is None:
            return values
        return [_apply_serializer_for_assertion(item, serializer) for item in values]

    return raw_inputs


def prepare_env_and_condition(
    ctx: EvaluatorContext,
    condition: str,
    *,
    serializer: type | typing.Callable | dict[str, type | typing.Callable] | None = None,
    serializer_fn: typing.Callable[[dict], typing.Any] | None = None,
) -> tuple[dict, str]:
    """
    Prepare environment variables and format condition string for evaluation.

    Args:
        ctx: EvaluatorContext containing inputs, output, expected, etc.
        condition: The condition string to format

    Returns:
        Tuple of (environment dict, formatted condition string)
    """
    actual_input = _normalize_input_for_assertion(ctx.inputs, serializer, serializer_fn)

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

    def __init__(
        self,
        condition: str,
        *,
        evaluation_name: str = "Assertion",
        serializer: type | typing.Callable | dict[str, type | typing.Callable] | None = None,
        serializer_fn: typing.Callable[[dict], typing.Any] | None = None,
    ):
        self.condition = condition
        self.evaluation_name = evaluation_name
        self.serializer = serializer
        self.serializer_fn = serializer_fn
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
        env, condition = prepare_env_and_condition(
            ctx,
            self.condition,
            serializer=self.serializer,
            serializer_fn=self.serializer_fn,
        )

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
            pass

        try:
            if _eval_assertion_restricted(self.condition, inputs):
                return EvaluationReason(
                    value=True, reason=f"Assertion passed for condition: {condition}"
                )
        except Exception as exc:
            logfire.info(
                "Restricted assertion eval failed; falling back to raw eval",
                condition=self.condition,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            with suppress(Exception):
                fallback_env = {**inputs, "__builtins__": SAFE_ASSERTION_BUILTINS}
                if eval(self.condition, fallback_env, fallback_env):
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
        try:
            expected_type = _eval_type_restricted(self.type)
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
            if self.expected_exception_match != exception_message and not re.search(
                self.expected_exception_match, exception_message, re.I
            ):
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
    # Imported lazily to avoid circular import at module import time.
    from .utils import _resolve_env_ref

    if config is None:
        config = {}

    model = config.pop("model", None) or os.getenv("JUDGE_MODEL")
    if not model:
        raise ValueError(
            "'model' must be specified in config or set JUDGE_MODEL environment variable"
        )

    model = _resolve_env_ref(model, field_name="model")
    rubric = _resolve_env_ref(rubric, field_name="rubric")

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

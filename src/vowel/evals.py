import importlib.util
import os
import re
import typing
from dataclasses import dataclass

import dotenv
from pydantic.type_adapter import TypeAdapter
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

dotenv.load_dotenv()

if os.environ.get("LOGFIRE_ENABLED") in ("1", "true", "True"):
    if importlib.util.find_spec("logfire"):
        import logfire

        logfire.configure()
        logfire.instrument_pydantic_ai()
    else:
        raise ImportError(
            "LOGFIRE_ENABLED is set but logfire is not installed. Please install logfire or set LOGFIRE_ENABLED=false"
        )


def prepare_env_and_condition(ctx: EvaluatorContext, condition: str) -> tuple[dict, str]:
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

    for key in env:
        if isinstance(env[key], str):
            value = f"`{env[key]}`"
        else:
            value = repr(env[key])
        condition = condition.replace(key, value)
    return env, condition


@dataclass
class AssertionEvaluator(Evaluator):

    def __init__(self, condition: str, *, evaluation_name: str = "Assertion"):
        self.condition = condition
        self.evaluation_name = evaluation_name

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        env, condition = prepare_env_and_condition(ctx, self.condition)
        try:
            assert eval(self.condition, env, env)
            return EvaluationReason(
                value=True, reason=f"Assertion passed for condition: {condition}"
            )
        except AssertionError:
            return EvaluationReason(
                value=False, reason=f"Assertion failed for condition: {condition}"
            )


@dataclass
class TypeAdapterEvaluator(Evaluator):

    type: str
    evaluation_name: str = "Exact Type"
    strict: bool | None = None

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        if isinstance(ctx.output, dict) and "_exception" in ctx.output:
            return EvaluationReason(value=True, reason="Skipped (exception case)")
        type_env = {"typing": typing}
        expected_type = eval(self.type, type_env, type_env)
        ta = TypeAdapter(expected_type)
        try:
            ta.validate_python(ctx.output, strict=self.strict)
            return EvaluationReason(value=True, reason=f"Output is of type {self.type!r}")
        except:
            return EvaluationReason(value=False, reason=f"Output is not of type {self.type!r}")


@dataclass
class ContainsInputEvaluator(Evaluator):

    evaluation_name: str = "ContainsInput"
    case_sensitive: bool = True
    as_strings: bool = False

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
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

    pattern: str
    evaluation_name: str = "RegEx Match"
    case_sensitive: bool = True

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
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
    """Evaluator for checking if function raised expected exception."""

    expected_exception_type: str
    expected_exception_match: str | None = None
    evaluation_name: str = "Raises"

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        if not isinstance(output, dict) or "_exception" not in output:
            return EvaluationReason(
                value=False,
                reason=f"Expected {self.expected_exception_type} to be raised, but function returned normally with output: {output!r}",
            )
        actual_exception = output["_exception"]
        actual_type = output["_exception_type"]
        if actual_type != self.expected_exception_type:
            return EvaluationReason(
                value=False,
                reason=f"Expected {self.expected_exception_type}, but got {actual_type}: {actual_exception}",
            )
        if self.expected_exception_match:
            exception_message = str(actual_exception)
            if not re.search(self.expected_exception_match, exception_message):
                return EvaluationReason(
                    value=False,
                    reason=f"Exception type matches ({actual_type}), but message doesn't match pattern {self.expected_exception_match!r}. Message: {exception_message}",
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
                f"Environment variable {env_var} is not set for judge model, set {env_var} to a valid model name."
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

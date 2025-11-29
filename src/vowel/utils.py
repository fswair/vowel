import asyncio
import builtins
import importlib
import importlib.util
import inspect
import os
import sys
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import click
import yaml
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected, Evaluator, MaxDuration
from pydantic_evals.reporting import EvaluationReport

from .eval_types import Evals, EvalsFile
from .evals import (
    AssertionEvaluator,
    ContainsInputEvaluator,
    PatternMatchingEvaluator,
    RaisesEvaluator,
    TypeAdapterEvaluator,
    create_llm_judge,
)

sys.path.insert(0, os.getcwd())


def unpack_inputs(func: Callable, has_raises: bool = False) -> Callable:
    """
    Create a wrapper that handles both single input and multiple inputs.
    Supports both sync and async functions.

    Expects a dict with either:
    - {'input': value} -> passes value as single argument: func(value)
    - {'inputs': [val1, val2, ...]} -> unpacks as multiple arguments: func(val1, val2, ...)

    Args:
        func: The function to wrap (sync or async)
        has_raises: If True, catches exceptions and returns them for validation

    Returns:
        Wrapped function that accepts dict with 'input' or 'inputs' key
    """
    is_async = inspect.iscoroutinefunction(func)

    def _call_func(f, arg_dict):
        """Helper to call function with proper argument unpacking."""
        if not isinstance(arg_dict, dict):
            return f(arg_dict)

        if "inputs" in arg_dict:
            inputs = arg_dict["inputs"]
            return f(**inputs) if isinstance(inputs, dict) else f(*inputs)
        elif "input" in arg_dict:
            return f(arg_dict["input"])

        return f(arg_dict)

    if is_async:

        @wraps(func)
        async def wrapper(arg_dict):
            try:
                return await _call_func(func, arg_dict)
            except Exception as e:
                if has_raises:
                    return {"_exception": e, "_exception_type": type(e).__name__}
                raise

        return wrapper
    else:

        @wraps(func)
        def wrapper(arg_dict):
            try:
                return _call_func(func, arg_dict)
            except Exception as e:
                if has_raises:
                    return {"_exception": e, "_exception_type": type(e).__name__}
                raise

        return wrapper


def import_function(module_path: str) -> Callable:
    """
    Import a function from a module path.
    Handles standard imports and file-based imports (when shadowing occurs).

    Args:
        module_path: Full module path like 'module.submodule.function', builtin name like 'len',
                    or builtin method like 'str.upper'

    Returns:
        The imported function

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function is not found in the module
    """
    if "." not in module_path:
        try:
            return getattr(builtins, module_path)
        except AttributeError:
            raise ImportError(
                f"Function '{module_path}' not found in builtins. "
                f"Use full module path like 'module.function' or pass the function directly."
            )

    parts = module_path.split(".")

    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        remaining_parts = parts[i:]

        module = None

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            relative_path = module_name.replace(".", os.sep) + ".py"
            file_path = os.path.join(os.getcwd(), relative_path)

            if os.path.exists(file_path):
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                except Exception:
                    pass

        if module:
            try:
                obj = module
                for part in remaining_parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue

    try:
        obj = getattr(builtins, parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        pass

    raise ImportError(
        f"Cannot import '{module_path}'. Tried all possible module/attribute combinations."
    )


def load_evals_file(yaml_path: str) -> Dict[str, Evals]:
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)

    evals_file = EvalsFile.model_validate(loaded)
    return evals_file.get_evals()


def load_evals_from_yaml_string(yaml_content: str) -> Dict[str, Evals]:
    loaded = yaml.safe_load(yaml_content)
    evals_file = EvalsFile.model_validate(loaded)
    return evals_file.get_evals()


def load_evals_from_dict(data: dict) -> Dict[str, Evals]:
    evals_file = EvalsFile.model_validate(data)
    return evals_file.get_evals()


def load_evals_from_object(evals_obj: EvalsFile) -> Dict[str, Evals]:
    return evals_obj.get_evals()


def load_evals(source: Union[str, Path, dict, EvalsFile]) -> Dict[str, Evals]:
    if isinstance(source, EvalsFile):
        return load_evals_from_object(source)
    elif isinstance(source, dict):
        return load_evals_from_dict(source)
    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if "\n" in source_str or source_str.strip().startswith("{") or ":" in source_str:
            return load_evals_from_yaml_string(source_str)
        else:
            return load_evals_file(source_str)
    else:
        raise TypeError(
            f"source must be a file path (str/Path), YAML string (str), dict, or EvalsFile object, got {type(source)}"
        )


def to_dataset(evals: Evals, *, name: str) -> Dataset:
    dataset_cases: List[Case] = []
    global_evaluators: List[Evaluator] = []

    for eval_case in evals.eval_cases:
        if eval_case.has_assertion:
            assertion = eval_case.case_data.assertion
            global_evaluators.append(AssertionEvaluator(assertion, evaluation_name=eval_case.id))
        elif eval_case.has_typecheck:
            type_str = eval_case.case_data.type
            strict = eval_case.case_data.strict
            global_evaluators.append(
                TypeAdapterEvaluator(type=type_str, evaluation_name=eval_case.id, strict=strict)
            )
        elif eval_case.has_duration:
            duration_seconds = eval_case.case_data.duration
            global_evaluators.append(MaxDuration(timedelta(seconds=duration_seconds)))
        elif eval_case.has_contains_input:
            case_sensitive = eval_case.case_data.case_sensitive
            as_strings = eval_case.case_data.as_strings
            global_evaluators.append(
                ContainsInputEvaluator(
                    evaluation_name=eval_case.id,
                    case_sensitive=case_sensitive,
                    as_strings=as_strings,
                )
            )
        elif eval_case.has_pattern_match:
            pattern = eval_case.case_data.pattern
            case_sensitive = eval_case.case_data.case_sensitive
            global_evaluators.append(
                PatternMatchingEvaluator(
                    pattern=pattern,
                    evaluation_name=eval_case.id,
                    case_sensitive=case_sensitive,
                )
            )
        elif eval_case.has_llm_judge:
            rubric = eval_case.case_data.rubric
            include = eval_case.case_data.include
            config = eval_case.case_data.config or {}
            judge = create_llm_judge(
                rubric=rubric,
                include=include,
                config=config,
            )
            judge.evaluation_name = eval_case.id
            global_evaluators.append(judge)

    for dataset_case in evals.dataset:
        match_case = dataset_case.case
        case_evaluators = []

        if match_case.has_expected:
            case_evaluators.append(EqualsExpected())

        if match_case.has_duration:
            case_evaluators.append(MaxDuration(timedelta(milliseconds=match_case.duration)))

        if match_case.has_contains:
            case_evaluators.append(Contains(match_case.contains))

        if match_case.has_assertion:
            case_evaluators.append(
                AssertionEvaluator(
                    match_case.assertion,
                    evaluation_name=f"CaseAssertion: {match_case.assertion[:50]}",
                )
            )

        if match_case.has_pattern:
            case_evaluators.append(
                PatternMatchingEvaluator(
                    pattern=match_case.pattern,
                    evaluation_name=f"CasePattern: {match_case.pattern[:50]}",
                    case_sensitive=match_case.case_sensitive,
                )
            )

        if match_case.has_raises:
            case_evaluators.append(
                RaisesEvaluator(
                    expected_exception_type=match_case.raises,
                    expected_exception_match=match_case.match,
                    evaluation_name=f"Raises: {match_case.raises}",
                )
            )

        if not case_evaluators:
            case_evaluators = []

        case_metadata = {}
        if match_case.has_raises:
            case_metadata["_expects_exception"] = True
            case_metadata["_exception_type"] = match_case.raises
            if match_case.match:
                case_metadata["_exception_match"] = match_case.match

        if match_case.inputs is not None:
            display_input = f"inputs: {match_case.inputs}"
            input_value = {"inputs": match_case.inputs}
        else:
            display_input = f"input: {match_case.input}"
            input_value = {"input": match_case.input}

        if any(case for case in dataset_cases if input_value == case.inputs):
            print("Already exists in dataset, skipping duplicate case.")
            continue

        dataset_cases.append(
            Case(
                name=dataset_case.id or display_input,
                inputs=input_value,
                expected_output=match_case.expected,
                evaluators=case_evaluators,
                metadata=case_metadata if case_metadata else None,
            )
        )

    return Dataset(name=name, cases=dataset_cases, evaluators=global_evaluators)


class EvalResult(BaseModel):
    eval_id: str
    report: Optional[EvaluationReport] = None
    error: Optional[str] = None
    success: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        eval_id: str,
        report: Optional[EvaluationReport] = None,
        error: Optional[Exception] = None,
        **data,
    ):
        error_str = str(error) if error else None
        success = error is None and not self._check_failures(report)
        super().__init__(eval_id=eval_id, report=report, error=error_str, success=success, **data)

    @staticmethod
    def _check_failures(report: Optional[EvaluationReport]) -> bool:
        if report is None:
            return False
        for case in report.cases:
            if hasattr(case, "error") and case.error is not None:
                return True
            for assertion in case.assertions.values():
                if not assertion.value:
                    return True
                if hasattr(assertion, "error") and assertion.error is not None:
                    return True
        return False

    def has_failures(self) -> bool:
        return self._check_failures(self.report)


class EvalSummary(BaseModel):
    results: List[EvalResult]
    success_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    error_count: int = Field(default=0)
    total_count: int = Field(default=0)

    func: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, results: List[EvalResult], **data):
        success_count = sum(1 for r in results if r.success and r.report)
        failed_count = sum(1 for r in results if r.report and r.has_failures())
        error_count = sum(1 for r in results if r.error is not None)
        total_count = len(results)
        super().__init__(
            results=results,
            success_count=success_count,
            failed_count=failed_count,
            error_count=error_count,
            total_count=total_count,
            **data,
        )

    @property
    def all_passed(self) -> bool:
        return self.success_count == self.total_count

    @property
    def failed_results(self) -> List[EvalResult]:
        return [r for r in self.results if r.report and r.has_failures()]

    def print(self, *, include_reports: bool = True, include_reasons: bool = False):
        """
        Print a formatted summary of evaluation results using rich tables.

        Args:
            include_reports: Whether to print individual evaluation reports
            include_reasons: Whether to include failure reasons in reports
        """
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
        except ImportError:

            self._print_simple(include_reports=include_reports)
            return

        console = Console()

        summary_table = Table(title="📊 Evaluation Summary", box=box.ROUNDED, show_header=True)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Total Functions", str(self.total_count))
        summary_table.add_row("✅ Passed", f"[green]{self.success_count}[/green]")
        summary_table.add_row(
            "❌ Failed", f"[red]{self.failed_count}[/red]" if self.failed_count > 0 else "0"
        )
        summary_table.add_row(
            "⚠️  Errors", f"[yellow]{self.error_count}[/yellow]" if self.error_count > 0 else "0"
        )
        summary_table.add_row(
            "Status",
            (
                "[green]✅ ALL PASSED[/green]"
                if self.all_passed
                else "[red]❌ FAILURES DETECTED[/red]"
            ),
        )

        console.print(summary_table)

        if self.results:
            console.print()
            results_table = Table(title="🔍 Function Results", box=box.SIMPLE, show_header=True)
            results_table.add_column("Function", style="cyan", no_wrap=True)
            results_table.add_column("Status", justify="center")
            results_table.add_column("Cases", justify="right")
            results_table.add_column("Success Rate", justify="right")
            results_table.add_column("Details", style="dim")

            for result in self.results:
                if result.error:
                    status = "⚠️"
                    cases = "N/A"
                    success_rate = "N/A"
                    details = f"Error: {str(result.error)[:50]}"
                elif result.report:
                    total_cases = len(result.report.cases)
                    passed_cases = sum(
                        1
                        for case in result.report.cases
                        if all(assertion.value for assertion in case.assertions.values())
                    )

                    if result.success:
                        status = "✅"
                        success_rate = "[green]100%[/green]"
                    else:
                        status = "❌"
                        success_rate = f"[yellow]{(passed_cases/total_cases*100):.1f}%[/yellow]"

                    cases = f"{passed_cases}/{total_cases}"

                    failed_assertions = sum(
                        sum(1 for assertion in case.assertions.values() if not assertion.value)
                        for case in result.report.cases
                    )
                    details = (
                        f"{failed_assertions} failed assertions"
                        if failed_assertions > 0
                        else "All passed"
                    )
                else:
                    status = "❓"
                    cases = "0"
                    success_rate = "N/A"
                    details = "No report"

                results_table.add_row(result.eval_id, status, cases, success_rate, details)

            console.print(results_table)

        if include_reports:
            for result in self.results:
                if result.report:
                    console.print()
                    console.print(Panel(f"[bold cyan]{result.eval_id}[/bold cyan]", expand=False))
                    result.report.print(include_reasons=True)

    def _print_simple(self, *, include_reports: bool = True):
        """Fallback simple print without rich library."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Functions: {self.total_count}")
        print(f"✅ Passed: {self.success_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"⚠️  Errors: {self.error_count}")
        print(f"Status: {'✅ ALL PASSED' if self.all_passed else '❌ FAILURES DETECTED'}")
        print("=" * 60)

        if self.results:
            print("\nFUNCTION RESULTS:")
            for result in self.results:
                status = "✅" if result.success else "❌" if result.report else "⚠️"
                print(f"  {status} {result.eval_id}")
                if result.error:
                    print(f"     Error: {result.error}")

        if include_reports:
            for result in self.results:
                if result.report:
                    print(f"\n{'='*60}")
                    print(f"{result.eval_id}")
                    print("=" * 60)
                    result.report.print()

    def json(self) -> dict:
        """
        Convert evaluation summary to detailed JSON format.
        Useful for LLM feedback - shows exactly which cases and evaluations failed.

        Returns:
            dict
        """
        result_dict = {
            "summary": {
                "total_functions": self.total_count,
                "passed": self.success_count,
                "failed": self.failed_count,
                "errors": self.error_count,
                "all_passed": self.all_passed,
            },
            "results": [],
        }

        for result in self.results:
            result_data = {
                "function": result.eval_id,
                "status": "error" if result.error else ("passed" if result.success else "failed"),
            }

            if result.error:
                result_data["error"] = result.error
                result_data["error_type"] = "Error"

            if result.report:
                result_data["cases"] = []

                for case in result.report.cases:

                    case_passed = all(assertion.value for assertion in case.assertions.values())

                    case_output = case.output
                    if isinstance(case_output, dict) and "_exception" in case_output:
                        case_output = {
                            "_exception_type": case_output.get("_exception_type"),
                            "_exception_message": str(case_output.get("_exception")),
                        }

                    case_data = {
                        "case_id": case.name,
                        "status": "passed" if case_passed else "failed",
                        "input": case.inputs,
                        "output": case_output,
                        "expected_output": case.expected_output,
                        "duration_ms": (
                            round(case.total_duration * 1000, 2)
                            if hasattr(case, "total_duration") and case.total_duration
                            else None
                        ),
                        "evaluations": [],
                    }

                    for eval_name, assertion in case.assertions.items():
                        eval_data = {
                            "name": eval_name,
                            "passed": assertion.value,
                        }

                        if hasattr(assertion, "reason") and assertion.reason:
                            eval_data["reason"] = assertion.reason

                        case_data["evaluations"].append(eval_data)

                    result_data["cases"].append(case_data)

            result_dict["results"].append(result_data)

        return result_dict


def run_evals(
    source: Union[str, Path, dict, EvalsFile],
    *,
    filter_funcs: Optional[List[str]] = None,
    functions: Optional[Dict[str, Callable]] = None,
    debug: bool = False,
) -> EvalSummary:
    """
    Run evaluations from various sources.

    Args:
        source: Evaluation source (file path, YAML string, dict, or EvalsFile)
        filter_funcs: Optional list of function names to evaluate
        functions: Optional dict of {name: function} to use instead of importing
        debug: Enable debug mode

    Returns:
        EvalSummary with aggregated results
    """
    all_evals = load_evals(source)

    if filter_funcs:
        filtered_evals = {k: v for k, v in all_evals.items() if k in filter_funcs}
        if not filtered_evals:
            available = list(all_evals.keys())
            raise ValueError(
                f"No functions found matching filters: {', '.join(filter_funcs)}. "
                f"Available: {', '.join(available)}"
            )
        all_evals = filtered_evals

    results = []

    for eval_id, evals in all_evals.items():
        try:

            if functions and eval_id in functions:
                func = functions[eval_id]
            else:
                func = import_function(eval_id)
            click.echo(f"  📦 Imported: {click.style(eval_id, fg='blue', dim=True)}")
        except (ImportError, AttributeError) as e:
            results.append(EvalResult(eval_id, error=e))
            continue

        try:
            dataset = to_dataset(name=eval_id, evals=evals)

            has_any_raises = any(
                case.metadata and case.metadata.get("_expects_exception") for case in dataset.cases
            )

            wrapped_func = unpack_inputs(func, has_raises=has_any_raises)

            if inspect.iscoroutinefunction(wrapped_func):
                report = asyncio.run(dataset.evaluate(wrapped_func))
            else:
                report = dataset.evaluate_sync(wrapped_func)

            results.append(EvalResult(eval_id, report=report))
        except Exception as e:
            if debug:
                raise
            results.append(EvalResult(eval_id, error=e))

    return EvalSummary(results)

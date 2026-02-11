"""LLM-powered evaluation specification generator and function healer.

This module provides:
- EvalGenerator: Generate eval specs and heal buggy functions using LLMs
- generate_eval_spec: Generate YAML eval specs from function definitions
- prepare_agent: Create a pydantic_ai Agent for eval generation

Key Features:
- Auto-generate YAML eval specs from function code and description
- Heal buggy functions based on failing test inputs
- Retry logic with configurable coverage thresholds
- Support for async and sync function generation

Example:
    from vowel import EvalGenerator, Function

    generator = EvalGenerator(model="openai:gpt-4o")

    func = Function(
        name="factorial",
        description="Calculate factorial of n",
        code="def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)"
    )

    summary = generator.generate_and_run(
        func,
        auto_retry=True,
        heal_function=True,
        min_coverage=0.9
    )
"""

import os
import time
from typing import Any

import logfire
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, format_as_xml

from vowel.context import EVAL_SPEC_CONTEXT
from vowel.eval_types import EvalsSource
from vowel.runner import Function, RunEvals
from vowel.utils import EvalSummary, check_compatibility, import_function
from vowel.validation import validate_and_fix_spec


class UnsupportedParameterTypeError(Exception):
    """Raised when a function has parameters that cannot be serialized to YAML."""

    def __init__(self, func_name: str, issues: list[str]):
        self.func_name = func_name
        self.issues = issues
        issues_str = "\n  - ".join(issues)
        super().__init__(
            f"Function '{func_name}' has parameters with types that cannot be passed from YAML:\n  - {issues_str}\n"
            f"Eval generation skipped. Consider using simpler types (int, float, str, bool, list, dict, etc.)"
        )


def _safe_repr(value: object) -> str:
    try:
        if isinstance(value, Exception):
            return f"{type(value).__name__}: {value!s}"
        return repr(value)
    except Exception:
        return str(type(value))


def extract_failed_cases(summary: EvalSummary) -> list[dict]:
    failed_cases_info = []
    for result in summary.results:
        if result.report:
            for case in result.report.cases:
                if not all(a.value for a in case.assertions.values()):
                    failed_cases_info.append(
                        {
                            "case_id": case.name,
                            "input": _safe_repr(case.inputs),
                            "actual_output": _safe_repr(case.output),
                            "expected_output": _safe_repr(case.expected_output),
                            "failed_assertions": [
                                {"name": n, "reason": str(getattr(a, "reason", "N/A"))}
                                for n, a in case.assertions.items()
                                if not a.value
                            ],
                        }
                    )
    return failed_cases_info


def format_failed_inputs(failed_cases_info: list[dict]) -> str:
    """Format failed cases showing only inputs, not expected values."""
    return "\n".join([f"- Input: {fc['input']}" for fc in failed_cases_info])


class GenerationResult(BaseModel):
    """Result of eval generation containing final yaml, function, and summary.

    Attributes:
        yaml_spec: The final YAML specification (after all retries)
        func: The final function (healed or original)
        summary: The evaluation summary with test results
        was_healed: Whether the function was healed during the process
    """

    yaml_spec: str = Field(description="Final YAML eval specification")
    func: Function = Field(description="Final function (healed or original)")
    summary: EvalSummary = Field(description="Evaluation results summary")
    was_healed: bool = Field(default=False, description="Whether function was healed")

    model_config = {"arbitrary_types_allowed": True}

    def print(
        self,
        *,
        show_yaml: bool = True,
        show_func: bool = True,
        show_summary: bool = True,
        show_evaluations: bool = True,
        theme: str = "monokai",
    ) -> None:
        """Pretty print the generation result with syntax highlighting.

        Args:
            show_yaml: Whether to show the YAML specification
            show_func: Whether to show the final function code
            show_summary: Whether to show the evaluation summary
            theme: Syntax highlighting theme (monokai, dracula, github-dark, etc.)
        """
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
        except ImportError:
            self._print_simple(show_yaml=show_yaml, show_func=show_func, show_summary=show_summary)
            return

        if show_evaluations:
            self.summary.print()

        console = Console()

        console.print("\n" + "=" * 60)
        console.print("[bold green]📊 GENERATION RESULT[/bold green]")
        console.print("=" * 60)

        if show_summary:
            console.print()
            summary_table = Table(title="📈 Evaluation Summary", box=box.ROUNDED, show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta")

            summary_table.add_row("Total Functions", str(self.summary.total_count))
            summary_table.add_row("✅ Passed", f"[green]{self.summary.success_count}[/green]")
            summary_table.add_row(
                "❌ Failed",
                f"[red]{self.summary.failed_count}[/red]" if self.summary.failed_count else "0",
            )
            summary_table.add_row(
                "⚠️ Errors",
                f"[yellow]{self.summary.error_count}[/yellow]" if self.summary.error_count else "0",
            )
            summary_table.add_row("Coverage", f"{self.summary.coverage * 100:.1f}%")
            summary_table.add_row(
                "Status",
                "[green]✅ ALL PASSED[/green]"
                if self.summary.all_passed
                else "[red]❌ FAILURES[/red]",
            )
            summary_table.add_row(
                "Was Healed", "[yellow]🔧 Yes[/yellow]" if self.was_healed else "No"
            )
            console.print(summary_table)

        if show_yaml:
            console.print()
            console.print(
                Panel(
                    Syntax(self.yaml_spec, "yaml", theme=theme, line_numbers=True),
                    title="[yellow]📝 YAML Specification[/yellow]",
                )
            )

        if show_func:
            console.print()
            healed = " [bold yellow](HEALED)[/bold yellow]" if self.was_healed else ""
            desc = (
                self.func.description[:60] + "..."
                if len(self.func.description) > 60
                else self.func.description
            )
            console.print(
                Panel(
                    Syntax(self.func.code, "python", theme=theme, line_numbers=True),
                    title=f"[green]🐍 Final Function{healed}[/green]",
                    subtitle=f"[dim]{self.func.name}: {desc}[/dim]",
                )
            )

        console.print()

    def _print_simple(
        self,
        *,
        show_yaml: bool = True,
        show_func: bool = True,
        show_summary: bool = True,
    ) -> None:
        """Fallback simple print without rich library."""
        print("\n" + "=" * 60)
        print("GENERATION RESULT")
        print("=" * 60)

        if show_summary:
            print("\n📈 Summary:")
            print(f"  Total: {self.summary.total_count}")
            print(f"  Passed: {self.summary.success_count}")
            print(f"  Failed: {self.summary.failed_count}")
            print(f"  Coverage: {self.summary.coverage * 100:.1f}%")
            print(f"  Was Healed: {'Yes' if self.was_healed else 'No'}")

        if show_yaml:
            print(f"\n📝 YAML Specification:\n{self.yaml_spec}")

        if show_func:
            healed = " (HEALED)" if self.was_healed else ""
            print(f"\n🐍 Final Function{healed}:\n{self.func.code}")


class EvalGenerator:
    def __init__(
        self,
        model: str | None = None,
        additional_context: str | list[str] | None = None,
        load_env: bool = False,
    ):
        if load_env:
            import dotenv

            dotenv.load_dotenv()
        self.model = model or os.getenv("MODEL_NAME")
        if not self.model:
            logfire.warn(
                "Model name must be provided or set via MODEL_NAME env var. Agent property will raise ValueError on access."
            )
        self.additional_context = additional_context
        self._agent: Agent[None, EvalsSource] | Any = None
        logfire.info("EvalGenerator initialized", model=self.model)

    @property
    def agent(self) -> Agent[None, EvalsSource]:
        if self._agent is None:
            if not self.model:
                raise ValueError("Model name must be provided or set via MODEL_NAME env var")
            self._agent = prepare_agent(self.model, self.additional_context)
        return self._agent

    def generate_spec(
        self,
        func: Function,
        additional_context: str = "",
        save_to_file: bool = True,
        retries: int = 5,
    ) -> tuple[RunEvals, str]:
        """Generate eval spec for a function.

        Returns:
            Tuple of (RunEvals runner, yaml_spec string)
        """
        is_compatible, issues = check_compatibility(func.impl)

        if not is_compatible:
            logfire.error("Unsupported parameter types", func_name=func.name, issues=issues)
            raise UnsupportedParameterTypeError(func.name, issues)

        return generate_eval_spec(
            func=func,
            eval_generator=self.agent,
            additional_context=additional_context,
            save_to_file=save_to_file,
            retries=retries,
        )

    def generate_and_run(
        self,
        func: Function,
        additional_context: str = "",
        auto_retry: bool = False,
        retry_delay: float = 5.0,
        max_retries: int = 1,
        min_coverage: float = 1.0,
        heal_function: bool = True,
        ignore_duration: bool = False,
    ) -> GenerationResult:
        """Generate eval spec, run tests, and optionally heal function.

        Returns:
            GenerationResult containing final yaml_spec, func (healed or original), and summary
        """
        current_func = func
        was_healed = False
        retries_left = max_retries

        def _run(r: RunEvals) -> EvalSummary:
            if ignore_duration:
                r = r.ignore_duration()
            return r.run()

        with logfire.span(
            "Generating and running evals for {func_name}",
            func_name=func.name,
            auto_retry=auto_retry,
            max_retries=max_retries,
            min_coverage=min_coverage,
        ):
            while True:
                runner, yaml_spec = self.generate_spec(current_func, additional_context)

                try:
                    if runner._functions:
                        summary = _run(runner)
                    else:
                        impl = (
                            import_function(current_func.func_path)
                            if current_func.file_path
                            else current_func.impl
                        )
                        summary = _run(runner.with_functions({current_func.name: impl}))
                except Exception as e:
                    logfire.error("Error running evals", func_name=current_func.name, error=str(e))
                    if auto_retry and retries_left > 0:
                        retries_left -= 1
                        additional_context = f"Previous error: {str(e)[:500]}"
                        time.sleep(retry_delay)
                        continue
                    raise

                logfire.info(
                    "Eval completed",
                    func_name=current_func.name,
                    passed=summary.success_count,
                    failed=summary.failed_count,
                    errors=summary.error_count,
                    coverage=f"{summary.coverage * 100:.1f}%",
                )

                if summary.meets_coverage(min_coverage) and not summary.has_errors:
                    return GenerationResult(
                        yaml_spec=yaml_spec,
                        func=current_func,
                        summary=summary,
                        was_healed=was_healed,
                    )

                if not (auto_retry and retries_left > 0):
                    return GenerationResult(
                        yaml_spec=yaml_spec,
                        func=current_func,
                        summary=summary,
                        was_healed=was_healed,
                    )

                time.sleep(retry_delay)

                if summary.has_errors:
                    with logfire.span("Retrying after errors", func_name=current_func.name):
                        runner, yaml_spec = self.generate_spec(
                            current_func, f"Fix ERRORS:\n{format_as_xml(summary.to_json())}"
                        )
                        summary = _run(
                            runner.with_functions({current_func.name: current_func.impl})
                        )
                        if summary.meets_coverage(min_coverage):
                            return GenerationResult(
                                yaml_spec=yaml_spec,
                                func=current_func,
                                summary=summary,
                                was_healed=was_healed,
                            )
                        retries_left -= 1
                        continue

                elif summary.has_failed_cases:
                    if heal_function:
                        with logfire.span("Healing buggy function", func_name=current_func.name):
                            time.sleep(retry_delay)
                            healed_func = self._heal_function(current_func, summary)
                            if healed_func:
                                healed_summary = _run(
                                    runner.with_functions({current_func.name: healed_func.impl})
                                )
                                if healed_summary.meets_coverage(min_coverage):
                                    logfire.info(
                                        "Function healed successfully",
                                        func_name=current_func.name,
                                        coverage=f"{healed_summary.coverage * 100:.1f}%",
                                    )
                                    return GenerationResult(
                                        yaml_spec=yaml_spec,
                                        func=healed_func,
                                        summary=healed_summary,
                                        was_healed=True,
                                    )
                                if healed_summary.coverage > summary.coverage:
                                    logfire.info(
                                        "Heal partial success, continuing with healed function",
                                        old_coverage=f"{summary.coverage * 100:.1f}%",
                                        new_coverage=f"{healed_summary.coverage * 100:.1f}%",
                                    )
                                    current_func = healed_func
                                    summary = healed_summary
                                    was_healed = True
                                else:
                                    logfire.warn(
                                        "Heal did not improve coverage, trying test case regeneration"
                                    )

                    with logfire.span("Regenerating test cases", func_name=current_func.name):
                        time.sleep(retry_delay)
                        runner, yaml_spec = self.generate_spec(
                            current_func,
                            f"Previous test cases had wrong expected values. Regenerate with correct expected outputs based on the function implementation.\n{format_as_xml(summary.to_json())}",
                        )
                        summary = _run(
                            runner.with_functions({current_func.name: current_func.impl})
                        )
                        if summary.meets_coverage(min_coverage):
                            return GenerationResult(
                                yaml_spec=yaml_spec,
                                func=current_func,
                                summary=summary,
                                was_healed=was_healed,
                            )

                retries_left -= 1
                if retries_left <= 0:
                    return GenerationResult(
                        yaml_spec=yaml_spec,
                        func=current_func,
                        summary=summary,
                        was_healed=was_healed,
                    )

    def _heal_function(self, func: Function, summary: EvalSummary) -> Function | None:
        failed_cases_info = extract_failed_cases(summary)
        if not failed_cases_info:
            return None

        failed_inputs_text = format_failed_inputs(failed_cases_info)
        is_async = "async def" in func.code

        prompt = f"""The function below has a bug. Fix it so it works correctly according to its description.

<BuggyFunction>
{func.code}
</BuggyFunction>

<FunctionDescription>
{func.description}
</FunctionDescription>

<FailingInputs>
These inputs are producing wrong results:
{failed_inputs_text}
</FailingInputs>

REQUIREMENTS:
- Keep the function name exactly: {func.name}
- Keep it {"ASYNC (async def)" if is_async else "SYNC (def)"}
- Fix the bug based on what the DESCRIPTION says the function should do
- Do NOT change the function signature
- Common bugs to look for: off-by-one errors, wrong operators (>= vs >), wrong variable names, missing base cases
"""
        try:
            healed_func = self.agent.run_sync(prompt, output_type=Function).output
            if healed_func.name != func.name:
                healed_func.name = func.name
            _ = healed_func.impl  # Trigger compilation
            logfire.info(
                "Function healed", func_name=healed_func.name, new_code=healed_func.code[:200]
            )
            return healed_func
        except Exception as e:
            logfire.error("Failed to heal function", error=str(e))
            return None

    def generate_function(self, prompt: str, async_func: bool = True) -> Function:
        with logfire.span("Generating function from prompt", async_func=async_func):
            full_prompt = prompt + "\n\nALL parameters MUST have type annotations."
            if async_func:
                full_prompt += "\n\nGenerate an ASYNC function (async def)."
            return self.agent.run_sync(full_prompt, output_type=Function).output


def generate_eval_spec(
    func: Function,
    eval_generator: Agent[None, EvalsSource],
    *,
    additional_context: str = "",
    save_to_file: bool = True,
    retries: int = 5,
) -> tuple[RunEvals, str]:
    """Generate YAML eval spec and return both runner and spec string.

    Returns:
        Tuple of (RunEvals runner, yaml_spec string)
    """
    with logfire.span("Generating YAML spec via LLM", func_name=func.name, retries_left=retries):
        _ = func.impl  # Ensure function is compiled

        if retries == 0:
            raise RuntimeError("Failed to generate valid eval spec after multiple attempts.")

        user_context = (
            f"\n\n<UserContext>\n{additional_context}\n</UserContext>" if additional_context else ""
        )

        prompt = f"""Generate vowel evals YAML spec for `{func.name}`:

<PythonImpl>{func.code}</PythonImpl>
<Docstring>{func.description}</Docstring>{user_context}

Use {func.name} as eval_id.

ABOUT CASES:
- Given case scenarios: {", ".join(func.cases) if func.cases else "[]"}
- Generate at least {max(len(func.cases), 5)} diverse test cases covering normal, edge, and error scenarios.
- Ensure all test cases are unique and cover a wide range of input values.
- Ensure all test cases are valid according to the function description.
- Use correct expected outputs based on the function logic.

CRITICAL - ASSERTION VARIABLES:
In assertions, use `input` (NOT `inputs`) to access the input value(s).
- For positional args: `input[0]`, `input[1]`, etc.
- Available variables: `input`, `output`, `expected`, `duration`, `metadata`
- WRONG: `inputs[0]` ❌
- CORRECT: `input[0]` ✅
"""

        result = eval_generator.run_sync(prompt, output_type=EvalsSource)

        try:
            yaml.safe_load(result.output.yaml_spec)

            # Static validation: fix common LLM generation mistakes
            validation = validate_and_fix_spec(
                result.output.yaml_spec,
                function_code=func.code,
            )
            if validation.has_warnings:
                logfire.info("Spec validation results", summary=validation.summary())
            spec_to_use = (
                validation.fixed_yaml if validation.was_modified else result.output.yaml_spec
            )

            if save_to_file:
                with open(f"{func.name}_evals.yml", "w") as f:
                    f.write(spec_to_use)

            runner = RunEvals.from_source(spec_to_use)
            if func.func:
                runner = runner.with_functions({func.name: func.func})
            return runner, spec_to_use

        except Exception as e:
            logfire.warn(
                "YAML validation failed, retrying",
                func_name=func.name,
                retries_left=retries - 1,
                error=str(e),
            )
            return generate_eval_spec(
                func, eval_generator, additional_context=additional_context, retries=retries - 1
            )


def prepare_agent(
    model: str, additional_context: str | list[str] | None = None
) -> Agent[None, EvalsSource]:
    ctx = ""
    if additional_context:
        ctx = (
            additional_context
            if isinstance(additional_context, str)
            else "\n\n".join([f"<ContextPart>\n{d}\n</ContextPart>" for d in additional_context])
        )

    system_prompt = f"""You are an expert vowel YAML SPEC generator.

<EvalsInstructions>{EVAL_SPEC_CONTEXT}</EvalsInstructions>
<AdditionalContext>{ctx}</AdditionalContext>
"""
    return Agent(model, output_type=EvalsSource, system_prompt=system_prompt)

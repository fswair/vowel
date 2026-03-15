"""CodeMode eval generation pipeline.

This module provides ``CodeModeGenerator`` — a two-phase pipeline that uses
a sandboxed code executor to produce ground-truth expected values before
generating YAML eval specs.

Phase 1 — **Exploration**
    The LLM writes small Python snippets that call ``target_func`` with various
    inputs.  Each snippet is executed via ``Executor`` (Monty sandbox by default)
    and the real outputs are collected.  This replaces guesswork with empirical
    observation.

Phase 2 — **Spec Generation**
    The exploration results (inputs → outputs, edge cases, exceptions) are fed
    back to the LLM together with the eval spec context.  The LLM produces the
    final YAML spec with verified expected values.

All steps are instrumented with ``logfire`` for full observability.
"""

from __future__ import annotations

import os
import time
from typing import Any

import logfire
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from vowel.context import EVAL_SPEC_CONTEXT
from vowel.eval_types import EvalsSource
from vowel.executor import ExecutionResult, Executor, get_executor
from vowel.monitoring import enable_monitoring
from vowel.runner import Function, RunEvals
from vowel.spec_validation import (
    build_call_code,
    build_failure_context,
    inject_durations,
    inject_missing_error_cases,
    validate_expected_values,
)
from vowel.utils import EvalSummary
from vowel.validation import validate_and_fix_spec

enable_monitoring(service_name="vowel-codemode")


# ---------------------------------------------------------------------------
# Exploration output model — what the LLM returns in Phase 1
# ---------------------------------------------------------------------------


class ExplorationSnippet(BaseModel):
    """A single exploration snippet that tests normal (non-error) behaviour."""

    description: str = Field(
        description="One-line description of what this snippet tests "
        "(e.g. 'empty list edge case', 'negative numbers').",
    )
    code: str = Field(
        description="Python code to execute.  May call ``target_func(...)`` "
        "which is the function under test.  The value of the last "
        "expression is captured as output.",
    )


class ErrorSnippet(BaseModel):
    """A snippet that should trigger an exception from the function."""

    description: str = Field(
        description="What error scenario this tests "
        "(e.g. 'None input', 'division by zero', 'wrong type').",
    )
    code: str = Field(
        description="Python code that should trigger an exception.  "
        "Use the function's real name — the source is prepended at runtime.",
    )


class ExplorationPlan(BaseModel):
    """LLM output for Phase 1: normal snippets + error-triggering snippets."""

    snippets: list[ExplorationSnippet] = Field(
        description="Snippets that test NORMAL (succeeding) behaviour: "
        "happy-path, boundary values, return type exploration, "
        "equivalence partitioning, invariants, composition.",
        min_length=10,
    )
    error_snippets: list[ErrorSnippet] = Field(
        description="Snippets that should TRIGGER EXCEPTIONS: wrong types, "
        "invalid values, None inputs, out-of-range arguments.  "
        "Every guard clause and raise statement in the function "
        "must be exercised by at least one error snippet.",
        min_length=3,
    )


# ---------------------------------------------------------------------------
# Exploration result — what we feed back to Phase 2
# ---------------------------------------------------------------------------


class SnippetResult(BaseModel):
    """Result of executing a single exploration snippet."""

    description: str
    code: str
    success: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_execution(
        cls,
        snippet: ExplorationSnippet | ErrorSnippet,
        result: ExecutionResult,
    ) -> SnippetResult:
        return cls(
            description=snippet.description,
            code=snippet.code,
            success=result.success,
            output=result.output,
            stdout=result.stdout,
            error=result.error,
            error_type=result.error_type,
            duration_ms=result.duration_ms,
        )

    def to_context_block(self) -> str:
        """Format as a context block for the spec-generation prompt."""
        if self.success:
            out = repr(self.output)
            return (
                f"# {self.description}\n"
                f">>> {self.code.strip()}\n"
                f"Output: {out}  ({self.duration_ms:.2f} ms)"
            )
        return (
            f"# {self.description}\n>>> {self.code.strip()}\nRAISED {self.error_type}: {self.error}"
        )


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


class CodeModeResult(BaseModel):
    """Full result of the CodeMode generation pipeline."""

    exploration_results: list[SnippetResult] = Field(
        description="Results from Phase 1 exploration.",
    )
    yaml_spec: str = Field(description="Final YAML eval specification.")
    summary: EvalSummary | None = Field(
        default=None,
        description="Eval run summary (if run_evals=True).",
    )
    refinement_rounds: int = Field(
        default=0,
        description="Number of refinement iterations needed (0 = first-pass success).",
    )

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# CodeModeGenerator
# ---------------------------------------------------------------------------


class CodeModeGenerator:
    """Two-phase eval generator: explore with executor, then generate spec.

    Parameters
    ----------
    model:
        LLM model identifier (e.g. ``"openai:gpt-4o"``).
    executor:
        Code execution backend.  Defaults to ``get_executor("auto")``
        which prefers MontyExecutor when available.
    additional_context:
        Extra instructions appended to the system prompt.
    """

    def __init__(
        self,
        model: str | None = None,
        executor: Executor | None = None,
        additional_context: str = "",
        min_snippets: int = 15,
        **opts,
    ) -> None:
        self.model = model or os.getenv("MODEL_NAME", "")
        if not self.model:
            logfire.warn("No model specified; set MODEL_NAME env var or pass model=")
        self.executor = executor or get_executor("auto")
        self.additional_context = additional_context
        self.min_snippets = min_snippets
        self._opts = opts

        # Lazy agents
        self._explorer_agent: Agent[None, ExplorationPlan] | None = None
        self._spec_agent: Agent[None, EvalsSource] | None = None

        logfire.info(
            "CodeModeGenerator initialized",
            model=self.model,
            executor=type(self.executor).__name__,
        )

    # -- Agent properties --------------------------------------------------

    @property
    def explorer_agent(self) -> Agent[None, ExplorationPlan]:
        if self._explorer_agent is None:
            self._explorer_agent = Agent(
                self.model,
                output_type=ExplorationPlan,
                system_prompt=self._explorer_system_prompt(),
                **self._opts,
            )
        return self._explorer_agent

    @property
    def spec_agent(self) -> Agent[None, EvalsSource]:
        if self._spec_agent is None:
            self._spec_agent = Agent(
                self.model,
                output_type=EvalsSource,
                system_prompt=self._spec_system_prompt(),
                **self._opts,
            )
        return self._spec_agent

    # -- System prompts ----------------------------------------------------

    def _explorer_system_prompt(self) -> str:
        return f"""You are a Python testing expert.  Your job is to write small
code snippets that explore a function's behaviour empirically.

You will receive:
- The function's source code (with its real name)
- The function's description

You produce TWO separate lists of snippets:

## `snippets` — Normal / succeeding behaviour
These snippets call the function with VALID inputs and capture the return
value.  They MUST cover:
1. Normal / happy-path behaviour (typical valid inputs)
2. Boundary values (empty collections, zero, negative, very large, min/max)
3. Return type and structure exploration
4. Equivalence partitioning (representative from each input class)
5. Invariant verification (e.g. idempotency, commutativity, sort stability)
6. Composition / interaction (combining parameters, dependent arguments)

Produce AT LEAST {self.min_snippets} normal snippets.

## `error_snippets` — Exception-triggering inputs
These snippets call the function with inputs that SHOULD RAISE exceptions.
They MUST cover:
1. Wrong types (None, int instead of list, str instead of int, etc.)
2. Invalid values (out-of-range, malformed strings, empty when not allowed)
3. Every `raise` statement and guard clause in the function source code

Produce AT LEAST 3 error snippets.  If the function has more raise
statements or guard clauses, produce MORE — one per distinct error path.

STRICT RULES:
- Each snippet MUST end with an expression whose value will be captured.
- Use the function's REAL NAME — the function source code will be prepended
  automatically at runtime.  Do NOT define the function yourself.
- Keep each snippet focused on ONE scenario.
- Do NOT guess outputs — the snippets will be executed and the real
  outputs collected automatically.
- NEVER use try/except in your snippets.  Let exceptions propagate
  naturally — the execution environment captures raised errors
  automatically.  For example, write `flatten(None)` NOT
  `try: flatten(None) except Exception as e: type(e)`.
- `snippets` must contain ONLY inputs expected to SUCCEED.
- `error_snippets` must contain ONLY inputs expected to RAISE exceptions.
  Do NOT mix them."""

    def _spec_system_prompt(self) -> str:
        ctx = ""
        if self.additional_context:
            ctx = f"\n\n<AdditionalContext>\n{self.additional_context}\n</AdditionalContext>"
        return f"""You are an expert vowel YAML SPEC generator.

<EvalsInstructions>{EVAL_SPEC_CONTEXT}</EvalsInstructions>{ctx}

CRITICAL: You have access to VERIFIED execution results below.  Use the
EXACT outputs shown — do NOT guess or calculate expected values yourself.
The execution results are ground-truth from running the real function."""

    # -- Phase 1: Exploration ----------------------------------------------

    async def explore(
        self,
        func: Function,
    ) -> list[SnippetResult]:
        """Phase 1: Generate and execute exploration snippets.

        Uses ``create_session()`` to compile the function source **once**,
        then feeds each snippet against the preserved runtime state —
        zero re-parse overhead per snippet.

        Returns a list of ``SnippetResult`` with real outputs from the
        executor.
        """
        with logfire.span(
            "codemode.explore",
            func_name=func.name,
            executor=type(self.executor).__name__,
        ):
            # 1. Ask the LLM for exploration snippets
            plan = await self._get_exploration_plan(func)

            # 2. Compile function source once, feed each snippet
            all_snippets = [
                *((s, "normal") for s in plan.snippets),
                *((s, "error") for s in plan.error_snippets),
            ]
            total = len(all_snippets)
            results: list[SnippetResult] = []
            with self.executor.create_session(func.code) as session:
                for i, (snippet, kind) in enumerate(all_snippets):
                    with logfire.span(
                        "codemode.execute_snippet",
                        index=i,
                        kind=kind,
                        description=snippet.description,
                    ):
                        logfire.info(
                            "Executing snippet {index}/{total} [{kind}]: {description}",
                            index=i + 1,
                            total=total,
                            kind=kind,
                            description=snippet.description,
                            code=snippet.code,
                        )

                        exec_result = session.feed(snippet.code)

                        sr = SnippetResult.from_execution(snippet, exec_result)
                        results.append(sr)

                        logfire.info(
                            "Snippet result: success={success}, output={output}, "
                            "duration={duration_ms:.2f}ms",
                            success=sr.success,
                            output=repr(sr.output)[:200],
                            duration_ms=sr.duration_ms,
                            error=sr.error,
                            error_type=sr.error_type,
                        )

            # Summary log
            successes = sum(1 for r in results if r.success)
            failures = len(results) - successes
            logfire.info(
                "Exploration complete: {successes} succeeded, {failures} raised errors",
                successes=successes,
                failures=failures,
            )

            return results

    async def _get_exploration_plan(self, func: Function) -> ExplorationPlan:
        """Ask the LLM for exploration snippets."""
        with logfire.span("codemode.llm_explore", func_name=func.name):
            prompt = f"""Explore the following function by writing test snippets:

<FunctionName>{func.name}</FunctionName>
<FunctionCode>
{func.code}
</FunctionCode>
<Description>{func.description}</Description>

Write diverse snippets that call {func.name}(...) to discover the function's
behaviour across all important scenarios.  Use the real function name
`{func.name}` — the implementation will be prepended automatically."""

            result = await self.explorer_agent.run(prompt)
            plan = result.output

            logfire.info(
                "LLM produced {normal} normal + {error} error snippets",
                normal=len(plan.snippets),
                error=len(plan.error_snippets),
                snippets=[s.description for s in plan.snippets],
                error_snippets=[s.description for s in plan.error_snippets],
            )
            return plan

    # -- Phase 2: Spec Generation ------------------------------------------

    async def generate_spec(
        self,
        func: Function,
        exploration_results: list[SnippetResult],
        failure_context: str | None = None,
    ) -> str:
        """Phase 2: Generate YAML spec using verified exploration results.

        Parameters
        ----------
        failure_context:
            When provided (on refinement rounds), appended to the prompt so
            the LLM can fix specific failures from the previous attempt.
        """
        with logfire.span(
            "codemode.generate_spec",
            func_name=func.name,
            n_results=len(exploration_results),
            is_refinement=failure_context is not None,
        ):
            # Build exploration context for the prompt
            success_results = [r for r in exploration_results if r.success]
            error_results = [r for r in exploration_results if not r.success]

            success_context = (
                "\n\n".join(r.to_context_block() for r in success_results)
                if success_results
                else "(none)"
            )
            error_context = (
                "\n\n".join(r.to_context_block() for r in error_results)
                if error_results
                else "(none)"
            )

            refinement_block = ""
            if failure_context:
                refinement_block = f"""

⚠️ PREVIOUS ATTEMPT FAILED — fix these issues:
{failure_context}

Regenerate the YAML spec addressing every failure above.  Keep all
passing cases intact — only fix the broken ones."""

            prompt = f"""Generate vowel evals YAML spec for `{func.name}`:

<PythonImpl>
{func.code}
</PythonImpl>

<Docstring>{func.description}</Docstring>

<VerifiedExecutionResults>
The following results are from ACTUALLY RUNNING the function — use these
exact outputs as expected values:

{success_context}
</VerifiedExecutionResults>

<ErrorResults count="{len(error_results)}">
These inputs RAISED exceptions when run against the real function.
Each one MUST become a `raises:` case in the spec — no exceptions.

{error_context}
</ErrorResults>

REQUIREMENTS:
- Use {func.name} as eval_id.
- Generate at least {max(len(exploration_results), 5)} diverse test cases.
- Use the EXACT outputs from the execution results above.
- You MUST generate exactly {len(error_results)} raises cases — one for
  each RAISED result above.  The spec is invalid without them.
- Cover normal, edge, and error cases.
- In assertions, use `input` (NOT `inputs`) for accessing input values.
{refinement_block}"""

            logfire.info(
                "Sending spec generation prompt",
                func_name=func.name,
                success_results=len(success_results),
                error_results=len(error_results),
            )

            result = await self.spec_agent.run(prompt)
            yaml_spec = result.output.yaml_spec

            # Sanitize: strip YAML tags that safe_load rejects
            import re

            yaml_spec = re.sub(r"!!python/[\w.:]+", "", yaml_spec)
            yaml_spec = re.sub(r"!!binary\b", "", yaml_spec)

            # Validate YAML syntax
            yaml.safe_load(yaml_spec)

            # Validate and auto-fix
            validation = validate_and_fix_spec(
                yaml_spec,
                function_code=func.code,
            )
            if validation.has_warnings:
                logfire.info(
                    "Spec validation applied fixes",
                    summary=validation.summary(),
                )
            final_spec = validation.fixed_yaml if validation.was_modified else yaml_spec

            # Executor-based validation: fix expected values against real execution
            final_spec = validate_expected_values(final_spec, func, self.executor)

            # Inject missing error cases from exploration
            error_snippet_dicts = [
                {
                    "code": r.code,
                    "error_type": r.error_type,
                    "error": r.error,
                    "description": r.description,
                }
                for r in exploration_results
                if not r.success and r.error_type
            ]
            final_spec = inject_missing_error_cases(final_spec, func.name, error_snippet_dicts)

            logfire.info(
                "YAML spec generated",
                func_name=func.name,
                spec_length=len(final_spec),
                spec_preview=final_spec[:500],
            )

            return final_spec

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _build_failure_context(summary: EvalSummary) -> str:
        """Build a concise failure report to inject into the retry prompt."""
        return build_failure_context(summary)

    def _inject_durations(
        self,
        yaml_spec: str,
        func: Function,
        *,
        buffer_pct: float = 0.5,
        floor_ms: float = 10.0,
    ) -> str:
        """Add per-case ``duration`` fields based on actual execution times."""
        return inject_durations(
            yaml_spec,
            func,
            self.executor,
            buffer_pct=buffer_pct,
            floor_ms=floor_ms,
        )

    @staticmethod
    def _build_call_code(func_name: str, case: dict) -> str | None:
        """Build a ``func(args...)`` call string from a case dict."""
        return build_call_code(func_name, case)

    # -- Full pipeline -----------------------------------------------------

    async def generate(
        self,
        func: Function,
        *,
        run_evals: bool = True,
        save_to_file: bool = False,
        max_refinement_rounds: int = 2,
        min_coverage: float = 1.0,
        inject_durations: bool = True,
    ) -> CodeModeResult:
        """Run the full CodeMode pipeline with post-generation validation.

        Pipeline::

            Phase 1: explore()                        (once)
            Phase 2: generate_spec()                  (may loop)
            Phase 3: validate via RunEvals            (per attempt)
            Phase 4: refine on failure                (up to N rounds)
            Phase 5: inject_durations()               (once, at end)

        Exploration (Phase 1) runs once — the ground-truth snippet results
        don't change.  Only spec generation (Phase 2) is re-run on failure,
        with a failure report injected into the prompt.

        Parameters
        ----------
        func:
            The function to generate evals for.
        run_evals:
            Whether to run the generated evals and include the summary.
        save_to_file:
            Whether to save the YAML spec to ``{func.name}_evals.yml``.
        max_refinement_rounds:
            Maximum number of spec-regeneration attempts after the initial
            generation (0 = single attempt, no refinement).
        min_coverage:
            Target pass-rate in 0.0–1.0 (default 1.0 = 100 %).  The loop
            exits early when coverage meets or exceeds this threshold.
        inject_durations:
            Whether to measure and inject per-case ``duration`` fields
            into the final YAML spec.

        Returns
        -------
        CodeModeResult
            Contains exploration results, YAML spec, summary, and
            the number of refinement rounds used.
        """
        with logfire.span(
            "codemode.pipeline",
            func_name=func.name,
            model=self.model,
            executor=type(self.executor).__name__,
        ):
            t0 = time.perf_counter()

            # Phase 1 — explore (once)
            exploration_results = await self.explore(func)

            # Phase 2–4 — generate spec + validate + refine
            yaml_spec = ""
            summary: EvalSummary | None = None
            refinement_rounds = 0
            failure_context: str | None = None
            total_attempts = max_refinement_rounds + 1 if run_evals else 1

            for attempt in range(total_attempts):
                with logfire.span(
                    "codemode.spec_attempt",
                    attempt=attempt + 1,
                    is_refinement=attempt > 0,
                ):
                    try:
                        yaml_spec = await self.generate_spec(
                            func,
                            exploration_results,
                            failure_context,
                        )
                    except Exception as gen_exc:
                        logfire.warn(
                            "Spec generation failed on attempt {attempt}, retrying",
                            attempt=attempt + 1,
                            error=str(gen_exc),
                        )
                        failure_context = f"Generation error: {gen_exc}"
                        refinement_rounds = attempt + 1
                        continue

                    if not run_evals:
                        break

                    # Validate: run evals with ignore_duration=True
                    try:
                        runner = (
                            RunEvals.from_source(yaml_spec)
                            .with_functions({func.name: func.impl})
                            .ignore_duration()
                        )
                        summary = runner.run()

                        logfire.info(
                            "Attempt {attempt}: {passed}/{total} passed, coverage={coverage:.1f}%",
                            attempt=attempt + 1,
                            passed=summary.success_count,
                            total=summary.total_count,
                            failed=summary.failed_count,
                            errors=summary.error_count,
                            coverage=summary.coverage * 100,
                        )

                        if summary.coverage >= min_coverage:
                            break

                        # Build failure context for next attempt
                        failure_context = self._build_failure_context(summary)
                        refinement_rounds = attempt + 1
                        logfire.warn(
                            "Coverage {coverage:.0f}% below target {target:.0f}%, refining",
                            coverage=summary.coverage * 100,
                            target=min_coverage * 100,
                            attempt=attempt + 1,
                        )

                    except Exception as exc:
                        logfire.warn(
                            "Failed to run evals on attempt {attempt}, retrying",
                            attempt=attempt + 1,
                            func_name=func.name,
                            error=str(exc),
                        )
                        failure_context = f"Eval run error: {exc}"
                        refinement_rounds = attempt + 1
                        continue

            # Phase 5 — inject per-case durations
            if inject_durations:
                with logfire.span("codemode.inject_durations", func_name=func.name):
                    yaml_spec = self._inject_durations(yaml_spec, func)

            # Final summary run (with durations now present, but still ignored)
            if run_evals and summary is not None:
                try:
                    final_runner = (
                        RunEvals.from_source(yaml_spec)
                        .with_functions({func.name: func.impl})
                        .ignore_duration()
                    )
                    summary = final_runner.run()
                except Exception:  # noqa: BLE001
                    pass  # keep last good summary

            if save_to_file:
                path = f"{func.name}_evals.yml"
                with open(path, "w") as f:
                    f.write(yaml_spec)
                logfire.info("Saved spec to {path}", path=path)

            elapsed = (time.perf_counter() - t0) * 1000
            logfire.info(
                "CodeMode pipeline complete in {elapsed:.0f}ms (refinements={rounds})",
                elapsed=elapsed,
                func_name=func.name,
                exploration_count=len(exploration_results),
                refinement_rounds=refinement_rounds,
                has_summary=summary is not None,
            )

            return CodeModeResult(
                exploration_results=exploration_results,
                yaml_spec=yaml_spec,
                summary=summary,
                refinement_rounds=refinement_rounds,
            )

"""CodeMode pipeline for execution-aware eval spec generation.

CodeMode uses real execution feedback to generate robust vowel eval specs:
1. Explore behavior by running LLM-generated snippets against the target code.
2. Generate and refine a spec from verified outputs/errors.

The pipeline supports both YAML output and structured bundle output, and keeps
traceability via logfire spans.
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
from vowel.executor import ExecutionResult, Executor, resolve_executors
from vowel.monitoring import enable_monitoring
from vowel.runner import Function, RunEvals
from vowel.schema import materialize_yaml_with_schema_header
from vowel.utils import EvalsBundle, EvalSummary
from vowel.validation import (
    build_call_code,
    build_failure_context,
    inject_durations,
    inject_missing_error_cases,
    validate_and_fix_spec,
    validate_expected_values,
)

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
    """Execution-guided eval generator.

    The generator first discovers behavior by running snippets, then produces
    a validated eval spec (YAML or bundle) from those verified results.
    """

    def __init__(
        self,
        spec_model: str | None = None,
        exploration_model: str | None = None,
        default_executor: Executor | None = None,
        fallback_executor: Executor | None = None,
        additional_context: str = "",
        min_snippets: int = 15,
        use_model_spec: bool = False,
        **opts,
    ) -> None:
        # Default fallback from kwargs (for backwards compatibility) or environment
        base_fallback = opts.pop("model", None) or os.getenv("MODEL_NAME", "")

        self.spec_model = spec_model or os.getenv("SPEC_MODEL") or base_fallback
        self.exploration_model = (
            exploration_model or os.getenv("EXPLORATION_MODEL") or base_fallback
        )

        if not self.spec_model or not self.exploration_model:
            raise ValueError(
                "Both spec_model and exploration_model must be specified. "
                "Provide them via constructor/kwargs, or set SPEC_MODEL, EXPLORATION_MODEL, or MODEL_NAME environment variables."
            )

        self.executor = resolve_executors(default_executor, fallback_executor)
        self.additional_context = additional_context
        self.min_snippets = min_snippets
        self.use_model_spec = use_model_spec
        self._opts = opts

        # Lazy agents
        self._explorer_agent: Agent[None, ExplorationPlan] | None = None
        self._spec_agent: Agent[None, EvalsSource | EvalsBundle] | None = None

        logfire.info(
            "CodeModeGenerator initialized",
            spec_model=self.spec_model,
            exploration_model=self.exploration_model,
            executor=type(self.executor).__name__,
        )

    # -- Agent properties --------------------------------------------------

    @property
    def explorer_agent(self) -> Agent[None, ExplorationPlan]:
        if self._explorer_agent is None:
            self._explorer_agent = Agent(
                self.exploration_model,
                output_type=ExplorationPlan,
                system_prompt=self._explorer_system_prompt(),
                **self._opts,
            )
        return self._explorer_agent

    @property
    def spec_agent(self) -> Agent[None, EvalsSource | EvalsBundle]:
        if self._spec_agent is None:
            output_type = EvalsBundle if self.use_model_spec else EvalsSource
            self._spec_agent = Agent(
                self.spec_model,
                output_type=output_type,
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
        *,
        exploration_rounds: int = 2,
    ) -> list[SnippetResult]:
        """Generate and execute exploration snippets.

        Round 1 discovers baseline behavior. Subsequent rounds receive prior
        execution evidence and target uncovered behavior classes.
        """
        with logfire.span(
            "codemode.explore",
            func_name=func.name,
            executor=type(self.executor).__name__,
            exploration_rounds=exploration_rounds,
        ):
            all_results: list[SnippetResult] = []

            for round_num in range(1, exploration_rounds + 1):
                with logfire.span(
                    "codemode.explore_round",
                    round=round_num,
                    prior_results=len(all_results),
                ):
                    # Get exploration plan (round 2+ includes prior context)
                    if round_num == 1:
                        plan = await self._get_exploration_plan(func)
                    else:
                        cluster_summary = self._build_cluster_summary(all_results)
                        plan = await self._get_targeted_exploration_plan(
                            func,
                            all_results,
                            cluster_summary,
                        )
                        # Early exit: if no new snippets were produced
                        if not plan.snippets and not plan.error_snippets:
                            logfire.info(
                                "Round {round} produced no new snippets, stopping",
                                round=round_num,
                            )
                            break

                    # Execute snippets
                    new_results = self._execute_plan(func, plan, round_num)
                    all_results.extend(new_results)

                    # Early exit: round 2+ found no new behaviour
                    if round_num > 1:
                        new_behaviors = self._count_new_behaviors(
                            all_results[: -len(new_results)],
                            new_results,
                        )
                        logfire.info(
                            "Round {round}: {new} new behaviour classes discovered",
                            round=round_num,
                            new=new_behaviors,
                        )

            # Summary log
            successes = sum(1 for r in all_results if r.success)
            failures = len(all_results) - successes
            logfire.info(
                "Exploration complete: {successes} succeeded, {failures} raised errors",
                successes=successes,
                failures=failures,
            )

            return all_results

    def _execute_plan(
        self,
        func: Function,
        plan: ExplorationPlan,
        round_num: int = 1,
    ) -> list[SnippetResult]:
        """Execute all snippets in an exploration plan and collect results."""
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
                    round=round_num,
                    description=snippet.description,
                ):
                    logfire.info(
                        "Executing snippet {index}/{total} R{round} [{kind}]: {description}",
                        index=i + 1,
                        total=total,
                        round=round_num,
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
        return results

    @staticmethod
    def _build_cluster_summary(results: list[SnippetResult]) -> str:
        """Summarize observed output/error clusters for targeted exploration."""
        # -- Success clusters --
        success_types: dict[str, int] = {}
        for r in results:
            if r.success:
                t = type(r.output).__name__
                success_types[t] = success_types.get(t, 0) + 1

        # -- Error clusters --
        error_clusters: dict[str, list[str]] = {}
        for r in results:
            if not r.success and r.error_type:
                msgs = error_clusters.setdefault(r.error_type, [])
                prefix = (r.error or "")[:60]
                if prefix not in msgs:
                    msgs.append(prefix)

        # -- Already-tried snippets (to avoid repeats) --
        tried_codes = [r.code.strip() for r in results]

        lines = ["## Observed Behaviour Clusters\n"]

        lines.append("### Success clusters")
        if success_types:
            for t, count in sorted(success_types.items()):
                lines.append(f"- output type `{t}`: {count} cases")
        else:
            lines.append("- (none)")

        lines.append("\n### Error clusters")
        if error_clusters:
            for etype, msgs in sorted(error_clusters.items()):
                lines.append(f"- `{etype}` ({len(msgs)} distinct messages):")
                for m in msgs:
                    lines.append(f'  - "{m}"')
        else:
            lines.append("- (none)")

        lines.append(f"\n### Already tried ({len(tried_codes)} snippets — do NOT repeat these)")
        for code in tried_codes:
            lines.append(f"- `{code}`")

        return "\n".join(lines)

    @staticmethod
    def _count_new_behaviors(
        prior: list[SnippetResult],
        new: list[SnippetResult],
    ) -> int:
        """Count new behavior signatures introduced by a round."""

        def _behavior_key(r: SnippetResult) -> str:
            if r.success:
                return f"ok:{type(r.output).__name__}"
            return f"err:{r.error_type}:{(r.error or '')[:40]}"

        prior_keys = {_behavior_key(r) for r in prior}
        new_keys = {_behavior_key(r) for r in new}
        return len(new_keys - prior_keys)

    async def _get_exploration_plan(self, func: Function) -> ExplorationPlan:
        """Request initial exploration snippets from the model."""
        with logfire.span("codemode.llm_explore", func_name=func.name, round=1):
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
                "Round 1: LLM produced {normal} normal + {error} error snippets",
                normal=len(plan.snippets),
                error=len(plan.error_snippets),
                snippets=[s.description for s in plan.snippets],
                error_snippets=[s.description for s in plan.error_snippets],
            )
            return plan

    async def _get_targeted_exploration_plan(
        self,
        func: Function,
        prior_results: list[SnippetResult],
        cluster_summary: str,
    ) -> ExplorationPlan:
        """Request targeted snippets using prior execution evidence."""
        with logfire.span("codemode.llm_explore", func_name=func.name, round=2):
            prompt = f"""You previously explored `{func.name}` and the snippets were
executed.  Below are the ACTUAL results and a cluster summary.

Your job now is to find **new behaviour classes** that were NOT covered
in Round 1.  Focus on:
- Syntax / input combinations not yet tried
- Edge cases at boundaries between observed clusters
- Error paths whose exact error type or message differs from expectation
- Interactions between parameters / sub-expressions

<FunctionName>{func.name}</FunctionName>
<FunctionCode>
{func.code}
</FunctionCode>
<Description>{func.description}</Description>

<Round1Results>
{chr(10).join(r.to_context_block() for r in prior_results)}
</Round1Results>

<ClusterSummary>
{cluster_summary}
</ClusterSummary>

RULES:
- Do NOT repeat any snippet from the "Already tried" list.
- Produce 8–12 NEW normal snippets targeting uncovered behaviour.
- Produce 3–5 NEW error snippets targeting untried error paths.
- Same strict rules as before: no try/except, real function name,
  one scenario per snippet, last expression captured."""

            result = await self.explorer_agent.run(prompt)
            plan = result.output

            logfire.info(
                "Round 2: LLM produced {normal} normal + {error} error snippets",
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
    ) -> str | EvalsBundle:
        """Generate a spec from verified exploration results.

        Returns YAML text in default mode, or ``EvalsBundle`` when
        ``use_model_spec=True``.
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
- The top-level YAML key MUST be `{func.name}` (the function name).
- Generate at least {max(len(exploration_results), 5)} diverse test cases.
- Use the EXACT outputs from the execution results above.
- You MUST generate exactly {len(error_results)} raises cases — one for
  each RAISED result above.  The spec is invalid without them.
- Cover normal, edge, and error cases.
- In assertions, use `input` (NOT `inputs`) for accessing input values.

YAML FORMAT — STRICT RULES (violations cause parse failure):
- NEVER use YAML tags: `!!python/tuple`, `!!python/object`, `!!binary`,
  `!!omap`, `!!str`, `!!int`, `!!float`, or ANY `!!` tag whatsoever.
  Plain YAML scalars and sequences only.  `yaml.safe_load()` will be used
  to parse the output — it rejects all `!!` tags and will hard-fail.
- Represent tuples as YAML sequences (lists).
- NEVER emit `!!python/...` or any non-standard YAML type annotation.
{refinement_block}"""

            logfire.info(
                "Sending spec generation prompt",
                func_name=func.name,
                success_results=len(success_results),
                error_results=len(error_results),
            )

            result = await self.spec_agent.run(prompt)

            if self.use_model_spec:
                bundle = result.output
                assert isinstance(bundle, EvalsBundle)
                logfire.info(
                    "Model spec bundle generated",
                    func_name=func.name,
                    eval_count=len(bundle.evals),
                    fixture_count=len(bundle.fixtures),
                )
                return bundle

            yaml_spec = result.output.yaml_spec

            # Sanitize: strip ALL !!<tag> annotations — safe_load only accepts
            # a tiny subset (str/int/float/bool/null/seq/map) and rejects
            # anything else (!!python/tuple, !!binary, !!omap, etc.).
            # Stripping them is safe: scalar values fall back to plain YAML types.
            import re

            yaml_spec = re.sub(r"!![^\s\[\]{},]+", "", yaml_spec)

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
        """Build retry context from failed assertions/errors."""
        return build_failure_context(summary)

    def _inject_durations(
        self,
        yaml_spec: str,
        func: Function,
        *,
        buffer_pct: float = 0.5,
        floor_ms: float = 10.0,
    ) -> str:
        """Inject measured duration thresholds into cases."""
        return inject_durations(
            yaml_spec,
            func,
            self.executor,
            buffer_pct=buffer_pct,
            floor_ms=floor_ms,
        )

    @staticmethod
    def _build_call_code(func_name: str, case: dict) -> str | None:
        """Build a callable expression from a dataset case."""
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
        """Run full CodeMode generation, validation, and optional refinement.

        Flow: explore -> generate spec -> validate -> refine (optional) ->
        inject durations (optional). Returns exploration artifacts, final spec,
        and evaluation summary when ``run_evals`` is enabled.
        """
        with logfire.span(
            "codemode.pipeline",
            func_name=func.name,
            spec_model=self.spec_model,
            exploration_model=self.exploration_model,
            executor=type(self.executor).__name__,
        ):
            t0 = time.perf_counter()

            # Phase 1 — explore (once)
            exploration_results = await self.explore(func)

            # Phase 2–4 — generate spec + validate + refine
            yaml_spec = ""
            generated_bundle: EvalsBundle | None = None
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
                        bundle = await self.generate_spec(
                            func,
                            exploration_results,
                            failure_context,
                        )

                        if isinstance(bundle, EvalsBundle):
                            generated_bundle = bundle
                            yaml_spec = bundle.to_yaml()
                        else:
                            generated_bundle = None
                            yaml_spec = bundle
                    except Exception as exc:
                        logfire.warn(
                            "Spec generation failed on attempt {attempt}, retrying",
                            attempt=attempt + 1,
                            error=str(exc),
                        )
                        failure_context = f"Generation error: {exc}"
                        refinement_rounds = attempt + 1
                        continue

                    if not run_evals:
                        break

                    # Validate: run evals with ignore_duration=True
                    try:
                        if generated_bundle is not None:
                            runner = (
                                RunEvals.from_bundle(generated_bundle)
                                .with_functions({func.name: func.impl})
                                .ignore_duration()
                            )
                        else:
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
                    if generated_bundle is not None:
                        final_runner = (
                            RunEvals.from_bundle(generated_bundle)
                            .with_functions({func.name: func.impl})
                            .ignore_duration()
                        )
                    else:
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
                spec_to_write = materialize_yaml_with_schema_header(yaml_spec)
                with open(path, "w") as f:
                    f.write(spec_to_write)
                logfire.info("Saved spec to {path}", path=path)

            elapsed = (time.perf_counter() - t0) * 1000
            logfire.info(
                "CodeMode pipeline complete in {elapsed:.0f}ms (refinements={refinement_rounds})",
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

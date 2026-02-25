"""TDD-based eval generation: Intent -> Signature -> Evals -> Implementation.

This module provides a true TDD approach where:
1. LLM generates function signature from description (intent)
2. LLM generates eval spec from signature (tests first)
3. LLM generates implementation that passes the evals (code last)

Example:
    from vowel.tdd import TDDGenerator

    generator = TDDGenerator(model="openai:gpt-4o")

    result = generator.generate_all(
        description="Binary search for target in sorted list. Returns index or -1.",
        name="binary_search"
    )

    print(result.signature.to_signature_str())
    print(result.yaml_spec)
    print(result.func.code)
"""

import os
import re
import time
from collections.abc import Callable
from typing import Any, cast

import dotenv
import logfire
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from vowel.context import EVAL_SPEC_CONTEXT
from vowel.eval_types import EvalsSource
from vowel.monitoring import enable_monitoring
from vowel.runner import Function, RunEvals
from vowel.utils import EvalSummary
from vowel.validation import validate_and_fix_spec

# Configure logfire for tracing
dotenv.load_dotenv()

# ── Logfire Monitoring & Observability ──
enable_monitoring(service_name="vowel-tdd")


class Param(BaseModel):
    """A function parameter specification."""

    name: str = Field(description="Parameter name (valid Python identifier)")
    type: str = Field(
        description="Type annotation as string. Supports: "
        "built-in types (int, str, float, bool, list, dict, tuple, set), "
        "typing types (List[T], Dict[K,V], Optional[T], Union[A,B], Any), "
        "union syntax (int | None, str | int), "
        "nested generics (list[dict[str, Any]])"
    )
    default: str | None = Field(
        default=None,
        description="Default value as string (e.g., 'None', '[]', '0'). None means required.",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of this parameter's purpose"
    )

    @property
    def is_required(self) -> bool:
        """Check if parameter is required (no default value)."""
        return self.default is None

    def to_signature_str(self) -> str:
        """Convert to Python signature string."""
        if self.default is not None:
            return f"{self.name}: {self.type} = {self.default}"
        return f"{self.name}: {self.type}"


class FunctionSignature(BaseModel):
    """Complete function signature specification.

    Represents the contract/intent of a function without implementation.
    Used in TDD flow: Description -> Signature -> Evals -> Implementation
    """

    name: str = Field(description="Function name (valid Python identifier)")
    params: list[Param] = Field(default_factory=list, description="Function parameters in order")
    return_type: str = Field(default="Any", description="Return type annotation")
    description: str = Field(description="Detailed description of what the function does")
    is_async: bool = Field(default=False, description="Whether this should be an async function")

    @property
    def required_params(self) -> list[Param]:
        """Get only required parameters."""
        return [p for p in self.params if p.is_required]

    @property
    def optional_params(self) -> list[Param]:
        """Get only optional parameters (with defaults)."""
        return [p for p in self.params if not p.is_required]

    def to_signature_str(self) -> str:
        """Generate Python function signature string."""
        params_str = ", ".join(p.to_signature_str() for p in self.params)
        prefix = "async def" if self.is_async else "def"
        return f"{prefix} {self.name}({params_str}) -> {self.return_type}"

    def to_stub(self) -> str:
        """Generate a function stub with docstring but no implementation."""
        lines = [f"{self.to_signature_str()}:"]
        doc_lines = [f'    """{self.description}']

        if self.params:
            doc_lines.append("")
            doc_lines.append("    Args:")
            for p in self.params:
                desc = p.description or "No description"
                if p.default is not None:
                    doc_lines.append(f"        {p.name}: {desc} (default: {p.default})")
                else:
                    doc_lines.append(f"        {p.name}: {desc}")

        doc_lines.append("")
        doc_lines.append("    Returns:")
        doc_lines.append(f"        {self.return_type}: Result value")

        doc_lines.append('    """')
        lines.extend(doc_lines)
        lines.append("    ...")

        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompts."""
        lines = [
            f"Function: {self.name}",
            f"Signature: {self.to_signature_str()}",
            f"Description: {self.description}",
            "",
            "Parameters:",
        ]

        for p in self.params:
            req = "required" if p.is_required else f"optional, default={p.default}"
            desc = p.description or "No description"
            lines.append(f"  - {p.name} ({p.type}, {req}): {desc}")

        lines.append("")
        lines.append(f"Returns: {self.return_type}")

        return "\n".join(lines)


class TDDResult(BaseModel):
    """Result of TDD generation flow."""

    signature: FunctionSignature = Field(description="Generated function signature")
    yaml_spec: str = Field(description="Generated eval specification")
    func: Function = Field(description="Generated function implementation")
    summary: EvalSummary = Field(description="Evaluation results")

    model_config = {"arbitrary_types_allowed": True}

    def print(self) -> None:
        """Pretty print the TDD result."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax

            console = Console()

            console.print("\n[bold cyan]═══ TDD Generation Result ═══[/]\n")

            console.print("[yellow]Step 1: Signature (Intent)[/]")
            console.print(Panel(self.signature.to_signature_str(), title="Function Signature"))

            console.print("\n[yellow]Step 2: Evals (Tests)[/]")
            console.print(
                Panel(
                    Syntax(self.yaml_spec, "yaml", theme="monokai", line_numbers=True),
                    title="Eval Specification",
                )
            )

            console.print("\n[yellow]Step 3: Implementation (Code)[/]")
            console.print(
                Panel(
                    Syntax(self.func.code, "python", theme="monokai", line_numbers=True),
                    title="Generated Function",
                )
            )

            status = "[green]✅ ALL PASSED[/]" if self.summary.all_passed else "[red]❌ FAILURES[/]"
            console.print(
                f"\n[yellow]Result:[/] {status} ({self.summary.coverage * 100:.0f}% coverage)"
            )

            # Print failure reasons if any
            if not self.summary.all_passed:
                console.print("\n[yellow]Failure Details:[/]")
                self.summary.print(include_reasons=True)

        except ImportError:
            print("\n=== TDD Generation Result ===")
            print(f"\nSignature: {self.signature.to_signature_str()}")
            print(f"\nYAML Spec:\n{self.yaml_spec}")
            print(f"\nImplementation:\n{self.func.code}")
            print(f"\nStatus: {'PASSED' if self.summary.all_passed else 'FAILED'}")


class TDDGenerator:
    """TDD-based function generator.

    Flow: Description -> Signature -> Evals -> Implementation
    """

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
            raise ValueError("Model name must be provided or set via MODEL_NAME env var")

        # Format additional context for injection into agent system prompts
        if additional_context is None:
            self._additional_context = ""
        elif isinstance(additional_context, str):
            self._additional_context = additional_context
        else:
            self._additional_context = "\n\n".join(
                f"<ContextPart>\n{d}\n</ContextPart>" for d in additional_context
            )

        self._eval_agent: Any = None
        self._impl_agent: Any = None
        self._signature_agent: Any = None

        logfire.info("TDDGenerator initialized", model=self.model)

    @property
    def signature_agent(self) -> Agent[None, FunctionSignature]:
        """Agent for generating function signatures."""
        if self._signature_agent is None:
            self._signature_agent = Agent(
                self.model,
                output_type=FunctionSignature,
                system_prompt="""You are an expert Python API designer.
Your task is to design clean, well-typed function signatures from descriptions.

Guidelines:
- Use modern Python type hints (list[int], dict[str, Any], int | None)
- Choose descriptive parameter names
- Include all necessary parameters based on the description
- Specify return type accurately
- Write a clear, complete description
""",
            )
        return cast(Agent[None, FunctionSignature], self._signature_agent)

    @property
    def eval_agent(self) -> Agent[None, EvalsSource]:
        """Agent for generating eval specs from signatures."""
        if self._eval_agent is None:
            ctx_block = (
                f"\n\n<AdditionalContext>\n{self._additional_context}\n</AdditionalContext>"
                if self._additional_context
                else ""
            )
            self._eval_agent = Agent(
                self.model,
                output_type=EvalsSource,
                system_prompt=f"""You are an expert test case generator.
Your task is to generate comprehensive eval specs from function signatures.

{EVAL_SPEC_CONTEXT}{ctx_block}

═══════════════════════════════════════════════════════════════════════════
CRITICAL RULES - READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════

## 1. INPUT FORMAT - ALWAYS USE INLINE LIST FORMAT

ALWAYS use `inputs:` with an INLINE LIST `[arg1, arg2]`, NEVER YAML list syntax with dashes.

✅ CORRECT (inline list on same line):
```yaml
inputs: [{{"a": 1, "b": 2}}, "a.b"]
inputs: ["hello world", true]
inputs: [[1, 2, 3], 5]
```

❌ WRONG (YAML list with dashes - breaks parsing):
```yaml
inputs:
  - {{"a": 1}}
  - "path"
```

❌ WRONG (named dict):
```yaml
inputs:
  data: {{"a": 1}}
  path: "a"
```

For single argument, use `input:` (singular):
```yaml
input: "2 + 3 * 4"
input: [1, 2, 3, 4, 5]
```

## 2. ASSERTION VARIABLES

In assertions, access inputs positionally:
- `input` - the raw input (for single `input:` field)
- `input[0]`, `input[1]` - positional args (for `inputs:` list)
- `output` - function return value
- `expected` - expected value if specified

## 3. EXPECTED VALUES - CALCULATE CAREFULLY!

⚠️ DO NOT GUESS expected values! Trace through the algorithm mentally.

For algorithms like Levenshtein distance, sorting, math expressions:
- Step through the algorithm manually
- Verify your calculation is correct
- If unsure, use `assertion` instead of `expected`

Example - Levenshtein("kitten", "sitting"):
- k→s (substitute): 1
- i→i (match): 0
- t→t (match): 0
- t→t (match): 0
- e→i (substitute): 1
- n→n (match): 0
- ""→g (insert): 1
- Total: 3 ✓

## 4. RAISES - USE FOR CLEAR ERROR CASES

Use `raises:` for **unambiguous** error scenarios:

✅ GOOD raises cases (clear errors):
```yaml
- input: "10 / 0"                    # Division by zero - always fails
  raises: ZeroDivisionError

- inputs: [123, "string"]            # Wrong type - clear TypeError
  raises: TypeError

- input: "(2 + 3"                    # Unbalanced parens - obvious syntax error
  raises: ValueError
```

⚠️ AVOID raises for ambiguous cases:
```yaml
# DON'T - "2 ++ 3" is VALID in Python (equals 2 + (+3) = 5)
- input: "2 ++ 3"
  raises: ValueError  # WRONG! This won't raise

# Instead, test the actual behavior:
- input: "2 ++ 3"
  expected: 5
```

When using `match:`, keep it SHORT:
```yaml
raises: ValueError
match: "invalid"           # ✅ Simple keyword, not full message
```

## 5. RETURN TYPE EVALUATOR - USE SIMPLE TYPES ONLY!

For `type:` evaluator, use ONLY built-in Python types:

✅ GOOD:
```yaml
evals:
  ReturnType:
    type: int
  # or
  ReturnType:
    type: str
  # or union of built-ins
  ReturnType:
    type: "int | str"
```

❌ BAD (will fail to parse - framework doesn't support these):
```yaml
evals:
  ReturnType:
    type: "Any | None"     # Any is NOT recognized!
  ReturnType:
    type: "Optional[str]"  # Generic types NOT supported
  ReturnType:
    type: "List[int]"      # Use 'list' instead
```

**If return type includes `Any` or `None`, SKIP the type evaluator entirely!**
Don't add a ReturnType evaluator for functions returning `Any | None`.

## 6. YAML SYNTAX - NO PYTHON EXPRESSIONS

YAML only accepts LITERAL values:

❌ FORBIDDEN:
- `[i for i in range(100)]` - list comprehension
- `"a" * 5` - string multiplication
- `range(10)` - function call

✅ CORRECT:
- `[0, 1, 2, 3, 4, 5]` - literal list
- `"aaaaa"` - literal string

## 6. ASSERTIONS - SINGLE LINE, NO LINE BREAKS!

Assertions can be as long as needed, but MUST be on a SINGLE LINE.
DO NOT insert newlines or line breaks inside assertion strings.

✅ CORRECT (single line, even if long):
```yaml
assertion: "len(output) > 0 if input[0].strip() else len(output) == 0"
assertion: "all(isinstance(x, str) for x in output) and len(output) >= 0"
```

❌ WRONG (line break inside string):
```yaml
assertion: "len(output) > 0 if input[0].strip()
            else len(output) == 0"
```

## 7. GLOBAL EVALUATORS - KEEP SIMPLE!

Global evaluators (under `evals:`) apply to ALL test cases.
Keep them simple and universally applicable.

✅ GOOD global evaluators:
```yaml
evals:
  ReturnType:
    type: int
  NonNegative:
    assertion: "output >= 0"
```

❌ BAD global evaluators (too complex, may fail on some cases):
```yaml
evals:
  ComplexCheck:
    assertion: "output == input[2] if len(input) > 2 else True"  # DON'T!
```

For complex validations, use case-specific assertions instead.
""",
            )
        return cast(Agent[None, EvalsSource], self._eval_agent)

    @property
    def impl_agent(self) -> Agent[None, Function]:
        """Agent for generating implementations."""
        if self._impl_agent is None:
            ctx_block = (
                f"\n\n<AdditionalContext>\n{self._additional_context}\n</AdditionalContext>"
                if self._additional_context
                else ""
            )
            self._impl_agent = Agent(
                self.model,
                output_type=Function,
                system_prompt=f"""You are an expert Python engineer.
Your task is to implement functions that PASS ALL test cases in the eval spec.

{EVAL_SPEC_CONTEXT}{ctx_block}

═══════════════════════════════════════════════════════════════════════════
CRITICAL - UNDERSTANDING TEST CASES
═══════════════════════════════════════════════════════════════════════════

## 1. INPUT FORMATS

Test cases use two input formats:

**Single argument** (`input:` singular):
```yaml
input: "2 + 3 * 4"
```
→ Function called as: `func("2 + 3 * 4")`

**Multiple arguments** (`inputs:` plural, as list):
```yaml
inputs: [{{"a": 1}}, "a.b"]
```
→ Function called as: `func({{"a": 1}}, "a.b")`

## 2. EXPECTED FIELD

The function MUST return EXACTLY this value:
```yaml
inputs: [[1, 3, 5, 7], 5]
expected: 2
```
→ `func([1, 3, 5, 7], 5)` MUST return `2`

## 3. RAISES FIELD - CRITICAL!

The function MUST RAISE this exception for the given inputs:
```yaml
inputs: [[1, 2, 3], "a"]
raises: TypeError
```
→ `func([1, 2, 3], "a")` MUST raise `TypeError`!

**This is NOT optional!** You must:
1. ADD explicit type/value checks at the start of the function
2. RAISE the exception when invalid input is detected
3. DO NOT make the function "tolerant" or use Union types to accept bad input

Example implementation pattern:
```python
def my_func(data: list, target: int) -> int:
    # EXPLICIT validation for raises cases
    if not isinstance(target, int):
        raise TypeError("target must be an integer")
    # ... rest of implementation
```

## 4. MATCH FIELD

When `match:` is specified with `raises:`, the error message must CONTAIN that substring:
```yaml
raises: ValueError
match: "invalid"
```
→ Error message must contain "invalid" (case-sensitive)

Make your error messages include these keywords:
```python
raise ValueError("invalid expression syntax")  # Contains "invalid" ✓
```

## 5. ASSERTION FIELD

A Python expression that must evaluate to True:
```yaml
assertion: "output >= 0 or output == -1"
```

## 6. CODE QUALITY - MUST BE VALID PYTHON!

⚠️ Your code MUST be syntactically valid Python. Common mistakes to avoid:

❌ WRONG (unclosed strings):
```python
punct = r",;! \\\\(\\\\)    # Missing closing quote!
```

✅ CORRECT:
```python
punct = r",;!\\\\(\\\\)"  # Properly closed
```

❌ WRONG (unescaped quotes in strings):
```python
msg = "It's a "test""  # Broken!
```

✅ CORRECT:
```python
msg = "It's a \"test\""  # Properly escaped
msg = 'It\'s a "test"'   # Or use different quotes
```

## 7. ALGORITHM IMPLEMENTATION - THINK STEP BY STEP!

Before writing code, analyze EACH test case and trace what the function should do:

**Example: Path extractor with `inputs: [{{"users": [{{"id": 1, "meta": {{"role": "admin"}}}}]}}, "users[0].meta.role"]`**

1. Parse path: "users[0].meta.role" → segments: ["users", "[0]", "meta", "role"]
2. Start with data: {{"users": [{{"id": 1, "meta": {{"role": "admin"}}}}]}}
3. Step "users" → [{{"id": 1, "meta": {{"role": "admin"}}}}]
4. Step "[0]" → {{"id": 1, "meta": {{"role": "admin"}}}}
5. Step "meta" → {{"role": "admin"}}
6. Step "role" → "admin" ✓

**For path/string parsing:**
- Use `re.findall()` or `re.split()` for complex patterns
- Handle both dict keys (strings) and list indices (integers)
- Test your regex pattern mentally against ALL test cases

**For recursive/nested structures:**
- Handle both dict and list at each level
- Return None for missing keys or out-of-bounds indices
- Don't assume structure - check types at each step

## 8. TEST EACH CASE MENTALLY

Before submitting, trace through EVERY test case:

```
Test case: inputs: [{{"a": {{"b": 1}}}}, "a.c"]
Expected: null (None)

Trace:
1. data = {{"a": {{"b": 1}}}}
2. path = "a.c" → segments = ["a", "c"]
3. data["a"] = {{"b": 1}} ✓
4. data["c"] = KeyError → return None ✓
```

If ANY test case would fail, fix your implementation BEFORE returning.

## IMPLEMENTATION CHECKLIST:
- [ ] Code is syntactically valid Python (no SyntaxError)
- [ ] Follow the exact signature provided
- [ ] Add explicit type checks for ALL `raises:` test cases
- [ ] Include match keywords in error messages
- [ ] TRACE through EACH test case to verify it passes
- [ ] Handle edge cases (empty input, None, etc.)
- [ ] For path parsing: handle both `key` and `[index]` formats
- [ ] For nested access: check type at EACH level before accessing
""",
            )
        return cast(Agent[None, Function], self._impl_agent)

    def generate_signature(
        self,
        description: str,
        name: str,
        is_async: bool = False,
    ) -> FunctionSignature:
        """Generate function signature from description.

        Args:
            description: What the function should do
            name: Function name
            is_async: Whether to generate async function

        Returns:
            FunctionSignature with params, return type, etc.
        """
        with logfire.span("Generating signature", name=name):
            prompt = f"""Design a Python function signature for:

Function name: {name}
Description: {description}
Async: {is_async}

Generate a complete FunctionSignature with:
- Appropriate parameter names and types
- Clear parameter descriptions
- Correct return type
- Any exceptions it might raise
"""
            result = self.signature_agent.run_sync(prompt, output_type=FunctionSignature)
            sig = result.output
            sig.name = name  # Ensure name matches
            sig.is_async = is_async

            logfire.info("Signature generated", signature=sig.to_signature_str())
            return sig

    def generate_evals_from_signature(
        self,
        signature: FunctionSignature,
        min_cases: int = 5,
        func: Callable | Function | None = None,
        max_retries: int = 0,
        min_coverage: float = 1.0,
        retry_delay: float = 2.0,
        ignore_duration: bool = True,
        additional_context: str = "",
        description: str = "",
    ) -> tuple[RunEvals, str]:
        """Step 2: Generate eval spec from signature.

        When ``func`` is provided the generated spec is executed and, if
        coverage is below ``min_coverage``, the spec is regenerated up to
        ``max_retries`` times with failure context injected into the prompt.

        Args:
            signature: Function signature to generate tests for
            min_cases: Minimum number of test cases
            func: Optional callable to validate the spec against
            max_retries: How many times to regenerate on low coverage (0 = single attempt)
            min_coverage: Target pass-rate in 0.0-1.0 (default 1.0 = 100%)
            retry_delay: Seconds to wait between retries
            ignore_duration: Ignore duration evaluator when running validation
            additional_context: Extra context to include in the prompt
            description: Original intent/description of the function (improves test quality)

        Returns:
            Tuple of (RunEvals runner, yaml_spec string)
        """
        last_failure_context: str | None = None
        summary: EvalSummary | None = None
        runner: RunEvals | None = None
        yaml_spec: str = ""

        for attempt in range(max_retries + 1):
            with logfire.span(
                "Generating evals from signature", name=signature.name, attempt=attempt + 1
            ):
                extra_context = ""
                if last_failure_context:
                    extra_context = f"""

⚠️ PREVIOUS ATTEMPT FAILED — fix these issues:
{last_failure_context}

Regenerate the YAML spec addressing every failure above.
"""

                prompt = f"""Generate eval YAML spec for this function signature:

{signature.to_prompt_context()}
{f"Original intent: {description}" if description else ""}

Requirements:
- Use `{signature.name}` as eval_id
- Generate at least {min_cases} diverse test cases
- Include normal cases, edge cases, and error cases
- Test all parameters and return type
- Add appropriate global evaluators (type checks, assertions)

IMPORTANT: In assertions, use `input[0]`, `input[1]` to access positional args.
{extra_context}
{f"<UserContext>{additional_context}</UserContext>" if additional_context else ""}"""
                result = self.eval_agent.run_sync(prompt)
                yaml_spec = result.output.yaml_spec  # type: ignore[attr-defined]

                # Sanitize: strip YAML tags that safe_load rejects
                yaml_spec = re.sub(r"!!python/[\w.:]+", "", yaml_spec)
                yaml_spec = re.sub(r"!!binary\b", "", yaml_spec)

                # Validate YAML syntax
                yaml.safe_load(yaml_spec)

                # Static validation: fix common LLM generation mistakes
                validation = validate_and_fix_spec(yaml_spec)
                if validation.has_warnings:
                    logfire.info("Spec validation results", summary=validation.summary())
                if validation.was_modified:
                    yaml_spec = validation.fixed_yaml

                runner = RunEvals.from_source(yaml_spec)
                logfire.info(
                    "Evals generated", cases=len(yaml_spec.split("- case:")), attempt=attempt + 1
                )

                # If no func provided, return without validation
                if func is None:
                    return runner, yaml_spec

                # Run spec against the provided function
                test_runner = runner.with_functions({signature.name: func})
                if ignore_duration:
                    test_runner = test_runner.ignore_duration()
                summary = test_runner.run()

                if summary.coverage >= min_coverage:
                    logfire.info(
                        "Eval spec meets coverage",
                        coverage=f"{summary.coverage * 100:.0f}%",
                        attempt=attempt + 1,
                    )
                    return runner, yaml_spec

                # Build failure context for next attempt
                last_failure_context = self._build_eval_failure_context(summary)
                logfire.warn(
                    "Eval spec below coverage, retrying",
                    coverage=f"{summary.coverage * 100:.0f}%",
                    target=f"{min_coverage * 100:.0f}%",
                    attempt=attempt + 1,
                )

                if attempt < max_retries:
                    time.sleep(retry_delay)

        # Exhausted retries — return last generated spec
        # (summary/runner/yaml_spec are always set when func is not None and loop ran at least once)
        assert summary is not None and runner is not None  # noqa: S101
        logfire.warn(
            "Eval generation exhausted retries",
            final_coverage=f"{summary.coverage * 100:.0f}%",
            target=f"{min_coverage * 100:.0f}%",
        )
        return runner, yaml_spec

    def _build_eval_failure_context(self, summary: EvalSummary) -> str:
        """Build a concise failure report to inject into the retry prompt."""
        lines: list[str] = []
        for result in summary.results:
            if result.report:
                for case in result.report.cases:
                    failed_assertions = {k: v for k, v in case.assertions.items() if not v.value}
                    if failed_assertions:
                        reasons = ", ".join(
                            f"{k}: {v.reason}" for k, v in failed_assertions.items() if v.reason
                        )
                        lines.append(f"- Case '{case.name}' FAILED [{reasons}]")
            if result.error:
                lines.append(f"- Error: {result.error}")
        return "\n".join(lines) if lines else "Unknown failures"

    def generate_implementation(
        self,
        signature: FunctionSignature,
        yaml_spec: str,
        additional_context: str = "",
        description: str = "",
    ) -> Function:
        """Step 3: Generate implementation that passes the evals.

        Args:
            signature: Function signature to implement
            yaml_spec: Eval spec the implementation must pass
            additional_context: Extra context to include in the prompt
            description: Original intent/description of the function (improves implementation quality)

        Returns:
            Function with implementation code
        """
        with logfire.span("Generating implementation", name=signature.name):
            prompt = f"""Implement this function to pass all test cases:

{f"Intent: {description}" if description else ""}
{signature.to_stub()}

Test cases (must pass all):
```yaml
{yaml_spec}
```

Requirements:
- Follow the exact signature: {signature.to_signature_str()}
- Make sure ALL test cases pass
- Handle edge cases properly
- {"Use async/await syntax" if signature.is_async else "Use regular def syntax"}
{f"<UserContext>{additional_context}</UserContext>" if additional_context else ""}"""
            result = self.impl_agent.run_sync(prompt)
            func = result.output  # type: ignore[assignment]
            func.name = signature.name  # type: ignore[attr-defined]
            func.description = signature.description  # type: ignore[attr-defined]
            _ = func.impl  # type: ignore[attr-defined]  # Compile the function

            logfire.info("Implementation generated", name=func.name)  # type: ignore[attr-defined]
            return func  # type: ignore[return-value]

    def generate_all(
        self,
        description: str,
        name: str,
        is_async: bool = False,
        min_cases: int = 5,
        max_impl_retries: int = 2,
        max_eval_retries: int = 0,
        max_flow_retries: int = 0,
        retry_delay: float = 3.0,
        ignore_duration: bool = True,
        min_coverage: float = 1.0,
        additional_context: str = "",
    ) -> TDDResult:
        """Run complete TDD flow: Signature -> Evals -> Implementation.

        Args:
            description: What the function should do
            name: Function name
            is_async: Whether to generate async function
            min_cases: Minimum test cases to generate
            max_impl_retries: Max retries for implementation (per flow attempt)
            max_eval_retries: Max retries for eval spec generation (validates against impl)
            max_flow_retries: Max retries for entire flow (regenerate evals if all impl attempts fail)
            retry_delay: Delay between retries
            ignore_duration: Ignore duration constraints
            min_coverage: Target coverage for eval validation (0.0-1.0)
            additional_context: Extra context to pass to eval and impl generation prompts

        Returns:
            TDDResult with signature, yaml_spec, func, and summary
        """

        # Step 1: Generate signature once (not regenerated on flow retries)
        logfire.info("Step 1: Generating signature")
        signature = self.generate_signature(description, name, is_async)

        for flow_attempt in range(max_flow_retries + 1):
            with logfire.span("TDD generation flow", name=name, flow_attempt=flow_attempt + 1):
                # Step 2: Generate evals
                logfire.info("Step 2: Generating evals", flow_attempt=flow_attempt + 1)
                runner, yaml_spec = self.generate_evals_from_signature(
                    signature,
                    min_cases,
                    additional_context=additional_context,
                    description=description,
                )

                # Step 3: Generate implementation (with retries)
                logfire.info("Step 3: Generating implementation")

                summary: EvalSummary | None = None
                for impl_attempt in range(max_impl_retries + 1):
                    func = self.generate_implementation(
                        signature, yaml_spec, additional_context, description
                    )

                    # If max_eval_retries > 0, re-validate evals against this impl
                    if max_eval_retries > 0 and impl_attempt == 0:
                        runner, yaml_spec = self.generate_evals_from_signature(
                            signature,
                            min_cases,
                            func=func.impl,
                            max_retries=max_eval_retries,
                            min_coverage=min_coverage,
                            retry_delay=retry_delay,
                            ignore_duration=ignore_duration,
                            additional_context=additional_context,
                            description=description,
                        )

                    # Run evals
                    test_runner = runner.with_functions({name: func.impl})
                    if ignore_duration:
                        test_runner = test_runner.ignore_duration()
                    summary = test_runner.run()

                    if summary.coverage >= min_coverage:
                        logfire.info(
                            "Implementation passed tests",
                            flow_attempt=flow_attempt + 1,
                            impl_attempt=impl_attempt + 1,
                            coverage=f"{summary.coverage * 100:.0f}%",
                        )
                        return TDDResult(
                            signature=signature,
                            yaml_spec=yaml_spec,
                            func=func,
                            summary=summary,
                        )

                    if impl_attempt < max_impl_retries:
                        logfire.warn(
                            "Implementation failed tests, retrying",
                            impl_attempt=impl_attempt + 1,
                            coverage=f"{summary.coverage * 100:.0f}%",
                        )
                        time.sleep(retry_delay)

                        # Add failure context for next attempt
                        yaml_spec = self._add_failure_context(yaml_spec, summary)

                # All impl attempts failed, retry entire flow if allowed
                if flow_attempt < max_flow_retries:
                    assert summary is not None  # noqa: S101  # always set after inner loop ran
                    logfire.warn(
                        "All implementation attempts failed, regenerating evals",
                        flow_attempt=flow_attempt + 1,
                        coverage=f"{summary.coverage * 100:.0f}%",
                    )
                    time.sleep(retry_delay)

        print("TDD Generation flow failed.")
        raise SystemExit(1)

    def _add_failure_context(self, yaml_spec: str, summary: EvalSummary) -> str:
        """Add failure information as YAML comments for retry."""
        failed_info = []
        for result in summary.results:
            if result.report:
                for case in result.report.cases:
                    if not all(a.value for a in case.assertions.values()):
                        failed_info.append(f"# FAILED: {case.name}")

        if failed_info:
            return "\n".join(failed_info) + "\n" + yaml_spec
        return yaml_spec

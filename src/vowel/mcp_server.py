"""Vowel MCP Server - Model Context Protocol server for eval generation.

This module exposes vowel's full evaluation, generation, and TDD capabilities via
MCP (Model Context Protocol), enabling AI assistants to run evaluations, generate
functions, create test specs, and perform TDD workflows.

Configuration is set via the ``env`` field in your MCP client JSON config.
The env field should contain API keys and model names only. All other parameters
(auto_retry, min_coverage, etc.) are tool parameters with sensible defaults.

Usage:
    # Add to MCP client config (e.g., Claude Desktop, VS Code Copilot)
    {
        "mcpServers": {
            "vowel": {
                "command": "python",
                "args": ["-m", "vowel.mcp_server"],
                "env": {
                    "MODEL_NAME": "openai:gpt-4o",
                    "OPENAI_API_KEY": "sk-..."
                }
            }
        }
    }

    # Or run directly (reads env vars from shell)
    python -m vowel.mcp_server

Supported env vars:
    MODEL_NAME          — Default LLM model (e.g. "openai:gpt-4o", "gemini-3-flash-preview")
    JUDGE_MODEL         — Model for LLM Judge evaluator
    OPENAI_API_KEY      — OpenAI API key
    ANTHROPIC_API_KEY   — Anthropic API key
    GOOGLE_API_KEY      — Google AI API key

Available Tools (14):
    Eval Runner:
        - run_evals_from_file: Run evaluations from a YAML file
        - run_evals_from_yaml: Run evaluations from YAML content string
        - run_evals_with_fixtures: Run evaluations with fixture injection
        - validate_yaml_spec: Validate a YAML eval specification
        - check_function_compatibility: Check function compatibility with eval generation
        - list_yaml_files: List YAML files in a directory

    EvalGenerator:
        - generate_function: Generate a Python function from description
        - generate_eval_spec: Generate eval spec for a function
        - generate_and_run_evals: Generate spec + run + auto-retry + heal

    TDDGenerator:
        - tdd_generate_signature: Generate function signature from description
        - tdd_generate_evals: Generate eval spec from a signature
        - tdd_generate_implementation: Generate implementation from signature + spec
        - tdd_generate_all: Full TDD flow: description → signature → evals → implementation
        - tdd_generate_and_validate: TDD with eval validation against implementation

Available Resources:
    - vowel://context: Eval specification documentation
    - vowel://example: Example YAML eval specification
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nest_asyncio
from mcp.server.fastmcp import FastMCP

from vowel import check_compatibility, load_evals_from_yaml_string, run_evals
from vowel.ai import EVAL_SPEC_CONTEXT, EvalGenerator
from vowel.monitoring import enable_monitoring
from vowel.runner import Function, RunEvals
from vowel.tdd import TDDGenerator

enable_monitoring(service_name="vowel-mcp")

nest_asyncio.apply()

# ── Helpers ──


def _exec_functions(functions: dict[str, str]) -> dict[str, Any]:
    """Execute function code strings and return name→callable mapping."""
    func_objects: dict[str, Any] = {}
    for name, code in functions.items():
        local_ns: dict[str, Any] = {}
        exec(code, {}, local_ns)  # noqa: S102
        if name in local_ns:
            func_objects[name] = local_ns[name]
    return func_objects


# ══════════════════════════════════════════════════════════════════════
#  MCP Server
# ══════════════════════════════════════════════════════════════════════

mcp = FastMCP("vowel")


# ── Eval Runner Tools ──────────────────────────────────────────────


@mcp.tool()
def run_evals_from_file(
    yaml_path: str,
    filter_funcs: list[str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run evaluations from a YAML file.

    Args:
        yaml_path: Path to the YAML eval specification file
        filter_funcs: Optional list of function names to filter
        debug: Enable debug mode with stack traces
    """
    path = Path(yaml_path)
    if not path.exists():
        return {"error": f"File not found: {yaml_path}"}
    try:
        summary = run_evals(path, filter_funcs=filter_funcs, debug=debug)
        return _summary_to_dict(summary)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_evals_from_yaml(
    yaml_content: str,
    functions: dict[str, str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run evaluations from inline YAML content.

    ⚠️ Security: function code values are executed via exec() — trusted environments only.

    Args:
        yaml_content: YAML eval specification as string
        functions: Optional dict mapping function names to their Python source code
        debug: Enable debug mode
    """
    try:
        runner = RunEvals.from_source(yaml_content)
        if functions:
            runner = runner.with_functions(_exec_functions(functions))
        if debug:
            runner = runner.debug()
        summary = runner.run()
        return _summary_to_dict(summary)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_evals_with_fixtures(
    yaml_content: str,
    functions: dict[str, str] | None = None,
    fixtures: dict[str, str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run evaluations with fixture injection.

    Fixtures are injected as keyword-only arguments to test functions.

    Args:
        yaml_content: YAML eval specification as string
        functions: Dict mapping function names to their Python source code
        fixtures: Dict mapping fixture names to setup code (must define ``setup()`` function)
        debug: Enable debug mode
    """
    try:
        runner = RunEvals.from_source(yaml_content)
        if functions:
            runner = runner.with_functions(_exec_functions(functions))
        if fixtures:
            fixture_objects: dict[str, Any] = {}
            for name, code in fixtures.items():
                local_ns: dict[str, Any] = {}
                exec(code, {}, local_ns)  # noqa: S102
                if "setup" in local_ns:
                    setup_fn = local_ns["setup"]
                    teardown_fn = local_ns.get("teardown")
                    fixture_objects[name] = (setup_fn, teardown_fn) if teardown_fn else setup_fn
                else:
                    for obj in local_ns.values():
                        if callable(obj) and not isinstance(obj, type):
                            fixture_objects[name] = obj
                            break
            if fixture_objects:
                runner = runner.with_fixtures(fixture_objects)
        if debug:
            runner = runner.debug()
        summary = runner.run()
        return _summary_to_dict(summary)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def validate_yaml_spec(yaml_content: str) -> dict[str, Any]:
    """Validate a YAML eval specification without running it.

    Args:
        yaml_content: YAML eval specification to validate
    """
    try:
        evals = load_evals_from_yaml_string(yaml_content)
        function_names = list(evals.keys())
        return {
            "valid": True,
            "functions": function_names,
            "count": len(function_names),
            "message": f"Valid spec with {len(function_names)} function(s)",
        }
    except Exception as e:
        return {"valid": False, "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
def check_function_compatibility(function_code: str, function_name: str) -> dict[str, Any]:
    """Check if a function is compatible with vowel eval generation.

    Verifies that all parameter types can be serialized to YAML.

    Args:
        function_code: The Python function source code
        function_name: Name of the function in the code
    """
    try:
        local_ns: dict[str, Any] = {}
        exec(function_code, {}, local_ns)  # noqa: S102
        func = local_ns.get(function_name)
        if not func:
            return {"error": f"Function '{function_name}' not found in code"}
        is_compatible, unsupported = check_compatibility(func)
        return {
            "compatible": is_compatible,
            "function_name": function_name,
            "unsupported_params": unsupported,
            "message": (
                "Function is compatible"
                if is_compatible
                else f"Unsupported parameters: {', '.join(unsupported)}"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_yaml_files(directory: str, recursive: bool = True) -> dict[str, Any]:
    """List all YAML eval files in a directory.

    Args:
        directory: Directory path to search
        recursive: Search subdirectories (default: true)
    """
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}
    yaml_files: list[Path] = []
    for pattern in ("*.yml", "*.yaml"):
        yaml_files.extend(path.rglob(pattern) if recursive else path.glob(pattern))
    return {
        "directory": str(path.absolute()),
        "recursive": recursive,
        "files": [str(f) for f in sorted(yaml_files)],
        "count": len(yaml_files),
    }


# ── EvalGenerator Tools ───────────────────────────────────────────


@mcp.tool()
def generate_function(
    description: str,
    model: str | None = None,
    async_func: bool = False,
    additional_context: str | None = None,
) -> dict[str, Any]:
    """Generate a Python function from a natural language description using LLM.

    Args:
        description: What the function should do (natural language)
        model: LLM model identifier (defaults to MODEL_NAME env var)
        async_func: Whether to generate an async function
        additional_context: Extra context for the generation prompt
    """
    try:
        gen = EvalGenerator(model=model, additional_context=additional_context, load_env=True)
        func = gen.generate_function(description, async_func=async_func)
        return {"name": func.name, "description": func.description, "code": func.code}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def generate_eval_spec(
    function_code: str,
    function_name: str,
    function_description: str = "",
    model: str | None = None,
    additional_context: str = "",
    save_to_file: bool = False,
    retries: int = 5,
) -> dict[str, Any]:
    """Generate a YAML eval specification for an existing function.

    Creates comprehensive test cases covering normal, edge, and error scenarios.

    Args:
        function_code: The full Python source code of the function
        function_name: Name of the function to test
        function_description: Optional description of what the function does
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation (e.g. "Focus on edge cases")
        save_to_file: Whether to save the spec as {function_name}_evals.yml
        retries: Max generation retries on YAML parse failure (default: 5)
    """
    try:
        gen = EvalGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )

        local_ns: dict[str, Any] = {}
        exec(function_code, {}, local_ns)  # noqa: S102
        func_callable = local_ns.get(function_name)

        func = Function(
            name=function_name,
            description=function_description or f"Function {function_name}",
            code=function_code,
            func=func_callable,
        )

        _, yaml_spec = gen.generate_spec(
            func,
            additional_context=additional_context,
            save_to_file=save_to_file,
            retries=retries,
        )
        return {
            "function_name": function_name,
            "yaml_spec": yaml_spec,
            "message": "Eval spec generated. Use run_evals_from_yaml to execute.",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def generate_and_run_evals(
    function_code: str,
    function_name: str,
    function_description: str = "",
    model: str | None = None,
    additional_context: str = "",
    auto_retry: bool = True,
    retry_delay: float = 5.0,
    max_retries: int = 3,
    min_coverage: float = 0.9,
    heal_function: bool = True,
    ignore_duration: bool = True,
) -> dict[str, Any]:
    """Generate eval spec for a function, run tests, auto-retry, and optionally heal the function.

    This is the full EvalGenerator workflow:
    1. Generate YAML eval spec from function code
    2. Run the generated tests
    3. If failures: retry with error context (when auto_retry=true)
    4. If persistent failures + heal_function=true: LLM fixes the function code
    5. Repeat until min_coverage met or max_retries exhausted

    Args:
        function_code: The full Python source code of the function
        function_name: Name of the function to test
        function_description: Optional description of the function
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation
        auto_retry: Retry on failures (default: true)
        retry_delay: Seconds between retries (default: 5.0)
        max_retries: Maximum retry attempts (default: 3)
        min_coverage: Target pass rate 0.0-1.0 (default: 0.9)
        heal_function: Auto-fix buggy function code on failures (default: true)
        ignore_duration: Skip duration evaluator checks (default: true)
    """
    try:
        gen = EvalGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )

        local_ns: dict[str, Any] = {}
        exec(function_code, {}, local_ns)  # noqa: S102
        func_callable = local_ns.get(function_name)

        func = Function(
            name=function_name,
            description=function_description or f"Function {function_name}",
            code=function_code,
            func=func_callable,
        )

        result = gen.generate_and_run(
            func,
            additional_context=additional_context,
            auto_retry=auto_retry,
            retry_delay=retry_delay,
            max_retries=max_retries,
            min_coverage=min_coverage,
            heal_function=heal_function,
            ignore_duration=ignore_duration,
        )

        return {
            "yaml_spec": result.yaml_spec,
            "was_healed": result.was_healed,
            "function": {
                "name": result.func.name,
                "description": result.func.description,
                "code": result.func.code,
            },
            "results": _summary_to_dict(result.summary),
        }
    except Exception as e:
        return {"error": str(e)}


# ── TDDGenerator Tools ────────────────────────────────────────────


@mcp.tool()
def tdd_generate_signature(
    description: str,
    name: str,
    is_async: bool = False,
    model: str | None = None,
    additional_context: str | None = None,
) -> dict[str, Any]:
    """Generate a function signature from a natural language description (TDD Step 1).

    The LLM designs appropriate parameter names, types, and return type.

    Args:
        description: What the function should do
        name: Desired function name
        is_async: Whether to generate an async function signature
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation
    """
    try:
        gen = TDDGenerator(model=model, additional_context=additional_context, load_env=True)
        sig = gen.generate_signature(description, name, is_async=is_async)
        return {
            "name": sig.name,
            "signature": sig.to_signature_str(),
            "stub": sig.to_stub(),
            "description": sig.description,
            "is_async": sig.is_async,
            "return_type": sig.return_type,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "description": p.description,
                    "required": p.is_required,
                }
                for p in sig.params
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def tdd_generate_evals(
    signature_name: str,
    signature_str: str,
    signature_description: str,
    params: list[dict[str, Any]],
    return_type: str = "Any",
    is_async: bool = False,
    min_cases: int = 5,
    model: str | None = None,
    additional_context: str = "",
) -> dict[str, Any]:
    """Generate eval spec from a function signature (TDD Step 2).

    Creates comprehensive test cases from the signature without an implementation.

    Args:
        signature_name: Function name
        signature_str: Full Python signature string (e.g. "def add(x: int, y: int) -> int")
        signature_description: Description of what the function does
        params: List of parameter dicts with keys: name, type, default (optional), description (optional)
        return_type: Return type annotation (default: "Any")
        is_async: Whether the function is async
        min_cases: Minimum number of test cases to generate (default: 5)
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation
    """
    try:
        from vowel.tdd import FunctionSignature, Param

        sig = FunctionSignature(
            name=signature_name,
            params=[
                Param(
                    name=p["name"],
                    type=p["type"],
                    default=p.get("default"),
                    description=p.get("description"),
                )
                for p in params
            ],
            return_type=return_type,
            description=signature_description,
            is_async=is_async,
        )

        gen = TDDGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )
        _, yaml_spec = gen.generate_evals_from_signature(
            sig,
            min_cases=min_cases,
            additional_context=additional_context,
        )
        return {
            "function_name": signature_name,
            "yaml_spec": yaml_spec,
            "message": "Eval spec generated from signature.",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def tdd_generate_implementation(
    signature_name: str,
    signature_description: str,
    params: list[dict[str, Any]],
    return_type: str,
    yaml_spec: str,
    is_async: bool = False,
    model: str | None = None,
    additional_context: str = "",
) -> dict[str, Any]:
    """Generate a function implementation that passes the given eval spec (TDD Step 3).

    The LLM writes code to satisfy all test cases in the spec.

    Args:
        signature_name: Function name
        signature_description: Description of the function
        params: List of parameter dicts with keys: name, type, default (optional), description (optional)
        return_type: Return type annotation
        yaml_spec: The YAML eval spec the implementation must pass
        is_async: Whether the function is async
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation
    """
    try:
        from vowel.tdd import FunctionSignature, Param

        sig = FunctionSignature(
            name=signature_name,
            params=[
                Param(
                    name=p["name"],
                    type=p["type"],
                    default=p.get("default"),
                    description=p.get("description"),
                )
                for p in params
            ],
            return_type=return_type,
            description=signature_description,
            is_async=is_async,
        )

        gen = TDDGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )
        func = gen.generate_implementation(sig, yaml_spec, additional_context=additional_context)
        return {
            "name": func.name,
            "description": func.description,
            "code": func.code,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def tdd_generate_all(
    description: str,
    name: str,
    is_async: bool = False,
    min_cases: int = 5,
    model: str | None = None,
    additional_context: str = "",
    max_impl_retries: int = 2,
    max_eval_retries: int = 0,
    max_flow_retries: int = 0,
    retry_delay: float = 3.0,
    ignore_duration: bool = True,
    min_coverage: float = 1.0,
) -> dict[str, Any]:
    """Full TDD flow: Description → Signature → Evals → Implementation.

    Complete automated TDD workflow:
    1. Generate function signature from description
    2. Generate eval spec (test cases) from signature
    3. Generate implementation that passes the tests
    4. Run tests and verify coverage

    Args:
        description: Natural language description of what the function should do
        name: Desired function name
        is_async: Whether to generate an async function
        min_cases: Minimum test cases to generate (default: 5)
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation prompts
        max_impl_retries: Max retries for implementation per flow attempt (default: 2)
        max_eval_retries: Max retries for eval spec regeneration (default: 0)
        max_flow_retries: Max retries for entire flow if all impl attempts fail (default: 0)
        retry_delay: Seconds between retries (default: 3.0)
        ignore_duration: Skip duration evaluator (default: true)
        min_coverage: Target pass rate 0.0-1.0 (default: 1.0)
    """
    try:
        gen = TDDGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )
        result = gen.generate_all(
            description=description,
            name=name,
            is_async=is_async,
            min_cases=min_cases,
            max_impl_retries=max_impl_retries,
            max_eval_retries=max_eval_retries,
            max_flow_retries=max_flow_retries,
            retry_delay=retry_delay,
            ignore_duration=ignore_duration,
            min_coverage=min_coverage,
            additional_context=additional_context,
        )
        return {
            "signature": result.signature.to_signature_str(),
            "yaml_spec": result.yaml_spec,
            "function": {
                "name": result.func.name,
                "description": result.func.description,
                "code": result.func.code,
            },
            "results": _summary_to_dict(result.summary),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def tdd_generate_and_validate(
    description: str,
    name: str,
    is_async: bool = False,
    min_cases: int = 5,
    model: str | None = None,
    additional_context: str = "",
    max_eval_retries: int = 2,
    min_coverage: float = 1.0,
    retry_delay: float = 2.0,
    ignore_duration: bool = True,
) -> dict[str, Any]:
    """TDD flow with eval spec validation against the generated implementation.

    Similar to tdd_generate_all, but validates the eval spec against the
    implementation — if test cases have wrong expected values, regenerates them.

    Flow:
    1. Generate signature
    2. Generate implementation
    3. Generate eval spec + validate against implementation (retry on failures)
    4. Return validated spec + passing implementation

    Args:
        description: Natural language description of the function
        name: Desired function name
        is_async: Whether to generate an async function
        min_cases: Minimum test cases (default: 5)
        model: LLM model identifier (defaults to MODEL_NAME env var)
        additional_context: Extra context for generation
        max_eval_retries: Max retries for eval spec validation (default: 2)
        min_coverage: Target coverage for validation (default: 1.0)
        retry_delay: Seconds between retries (default: 2.0)
        ignore_duration: Skip duration evaluator (default: true)
    """
    try:
        gen = TDDGenerator(
            model=model, additional_context=additional_context or None, load_env=True
        )

        # Step 1: Signature
        sig = gen.generate_signature(description, name, is_async=is_async)

        # Step 2: Implementation
        # First generate a simple spec to guide implementation
        _, initial_spec = gen.generate_evals_from_signature(
            sig, min_cases=min_cases, additional_context=additional_context, description=description
        )
        func = gen.generate_implementation(
            sig, initial_spec, additional_context=additional_context, description=description
        )

        # Step 3: Generate validated eval spec
        runner, yaml_spec = gen.generate_evals_from_signature(
            sig,
            min_cases=min_cases,
            func=func.impl,
            max_retries=max_eval_retries,
            min_coverage=min_coverage,
            retry_delay=retry_delay,
            ignore_duration=ignore_duration,
            additional_context=additional_context,
            description=description,
        )

        # Run final validation
        test_runner = runner.with_functions({name: func.impl})
        if ignore_duration:
            test_runner = test_runner.ignore_duration()
        summary = test_runner.run()

        return {
            "signature": sig.to_signature_str(),
            "yaml_spec": yaml_spec,
            "function": {
                "name": func.name,
                "description": func.description,
                "code": func.code,
            },
            "results": _summary_to_dict(summary),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Resources ─────────────────────────────────────────────────────


@mcp.resource("vowel://context")
def get_eval_context() -> str:
    """Get the vowel eval specification context/documentation."""
    return EVAL_SPEC_CONTEXT


@mcp.resource("vowel://example")
def get_example_yaml() -> str:
    """Get an example YAML eval specification."""
    return """# Example vowel eval specification

# Example 1: Basic function with type checking and assertions
add:
  evals:
    IsInteger:
      type: int
    IsPositive:
      assertion: "output >= 0"
    FastEnough:
      duration: 0.1  # seconds (global level)

  dataset:
    - case:
        id: basic_add
        inputs: [2, 3]
        expected: 5

    - case:
        id: negative_numbers
        inputs: [-5, 3]
        expected: -2

    - case:
        id: with_duration
        inputs: [100, 200]
        expected: 300
        duration: 50  # milliseconds (case level)

# Example 2: Function with fixtures (dependency injection)
fixtures:
  db:
    setup: my_module.setup_db
    teardown: my_module.close_db
    scope: function
    params:
      host: localhost
      port: 5432

process_user:
  fixture:
    - db
  evals:
    IsDict:
      type: dict
  dataset:
    - case:
        inputs: {user_id: 123}
        expected: {id: 123, name: "John", processed: true}

# Example 3: Exception testing
validate_email:
  dataset:
    - case:
        id: invalid_email
        input: "not-an-email"
        raises: ValueError
        match: "invalid email"

    - case:
        id: must_raise_something
        input: ""
        raises: any

# Example 4: Pattern matching
format_phone:
  dataset:
    - case:
        input: "1234567890"
        pattern: "^\\d{3}-\\d{3}-\\d{4}$"
        expected: "123-456-7890"

# Example 5: LLM Judge
summarize:
  evals:
    LLMJudge:
      rubric: "Summary captures the main points"
      config:
        model: "openai:gpt-4o-mini"
  dataset:
    - case:
        input: "Long article about AI..."
"""


# ── Helpers ───────────────────────────────────────────────────────


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    """Convert EvalSummary to a JSON-serializable dict."""
    results = []
    for result in summary.results:
        result_data: dict[str, Any] = {
            "eval_id": result.eval_id,
            "error": result.error,
            "success": result.success,
            "cases": [],
        }
        if result.report:
            for case in result.report.cases:
                case_data = {
                    "name": case.name,
                    "assertions": {
                        name: {
                            "passed": res.value,
                            "reason": str(res.reason) if res.reason else None,
                        }
                        for name, res in case.assertions.items()
                    },
                }
                result_data["cases"].append(case_data)
        results.append(result_data)

    return {
        "total_count": summary.total_count,
        "success_count": summary.success_count,
        "failed_count": summary.failed_count,
        "error_count": getattr(summary, "error_count", 0),
        "all_passed": summary.all_passed,
        "coverage": getattr(summary, "coverage", None),
        "results": results,
    }


def main():
    mcp.run()

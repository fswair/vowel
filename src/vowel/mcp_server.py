"""Vowel MCP Server - Model Context Protocol server for eval generation.

This module exposes vowel's evaluation capabilities via MCP (Model Context Protocol),
enabling AI assistants to run evaluations, generate functions, and test code.

Usage:
    # Run directly
    python -m vowel.mcp_server

    # Add to MCP client config (e.g., Claude Desktop, VS Code Copilot)
    {
        "mcpServers": {
            "vowel": {
                "command": "python",
                "args": ["-m", "vowel.mcp_server"]
            }
        }
    }

Available Tools:
    - run_evals_from_file: Run evaluations from a YAML file
    - run_evals_from_yaml: Run evaluations from YAML content string
    - run_evals_with_fixtures: Run evaluations with fixture injection
    - validate_yaml_spec: Validate a YAML eval specification
    - check_function_compatibility: Check if a function is compatible with eval generation
    - generate_function: Generate a function from natural language description
    - generate_eval_spec: Generate eval specification for a function
    - generate_and_run: Full workflow: description → function → evals → results
    - list_yaml_files: List YAML files in a directory

Available Resources:
    - vowel://context: Eval specification documentation
    - vowel://example: Example YAML eval specification
"""

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from vowel import check_compatibility, load_evals_from_yaml_string, run_evals
from vowel.ai import EVAL_SPEC_CONTEXT, EvalGenerator
from vowel.runner import Function, RunEvals

mcp = FastMCP("vowel")


@mcp.tool()
def run_evals_from_file(
    yaml_path: str,
    filter_funcs: list[str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Run evaluations from a YAML file.

    Args:
        yaml_path: Path to the YAML eval specification file
        filter_funcs: Optional list of function names to filter
        debug: Enable debug mode

    Returns:
        Summary of evaluation results including pass/fail counts and details
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
    """
    Run evaluations from YAML content string.

    ⚠️ Security: ``functions`` values are executed via ``exec()`` — only use in
    trusted environments.  Never expose this tool to untrusted input.

    Args:
        yaml_content: YAML eval specification as a string
        functions: Optional dict mapping function names to their code (will be exec'd)
        debug: Enable debug mode

    Returns:
        Summary of evaluation results
    """

    try:
        runner = RunEvals.from_source(yaml_content)

        if functions:
            func_objects = {}
            for name, code in functions.items():
                local_ns: dict[str, Any] = {}
                exec(code, {}, local_ns)
                if name in local_ns:
                    func_objects[name] = local_ns[name]
            runner = runner.with_functions(func_objects)

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
    """
    Run evaluations with fixture injection.

    Fixtures are injected as keyword-only arguments to functions.
    This enables dependency injection for setup/teardown logic.

    Args:
        yaml_content: YAML eval specification as a string
        functions: Optional dict mapping function names to their code
        fixtures: Optional dict mapping fixture names to their setup code
        debug: Enable debug mode

    Returns:
        Summary of evaluation results

    Example:
        fixtures = {
            "db": "def setup(): return {'connected': True}"
        }
    """
    try:
        runner = RunEvals.from_source(yaml_content)

        if functions:
            func_objects = {}
            for name, code in functions.items():
                local_ns: dict[str, Any] = {}
                exec(code, {}, local_ns)
                if name in local_ns:
                    func_objects[name] = local_ns[name]
            runner = runner.with_functions(func_objects)

        if fixtures:
            fixture_objects = {}
            for name, code in fixtures.items():
                local_ns: dict[str, Any] = {}
                exec(code, {}, local_ns)
                # Look for setup function or use the first callable found
                if "setup" in local_ns:
                    setup_fn = local_ns["setup"]
                    teardown_fn = local_ns.get("teardown")
                    fixture_objects[name] = (setup_fn, teardown_fn) if teardown_fn else setup_fn
                else:
                    # Use first callable found
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
    """
    Validate a YAML eval specification without running it.

    Checks for syntax errors, schema validation, and structural issues.

    Args:
        yaml_content: YAML eval specification to validate

    Returns:
        Validation results with any errors or warnings
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
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


@mcp.tool()
def check_function_compatibility(function_code: str, function_name: str) -> dict[str, Any]:
    """
    Check if a function is compatible with vowel eval generation.

    Verifies that all parameters have YAML-serializable types.

    Args:
        function_code: The Python function code
        function_name: Name of the function to check

    Returns:
        Compatibility info including unsupported parameters if any
    """
    try:
        local_ns: dict[str, Any] = {}
        exec(function_code, {}, local_ns)
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
def generate_function(
    description: str,
    model: str | None = None,
    async_func: bool = True,
) -> dict[str, Any]:
    """
    Generate a Python function from a description using LLM.

    Args:
        description: Natural language description of the function to generate
        model: LLM model to use (defaults to MODEL_NAME env var)
        async_func: Whether to generate an async function

    Returns:
        Generated function with name, description, and code
    """

    try:
        model = model or os.getenv("MODEL_NAME")
        if not model:
            return {
                "error": "Model not specified. Set MODEL_NAME env var or provide model parameter."
            }

        generator = EvalGenerator(model=model)
        func = generator.generate_function(description, async_func=async_func)

        return {
            "name": func.name,
            "description": func.description,
            "code": func.code,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def generate_eval_spec(
    function_code: str,
    function_name: str,
    function_description: str = "",
    model: str | None = None,
    additional_context: str = "",
) -> dict[str, Any]:
    """
    Generate an eval specification for a function.

    Args:
        function_code: The Python function code
        function_name: Name of the function
        function_description: Optional description of the function
        model: LLM model to use (defaults to MODEL_NAME env var)
        additional_context: Additional context for generation

    Returns:
        Generated YAML eval specification
    """

    try:
        model = model or os.getenv("MODEL_NAME")
        if not model:
            return {
                "error": "Model not specified. Set MODEL_NAME env var or provide model parameter."
            }

        local_ns: dict[str, Any] = {}
        exec(function_code, {}, local_ns)
        func_callable = local_ns.get(function_name)

        func = Function(
            name=function_name,
            description=function_description or f"Function {function_name}",
            code=function_code,
            func=func_callable,
        )

        generator = EvalGenerator(model=model, additional_context=additional_context)
        runner = generator.generate_spec(func, save_to_file=False)

        return {
            "function_name": function_name,
            "yaml_spec": (
                runner._source
                if hasattr(runner, "_source") and isinstance(runner._source, str)
                else "Generated successfully"
            ),
            "message": "Eval spec generated. Use run_evals_from_yaml to execute.",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def generate_and_run(
    description: str,
    model: str | None = None,
    auto_retry: bool = True,
    max_retries: int = 3,
    min_coverage: float = 0.9,
    heal_function: bool = True,
) -> dict[str, Any]:
    """
    Generate a function from description, create evals, and run them.

    This is the full workflow: description -> function -> evals -> results.

    Args:
        description: Natural language description of the function
        model: LLM model to use (defaults to MODEL_NAME env var)
        auto_retry: Retry on failures
        max_retries: Maximum retry attempts
        min_coverage: Required pass rate (0-1)
        heal_function: Attempt to fix buggy functions

    Returns:
        Complete results including function code and eval summary
    """

    try:
        model = model or os.getenv("MODEL_NAME")
        if not model:
            return {
                "error": "Model not specified. Set MODEL_NAME env var or provide model parameter."
            }

        generator = EvalGenerator(model=model)

        func = generator.generate_function(description)

        summary = generator.generate_and_run(
            func,
            auto_retry=auto_retry,
            max_retries=max_retries,
            min_coverage=min_coverage,
            heal_function=heal_function,
        )

        return {
            "function": {
                "name": func.name,
                "description": func.description,
                "code": func.code,
            },
            "results": _summary_to_dict(summary),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_yaml_files(directory: str, recursive: bool = True) -> dict[str, Any]:
    """
    List all YAML eval files in a directory.

    Args:
        directory: Directory path to search
        recursive: Search subdirectories

    Returns:
        List of YAML file paths found
    """
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}

    yaml_files = []
    patterns = ["*.yml", "*.yaml"]

    for pattern in patterns:
        if recursive:
            yaml_files.extend(path.rglob(pattern))
        else:
            yaml_files.extend(path.glob(pattern))

    return {
        "directory": str(path.absolute()),
        "recursive": recursive,
        "files": [str(f) for f in sorted(yaml_files)],
        "count": len(yaml_files),
    }


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

# Example 4: Pattern matching
format_phone:
  dataset:
    - case:
        input: "1234567890"
        pattern: "^\\d{3}-\\d{3}-\\d{4}$"
        expected: "123-456-7890"
"""


def _summary_to_dict(summary) -> dict[str, Any]:
    """Convert EvalSummary to a JSON-serializable dict."""
    results = []
    for result in summary.results:
        result_data = {
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
        "error_count": summary.error_count,
        "all_passed": summary.all_passed,
        "coverage": getattr(summary, "coverage", None),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run()

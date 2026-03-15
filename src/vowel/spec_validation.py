"""Shared spec validation utilities for eval generation pipelines.

Functions in this module are used by both ``CodeModeGenerator`` and
``TDDGenerator`` to validate generated YAML specs against real execution
and to inject measured durations.
"""

from __future__ import annotations

from typing import Any

import logfire
import yaml

from vowel.executor import Executor, get_executor
from vowel.runner import Function
from vowel.utils import EvalSummary


def build_failure_context(summary: EvalSummary) -> str:
    """Build a concise failure report to inject into a retry prompt.

    Iterates over :class:`EvalSummary` results and formats each failed
    case/assertion as a single line.  Returns a multi-line string suitable
    for LLM prompts.
    """
    lines: list[str] = []
    for result in summary.results:
        if result.report:
            for case in result.report.cases:
                failed_assertions = {k: v for k, v in case.assertions.items() if not v.value}
                if failed_assertions:
                    parts = []
                    for k, v in failed_assertions.items():
                        if v.reason:
                            parts.append(f"{k}: {v.reason}")
                        else:
                            parts.append(f"{k}: FAILED")
                    lines.append(f"- Case '{case.name}' FAILED [{', '.join(parts)}]")
        if result.error:
            lines.append(f"- Error: {result.error}")
    return "\n".join(lines) if lines else "Unknown failures"


def build_call_code(
    func_name: str, case: dict
) -> (
    str | None
):  # TODO: intead of building call code, consider passing arguments through executor inputs
    """Build a ``func(args...)`` call string from a YAML case dict.

    Returns ``None`` when no input is present (e.g. raises-only case
    without input).
    """
    if "inputs" in case and case["inputs"] is not None:
        args = case["inputs"]
        if isinstance(args, list):
            arg_strs = ", ".join(repr(a) for a in args)
            return f"{func_name}({arg_strs})"
        if isinstance(args, dict):
            kwarg_strs = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return f"{func_name}({kwarg_strs})"
    elif "input" in case and case["input"] is not None:
        return f"{func_name}({case['input']!r})"
    return None


def inject_durations(
    yaml_spec: str,
    func: Function,
    executor: Executor,
    *,
    buffer_pct: float = 0.5,
    floor_ms: float = 10.0,
) -> str:
    """Add per-case ``duration`` fields based on actual execution times.

    Each non-raises case is executed once via the executor session.
    The measured ``duration_ms`` is inflated by *buffer_pct* (default 50%)
    with a minimum of *floor_ms* (default 10 ms) to absorb noise.

    Parameters
    ----------
    yaml_spec:
        YAML string to augment.
    func:
        Function to execute cases against.
    executor:
        Executor backend to use for timing.
    buffer_pct:
        Fractional buffer added on top of measured time (0.5 = +50%).
    floor_ms:
        Absolute minimum duration in ms — protects sub-ms cases from
        flaky failures due to measurement noise.
    """
    spec = yaml.safe_load(yaml_spec)
    if not isinstance(spec, dict):
        return yaml_spec

    try:
        session = executor.create_session(func.code)
    except Exception:
        logfire.warn("Could not create session for duration injection")
        return yaml_spec

    with session:
        for eval_id, eval_def in spec.items():
            if not isinstance(eval_def, dict):
                continue
            for case_entry in eval_def.get("dataset", []):
                case = case_entry.get("case", {})
                if not isinstance(case, dict):
                    continue
                # Skip cases that expect exceptions
                if case.get("raises"):
                    continue

                call_code = build_call_code(eval_id, case)
                if call_code is None:
                    continue

                result = session.feed(call_code)
                if result.success:
                    dur = max(
                        result.duration_ms * (1 + buffer_pct),
                        floor_ms,
                    )
                    case["duration"] = round(dur, 1)

    return yaml.dump(spec, default_flow_style=False, allow_unicode=True, sort_keys=False)


def validate_expected_values(
    yaml_spec: str,
    func: Function,
    executor: Executor | None = None,
) -> str:
    """Validate and fix expected values in a YAML spec by executing cases.

    For each case that has ``expected`` and no ``raises``, executes the
    function call and compares the result.  If the actual output differs
    from the YAML expected value, the YAML is updated to the real value.

    Also validates ``raises`` cases: if the case expects an exception but
    the function doesn't raise (or raises a different type), the case is
    corrected.

    Parameters
    ----------
    yaml_spec:
        YAML spec string to validate.
    func:
        Function to execute.
    executor:
        Executor backend.  Defaults to ``get_executor("auto")``.

    Returns
    -------
    str
        Fixed YAML spec with corrected expected values.
    """
    executor = executor or get_executor("auto")

    spec = yaml.safe_load(yaml_spec)
    if not isinstance(spec, dict):
        return yaml_spec

    try:
        session = executor.create_session(func.code)
    except Exception:
        logfire.warn("Could not create session for expected value validation")
        return yaml_spec

    fixes_applied = 0

    with session:
        for eval_id, eval_def in spec.items():
            if not isinstance(eval_def, dict):
                continue
            for case_entry in eval_def.get("dataset", []):
                case = case_entry.get("case", {})
                if not isinstance(case, dict):
                    continue

                call_code = build_call_code(eval_id, case)
                if call_code is None:
                    continue

                result = session.feed(call_code)

                # --- Fix expected values ---
                if "expected" in case and not case.get("raises"):
                    if result.success and result.output != case["expected"]:
                        logfire.info(
                            "Fixing expected value for case: {expected} → {actual}",
                            expected=repr(case["expected"]),
                            actual=repr(result.output),
                        )
                        case["expected"] = result.output
                        fixes_applied += 1

                # --- Fix raises cases ---
                if case.get("raises"):
                    expected_exc = case["raises"]
                    if result.success:
                        # Function didn't raise — remove raises, set expected
                        logfire.info(
                            "Case expected {exc} but function returned {output}, fixing",
                            exc=expected_exc,
                            output=repr(result.output),
                        )
                        del case["raises"]
                        if "match" in case:
                            del case["match"]
                        case["expected"] = result.output
                        fixes_applied += 1
                    elif result.error_type and result.error_type != expected_exc:
                        # Wrong exception type
                        logfire.info(
                            "Case expected {expected} but got {actual}, fixing",
                            expected=expected_exc,
                            actual=result.error_type,
                        )
                        case["raises"] = result.error_type
                        fixes_applied += 1

    if fixes_applied > 0:
        logfire.info("Validated spec: {count} fixes applied", count=fixes_applied)
        return yaml.dump(spec, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return yaml_spec


def inject_missing_error_cases(
    yaml_spec: str,
    func_name: str,
    error_snippets: list[dict],
) -> str:
    """Inject error cases from exploration into the spec if the LLM missed them.

    Each item in *error_snippets* should have keys:

    - ``code``: Python snippet that triggered the error (e.g. ``"flatten(None)"``)
    - ``error_type``: Exception class name (e.g. ``"TypeError"``)
    - ``error``: Full error message
    - ``description``: One-line description

    Uses :mod:`ast` to extract function call arguments from the snippet
    code.  If parsing fails (multi-line code, complex expressions), the
    snippet is silently skipped.

    Returns the (possibly modified) YAML spec string.
    """
    import ast

    if not error_snippets:
        return yaml_spec

    spec = yaml.safe_load(yaml_spec)
    if not isinstance(spec, dict) or func_name not in spec:
        return yaml_spec

    eval_def = spec[func_name]
    dataset = eval_def.setdefault("dataset", [])

    # Collect existing raises case inputs to avoid duplicates
    existing_raises_inputs: set[str] = set()
    for entry in dataset:
        case = entry.get("case", {})
        if isinstance(case, dict) and case.get("raises"):
            # Normalise existing input for comparison
            inp = case.get("input")
            inps = case.get("inputs")
            existing_raises_inputs.add(repr((inp, inps)))

    injected = 0

    for snippet in error_snippets:
        code = snippet["code"].strip()
        error_type = snippet["error_type"]
        description = snippet.get("description", "")

        # Try to extract arguments from a simple function call
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError:
            continue

        if not isinstance(tree.body, ast.Call):
            continue

        try:
            args = [ast.literal_eval(a) for a in tree.body.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
        except (ValueError, TypeError):
            # Complex expression that can't be literal-evaluted — skip
            continue

        # Determine input/inputs format
        if kwargs:
            input_repr = repr((None, kwargs))
            if input_repr in existing_raises_inputs:
                continue
            case_dict: dict[str, Any] = {
                "id": f"error_{error_type.lower()}_{injected}",
                "inputs": kwargs,
                "raises": error_type,
            }
        elif len(args) == 1:
            input_repr = repr((args[0], None))
            if input_repr in existing_raises_inputs:
                continue
            case_dict = {
                "id": f"error_{error_type.lower()}_{injected}",
                "input": args[0],
                "raises": error_type,
            }
        elif len(args) > 1:
            input_repr = repr((None, args))
            if input_repr in existing_raises_inputs:
                continue
            case_dict = {
                "id": f"error_{error_type.lower()}_{injected}",
                "inputs": args,
                "raises": error_type,
            }
        else:
            continue

        dataset.append({"case": case_dict})
        injected += 1
        logfire.info(
            "Injected error case: {desc} → raises {exc}",
            desc=description,
            exc=error_type,
        )

    if injected > 0:
        logfire.info("Injected {count} missing error cases into spec", count=injected)
        return yaml.dump(spec, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return yaml_spec

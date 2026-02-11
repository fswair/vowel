"""Static validator for LLM-generated eval specifications.

Catches common LLM generation mistakes BEFORE the spec is used:
1. Extra fields in cases (comment, note, description, etc.)
2. YAML-unparseable type remnants (set literals, tuple strings, float('inf'), etc.)
3. Invented exception types not in function code
4. Removes or fixes problematic cases, returns clean YAML

Usage:
    from vowel.validation import validate_and_fix_spec

    fixed_yaml, warnings = validate_and_fix_spec(yaml_str, function_code="def foo(x): ...")
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import logfire
import yaml

# Fields allowed in a case block (from MatchCase model)
ALLOWED_CASE_FIELDS = frozenset(
    {
        "id",
        "input",
        "inputs",
        "expected",
        "duration",
        "contains",
        "assertion",
        "pattern",
        "case_sensitive",
        "raises",
        "type",
        "strict_type",
        "match",
    }
)

# Patterns that indicate YAML parsed a Python expression as a string
# These are values that look like Python code but YAML turned into strings
YAML_TYPE_REMNANTS: list[tuple[re.Pattern, str]] = [
    # float('inf'), float('nan'), float('-inf')
    (
        re.compile(r"^float\(['\"][-+]?(inf|nan)['\"]\)$", re.IGNORECASE),
        "float('inf')/float('nan') is parsed as string in YAML",
    ),
    # tuple: (1, 2, 3)
    (re.compile(r"^\(.*,.*\)$"), "tuple literal is parsed as string in YAML"),
    # set(): set([1,2])
    (re.compile(r"^set\("), "set() constructor is parsed as string in YAML"),
    # frozenset()
    (re.compile(r"^frozenset\("), "frozenset() constructor is parsed as string in YAML"),
    # bytes: b"hello", b'hello'
    (re.compile(r"^b['\"]"), "bytes literal is parsed as string in YAML"),
    # complex: 1+2j
    (re.compile(r"^\d+[+-]\d+j$"), "complex number is parsed as string in YAML"),
    # Python expressions with function calls
    (
        re.compile(r"^(range|slice|memoryview|bytearray)\("),
        "Python built-in constructor is parsed as string in YAML",
    ),
]

# Exception types that Python can actually raise
COMMON_EXCEPTIONS = frozenset(
    {
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "AttributeError",
        "FileNotFoundError",
        "RuntimeError",
        "NotImplementedError",
        "StopIteration",
        "OverflowError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeError",
        "IOError",
        "OSError",
        "PermissionError",
        "TimeoutError",
        "ConnectionError",
        "ImportError",
        "ModuleNotFoundError",
        "ArithmeticError",
        "LookupError",
        "AssertionError",
        "RecursionError",
        "MemoryError",
        "BufferError",
        "EOFError",
        "SystemError",
    }
)

# Exception types that LLMs commonly invent but rarely exist in real code
SUSPICIOUS_EXCEPTIONS = {
    "OverflowError": "Python handles arbitrarily large ints natively. OverflowError is rare except for float overflow. Verify the function actually raises this.",
    "decimal.InvalidOperation": "Must be qualified as 'InvalidOperation' and the function must explicitly raise it or use a Decimal context that triggers it.",
}


@dataclass
class ValidationWarning:
    """A warning generated during spec validation."""

    case_id: str | None
    category: str | Literal["extra_field", "type_remnant", "invented_raises", "yaml_set_as_dict"]
    message: str
    auto_fixed: bool = False


@dataclass
class ValidationResult:
    """Result of static spec validation."""

    original_yaml: str
    fixed_yaml: str
    warnings: list[ValidationWarning] = field(default_factory=list)
    cases_removed: int = 0
    cases_fixed: int = 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def was_modified(self) -> bool:
        return self.original_yaml != self.fixed_yaml

    def summary(self) -> str:
        if not self.has_warnings:
            return "No issues found."
        lines = [f"Found {len(self.warnings)} issue(s):"]
        for w in self.warnings:
            prefix = "[FIXED]" if w.auto_fixed else "[WARN]"
            case_info = f" (case: {w.case_id})" if w.case_id else ""
            lines.append(f"  {prefix} [{w.category}]{case_info}: {w.message}")
        if self.cases_removed > 0:
            lines.append(f"  Removed {self.cases_removed} invalid case(s).")
        if self.cases_fixed > 0:
            lines.append(f"  Fixed {self.cases_fixed} case(s).")
        return "\n".join(lines)


def _check_value_for_type_remnants(value: object, path: str) -> list[ValidationWarning]:
    """Check a single value for YAML type remnant patterns."""
    warnings = []

    if isinstance(value, str):
        for pattern, reason in YAML_TYPE_REMNANTS:
            if pattern.search(value):
                warnings.append(
                    ValidationWarning(
                        case_id=None,  # filled by caller
                        category="type_remnant",
                        message=f"{path}: value '{value}' — {reason}",
                    )
                )
                break

    elif isinstance(value, list):
        for i, item in enumerate(value):
            warnings.extend(_check_value_for_type_remnants(item, f"{path}[{i}]"))

    elif isinstance(value, dict):
        for k, v in value.items():
            warnings.extend(_check_value_for_type_remnants(v, f"{path}.{k}"))

    return warnings


def _is_yaml_set_as_dict(value: object) -> bool:
    """Check if a value looks like YAML parsed {1, 2, 3} as {1: null, 2: null, 3: null}."""
    if isinstance(value, dict) and len(value) > 0:
        return all(v is None for v in value.values()) and all(
            isinstance(k, (int, float)) for k in value
        )
    return False


def _check_raises_in_code(raises_type: str, function_code: str) -> bool:
    """Check if a function plausibly raises the given exception type.

    Returns True if:
    - The exception type appears in a `raise` statement
    - The code has operations that naturally produce this exception
    - The exception comes from a standard library call in the code
    """
    # Normalize: "decimal.InvalidOperation" -> check both forms
    short_name = raises_type.rsplit(".", 1)[-1] if "." in raises_type else raises_type

    # Direct raise statement
    if re.search(rf"\braise\s+{re.escape(short_name)}", function_code):
        return True

    # Exception appears in except clause (might be re-raised)
    if re.search(rf"\bexcept\s+.*{re.escape(short_name)}", function_code):
        return True

    # Natural exceptions from operations
    natural_raisers = {
        "ZeroDivisionError": [r"[/]", r"\bdivmod\b"],
        "KeyError": [r"\[.+\]"],  # dict access
        "IndexError": [r"\[.+\]"],  # list access
        "TypeError": [r"\+", r"\bint\(", r"\bfloat\(", r"\blen\("],
        "ValueError": [r"\bint\(", r"\bfloat\(", r"\.index\(", r"\.remove\("],
        "AttributeError": [r"\."],  # any attribute access
        "StopIteration": [r"\bnext\("],
        "UnicodeDecodeError": [r"\.decode\(", r"\bstr\(.*encoding"],
        "UnicodeEncodeError": [r"\.encode\("],
        "FileNotFoundError": [r"\bopen\("],
        "OverflowError": [r"\bmath\.", r"\bfloat\("],  # Only from float/math ops
    }

    if short_name in natural_raisers:
        for pattern in natural_raisers[short_name]:
            if re.search(pattern, function_code):
                return True

    # Module-qualified exceptions from imports
    if "." in raises_type:
        module = raises_type.rsplit(".", 1)[0]
        if re.search(
            rf"\b(import\s+{re.escape(module)}|from\s+{re.escape(module)})", function_code
        ):
            return True

    return False


def validate_and_fix_spec(
    yaml_str: str,
    *,
    function_code: str | None = None,
) -> ValidationResult:
    """Validate and fix a generated YAML eval spec.

    Performs static checks and auto-fixes where possible:
    1. Removes extra fields from cases (comment, note, etc.)
    2. Removes cases with YAML type remnants in inputs
    3. Warns about suspicious raises cases

    Args:
        yaml_str: The YAML eval spec string
        function_code: Optional function source code for raises validation

    Returns:
        ValidationResult with fixed YAML and warnings
    """
    result = ValidationResult(original_yaml=yaml_str, fixed_yaml=yaml_str)

    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        # If YAML can't parse, nothing we can do
        return result

    if not isinstance(data, dict):
        return result

    modified = False

    for _func_name, func_spec in data.items():
        if not isinstance(func_spec, dict):
            continue

        dataset = func_spec.get("dataset")
        if not isinstance(dataset, list):
            continue

        cases_to_remove: list[int] = []

        for i, entry in enumerate(dataset):
            if not isinstance(entry, dict) or "case" not in entry:
                continue

            case = entry["case"]
            if not isinstance(case, dict):
                continue

            case_id = case.get("id", f"case_{i}")

            # CHECK 1: Extra fields
            extra_fields = set(case.keys()) - ALLOWED_CASE_FIELDS
            if extra_fields:
                for ef in extra_fields:
                    result.warnings.append(
                        ValidationWarning(
                            case_id=case_id,
                            category="extra_field",
                            message=f"Removed unknown field '{ef}' (not in MatchCase schema)",
                            auto_fixed=True,
                        )
                    )
                    del case[ef]
                modified = True
                result.cases_fixed += 1

            # CHECK 2: YAML set-as-dict detection {1: null, 2: null} = was {1, 2}
            for field_name in ("input", "expected"):
                val = case.get(field_name)
                if _is_yaml_set_as_dict(val):
                    result.warnings.append(
                        ValidationWarning(
                            case_id=case_id,
                            category="yaml_set_as_dict",
                            message=f"'{field_name}' looks like a set parsed as dict (all values are null). Removing case.",
                        )
                    )
                    cases_to_remove.append(i)
                    break

            if i in cases_to_remove:
                continue

            # Also check inputs list items
            inputs_val = case.get("inputs")
            if isinstance(inputs_val, list):
                for j, inp in enumerate(inputs_val):
                    if _is_yaml_set_as_dict(inp):
                        result.warnings.append(
                            ValidationWarning(
                                case_id=case_id,
                                category="yaml_set_as_dict",
                                message=f"'inputs[{j}]' looks like a set parsed as dict. Removing case.",
                            )
                        )
                        cases_to_remove.append(i)
                        break

            if i in cases_to_remove:
                continue

            # CHECK 3: Type remnants in string values
            case_warnings: list[ValidationWarning] = []
            for field_name in ("input", "expected"):
                val = case.get(field_name)
                if val is not None:
                    ws = _check_value_for_type_remnants(val, field_name)
                    for w in ws:
                        w.case_id = case_id
                    case_warnings.extend(ws)

            if isinstance(inputs_val, list):
                for j, inp in enumerate(inputs_val):
                    ws = _check_value_for_type_remnants(inp, f"inputs[{j}]")
                    for w in ws:
                        w.case_id = case_id
                    case_warnings.extend(ws)

            if case_warnings:
                result.warnings.extend(case_warnings)
                cases_to_remove.append(i)

            # CHECK 4: Suspicious raises
            raises_type = case.get("raises")
            if raises_type and function_code:
                if not _check_raises_in_code(raises_type, function_code):
                    result.warnings.append(
                        ValidationWarning(
                            case_id=case_id,
                            category="invented_raises",
                            message=f"Exception '{raises_type}' not found in function code. Removing case.",
                        )
                    )
                    cases_to_remove.append(i)
                elif raises_type in SUSPICIOUS_EXCEPTIONS:
                    result.warnings.append(
                        ValidationWarning(
                            case_id=case_id,
                            category="suspicious_raises",
                            message=f"'{raises_type}': {SUSPICIOUS_EXCEPTIONS[raises_type]}",
                        )
                    )

        # Remove problematic cases (reverse order to preserve indices)
        unique_removals = sorted(set(cases_to_remove), reverse=True)
        for idx in unique_removals:
            dataset.pop(idx)
            result.cases_removed += 1
            modified = True

    if modified:
        result.fixed_yaml = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
        logfire.info(
            "Spec validation fixed issues",
            warnings=len(result.warnings),
            cases_removed=result.cases_removed,
            cases_fixed=result.cases_fixed,
        )

    return result

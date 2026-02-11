"""Tests for the static eval spec validator."""

import yaml

from vowel.validation import (
    ALLOWED_CASE_FIELDS,
    _check_raises_in_code,
    _check_value_for_type_remnants,
    _is_yaml_set_as_dict,
    validate_and_fix_spec,
)

# ─── ALLOWED_CASE_FIELDS ────────────────────────────────────────────


class TestAllowedFields:
    def test_known_fields_present(self):
        """All MatchCase fields must be in the allowed set."""
        expected = {
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
        assert expected == ALLOWED_CASE_FIELDS


# ─── YAML SET-AS-DICT ───────────────────────────────────────────────


class TestIsYamlSetAsDict:
    def test_set_parsed_as_dict(self):
        """YAML {1, 2, 3} becomes {1: null, 2: null, 3: null}."""
        assert _is_yaml_set_as_dict({1: None, 2: None, 3: None}) is True

    def test_normal_dict(self):
        assert _is_yaml_set_as_dict({"a": 1, "b": 2}) is False

    def test_empty_dict(self):
        assert _is_yaml_set_as_dict({}) is False

    def test_not_dict(self):
        assert _is_yaml_set_as_dict([1, 2, 3]) is False

    def test_mixed_values(self):
        """Dict with some None values but not all."""
        assert _is_yaml_set_as_dict({1: None, 2: "x"}) is False

    def test_string_keys_with_none_values(self):
        """Only suspicious when keys are numeric (like set literals)."""
        assert _is_yaml_set_as_dict({"a": None, "b": None}) is False


# ─── TYPE REMNANTS ───────────────────────────────────────────────────


class TestTypeRemnants:
    def test_float_inf(self):
        warnings = _check_value_for_type_remnants("float('inf')", "input")
        assert len(warnings) == 1
        assert "float('inf')" in warnings[0].message

    def test_float_nan(self):
        warnings = _check_value_for_type_remnants("float('nan')", "input")
        assert len(warnings) == 1

    def test_float_neg_inf(self):
        warnings = _check_value_for_type_remnants("float('-inf')", "input")
        assert len(warnings) == 1

    def test_tuple_string(self):
        warnings = _check_value_for_type_remnants("(1, 2, 3)", "input")
        assert len(warnings) == 1
        assert "tuple" in warnings[0].message

    def test_set_constructor(self):
        warnings = _check_value_for_type_remnants("set([1,2,3])", "input")
        assert len(warnings) == 1

    def test_bytes_literal(self):
        warnings = _check_value_for_type_remnants("b'hello'", "input")
        assert len(warnings) == 1
        assert "bytes" in warnings[0].message

    def test_bytes_double_quote(self):
        warnings = _check_value_for_type_remnants('b"hello"', "input")
        assert len(warnings) == 1

    def test_complex_number(self):
        warnings = _check_value_for_type_remnants("1+2j", "input")
        assert len(warnings) == 1

    def test_range_constructor(self):
        warnings = _check_value_for_type_remnants("range(10)", "input")
        assert len(warnings) == 1

    def test_normal_string(self):
        warnings = _check_value_for_type_remnants("hello world", "input")
        assert len(warnings) == 0

    def test_normal_int(self):
        warnings = _check_value_for_type_remnants(42, "input")
        assert len(warnings) == 0

    def test_nested_in_list(self):
        warnings = _check_value_for_type_remnants([1, "float('inf')", 3], "inputs")
        assert len(warnings) == 1
        assert "inputs[1]" in warnings[0].message

    def test_nested_in_dict(self):
        warnings = _check_value_for_type_remnants({"val": "float('nan')"}, "input")
        assert len(warnings) == 1
        assert "input.val" in warnings[0].message

    def test_normal_list(self):
        warnings = _check_value_for_type_remnants([1, 2, 3], "input")
        assert len(warnings) == 0


# ─── CHECK RAISES IN CODE ───────────────────────────────────────────


class TestCheckRaisesInCode:
    def test_explicit_raise(self):
        code = """
def foo(x):
    if x < 0:
        raise ValueError("must be positive")
    return x
"""
        assert _check_raises_in_code("ValueError", code) is True

    def test_no_raise(self):
        code = """
def foo(x):
    return x + 1
"""
        assert _check_raises_in_code("ValueError", code) is False

    def test_overflow_not_in_code(self):
        code = """
def foo(x):
    return str(x) + "th"
"""
        assert _check_raises_in_code("OverflowError", code) is False

    def test_except_clause(self):
        code = """
def foo(x):
    try:
        return int(x)
    except ValueError:
        return None
"""
        assert _check_raises_in_code("ValueError", code) is True

    def test_zero_division_natural(self):
        code = """
def foo(a, b):
    return a / b
"""
        assert _check_raises_in_code("ZeroDivisionError", code) is True

    def test_key_error_from_dict_access(self):
        code = """
def foo(d, key):
    return d[key]
"""
        assert _check_raises_in_code("KeyError", code) is True

    def test_qualified_exception_with_import(self):
        code = """
import decimal
def foo(x):
    return decimal.Decimal(x)
"""
        assert _check_raises_in_code("decimal.InvalidOperation", code) is True

    def test_qualified_exception_without_import(self):
        code = """
def foo(x):
    return x + 1
"""
        assert _check_raises_in_code("decimal.InvalidOperation", code) is False

    def test_unicode_decode_from_decode(self):
        code = """
def foo(s, encoding='utf-8'):
    return s.decode(encoding)
"""
        assert _check_raises_in_code("UnicodeDecodeError", code) is True

    def test_type_error_from_int_cast(self):
        code = """
def foo(x):
    return int(x) + 1
"""
        assert _check_raises_in_code("TypeError", code) is True

    def test_type_error_not_plausible(self):
        """Simple return with no type-sensitive ops."""
        code = """
def foo(x):
    return "hello"
"""
        assert _check_raises_in_code("TypeError", code) is False


# ─── FULL VALIDATION ────────────────────────────────────────────────


class TestValidateAndFixSpec:
    def test_clean_spec_no_changes(self):
        yaml_str = """
add:
  dataset:
    - case:
        id: basic
        inputs: [2, 3]
        expected: 5
"""
        result = validate_and_fix_spec(yaml_str)
        assert not result.has_warnings
        assert not result.was_modified
        assert result.cases_removed == 0

    def test_extra_field_removed(self):
        yaml_str = """
foo:
  dataset:
    - case:
        id: test1
        input: 5
        expected: 10
        comment: "this should be removed"
"""
        result = validate_and_fix_spec(yaml_str)
        assert result.has_warnings
        assert result.was_modified
        assert result.cases_fixed == 1
        # Verify comment is gone from the fixed YAML
        fixed = yaml.safe_load(result.fixed_yaml)
        case = fixed["foo"]["dataset"][0]["case"]
        assert "comment" not in case
        assert case["expected"] == 10  # other fields preserved

    def test_multiple_extra_fields(self):
        yaml_str = """
foo:
  dataset:
    - case:
        id: test1
        input: 5
        expected: 10
        comment: "remove me"
        note: "also remove"
        description: "and this"
"""
        result = validate_and_fix_spec(yaml_str)
        assert len([w for w in result.warnings if w.category == "extra_field"]) == 3

    def test_set_as_dict_removed(self):
        """YAML {1, 2, 3} is parsed as {1: null, 2: null, 3: null} - detect and remove."""
        # Build YAML that simulates what YAML does to set literals
        data = {
            "foo": {
                "dataset": [
                    {"case": {"id": "good", "input": [1, 2, 3], "expected": 6}},
                    {
                        "case": {
                            "id": "bad_set",
                            "input": {1: None, 2: None, 3: None},
                            "expected": 6,
                        }
                    },
                ]
            }
        }
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        result = validate_and_fix_spec(yaml_str)
        assert result.cases_removed == 1
        fixed = yaml.safe_load(result.fixed_yaml)
        assert len(fixed["foo"]["dataset"]) == 1
        assert fixed["foo"]["dataset"][0]["case"]["id"] == "good"

    def test_type_remnant_in_input_removes_case(self):
        yaml_str = """
foo:
  dataset:
    - case:
        id: good
        input: 5
        expected: 10
    - case:
        id: bad_inf
        inputs: [1.0, "float('inf')"]
        raises: ValueError
"""
        result = validate_and_fix_spec(yaml_str)
        assert result.cases_removed == 1
        fixed = yaml.safe_load(result.fixed_yaml)
        assert len(fixed["foo"]["dataset"]) == 1

    def test_type_remnant_in_expected_removes_case(self):
        yaml_str = """
foo:
  dataset:
    - case:
        id: bad_tuple
        input: [1, 2]
        expected: "(1, 2)"
"""
        result = validate_and_fix_spec(yaml_str)
        assert result.cases_removed == 1

    def test_invented_raises_removed(self):
        code = """
def ordinal(num):
    return str(num) + "th"
"""
        yaml_str = """
ordinal:
  dataset:
    - case:
        id: basic
        input: 1
        expected: "1th"
    - case:
        id: overflow
        input: 99999999999999999999999999999999
        raises: OverflowError
"""
        result = validate_and_fix_spec(yaml_str, function_code=code)
        assert result.cases_removed == 1
        fixed = yaml.safe_load(result.fixed_yaml)
        assert len(fixed["ordinal"]["dataset"]) == 1
        assert fixed["ordinal"]["dataset"][0]["case"]["id"] == "basic"

    def test_valid_raises_kept(self):
        code = """
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b
"""
        yaml_str = """
divide:
  dataset:
    - case:
        id: basic
        inputs: [10, 2]
        expected: 5.0
    - case:
        id: zero_div
        inputs: [10, 0]
        raises: ZeroDivisionError
"""
        result = validate_and_fix_spec(yaml_str, function_code=code)
        assert result.cases_removed == 0
        fixed = yaml.safe_load(result.fixed_yaml)
        assert len(fixed["divide"]["dataset"]) == 2

    def test_no_function_code_skips_raises_check(self):
        """Without function_code, raises check is skipped."""
        yaml_str = """
foo:
  dataset:
    - case:
        id: bad
        input: 1
        raises: OverflowError
"""
        result = validate_and_fix_spec(yaml_str, function_code=None)
        assert result.cases_removed == 0  # no code to check against

    def test_combined_issues(self):
        """Multiple issue types in one spec."""
        code = """
def foo(x):
    return x + 1
"""
        yaml_str = """
foo:
  dataset:
    - case:
        id: good
        input: 5
        expected: 6
    - case:
        id: has_comment
        input: 10
        expected: 11
        comment: "extra field"
    - case:
        id: bad_type
        inputs: [1, "float('nan')"]
        expected: 2
    - case:
        id: invented_error
        input: -1
        raises: OverflowError
"""
        result = validate_and_fix_spec(yaml_str, function_code=code)
        # comment case: fixed (field removed, case kept)
        # bad_type case: removed
        # invented_error case: removed
        assert result.cases_fixed == 1
        assert result.cases_removed == 2
        fixed = yaml.safe_load(result.fixed_yaml)
        assert len(fixed["foo"]["dataset"]) == 2
        ids = [c["case"]["id"] for c in fixed["foo"]["dataset"]]
        assert "good" in ids
        assert "has_comment" in ids

    def test_invalid_yaml_returns_unchanged(self):
        """If YAML can't parse, return original unchanged."""
        bad_yaml = "{{{{invalid yaml"
        result = validate_and_fix_spec(bad_yaml)
        assert result.fixed_yaml == bad_yaml
        assert not result.has_warnings

    def test_non_dict_yaml_returns_unchanged(self):
        result = validate_and_fix_spec("- just a list")
        assert not result.has_warnings

    def test_validation_result_summary(self):
        yaml_str = """
foo:
  dataset:
    - case:
        id: test
        input: 5
        comment: "extra"
"""
        result = validate_and_fix_spec(yaml_str)
        summary = result.summary()
        assert "extra_field" in summary
        assert "FIXED" in summary

    def test_empty_dataset(self):
        yaml_str = """
foo:
  dataset: []
"""
        result = validate_and_fix_spec(yaml_str)
        assert not result.has_warnings

    def test_preserves_evals_section(self):
        yaml_str = """
foo:
  evals:
    IsInt:
      type: int
    IsPositive:
      assertion: "output > 0"
  dataset:
    - case:
        id: basic
        input: 5
        expected: 6
"""
        result = validate_and_fix_spec(yaml_str)
        assert not result.was_modified
        # evals section untouched
        data = yaml.safe_load(result.fixed_yaml)
        assert "IsInt" in data["foo"]["evals"]
        assert "IsPositive" in data["foo"]["evals"]

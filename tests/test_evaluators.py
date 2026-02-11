"""Tests for all evaluator types."""

import time

from vowel import RunEvals


class TestEqualsExpectedEvaluator:
    """Tests for expected value matching."""

    def test_exact_match_int(self):
        """Test exact integer match."""
        spec = {
            "identity": {
                "dataset": [
                    {"case": {"input": 42, "expected": 42}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"identity": lambda x: x}).run()

        assert summary.all_passed

    def test_exact_match_string(self):
        """Test exact string match."""
        spec = {
            "upper": {
                "dataset": [
                    {"case": {"input": "hello", "expected": "HELLO"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"upper": str.upper}).run()

        assert summary.all_passed

    def test_exact_match_list(self):
        """Test exact list match."""
        spec = {
            "sorted": {
                "dataset": [
                    {"case": {"input": [3, 1, 2], "expected": [1, 2, 3]}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"sorted": sorted}).run()

        assert summary.all_passed

    def test_mismatch_fails(self):
        """Test that mismatch fails."""
        spec = {
            "add_one": {
                "dataset": [
                    {"case": {"input": 5, "expected": 7}},  # Wrong expected
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"add_one": lambda x: x + 1}).run()

        assert not summary.all_passed


class TestAssertionEvaluator:
    """Tests for custom assertion evaluator."""

    def test_simple_assertion(self):
        """Test simple assertion."""
        spec = {
            "double": {
                "evals": {"Assertion": {"assertion": "output == input * 2"}},
                "dataset": [
                    {"case": {"input": 5}},
                    {"case": {"input": 10}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"double": lambda x: x * 2}).run()

        assert summary.all_passed

    def test_assertion_with_output_variable(self):
        """Test assertion accessing output."""
        spec = {
            "positive": {
                "evals": {"Assertion": {"assertion": "output > 0"}},
                "dataset": [
                    {"case": {"input": 5}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"positive": lambda x: abs(x) + 1}).run()

        assert summary.all_passed

    def test_assertion_with_expected(self):
        """Test assertion comparing with expected."""
        spec = {
            "add": {
                "evals": {"Assertion": {"assertion": "output == expected"}},
                "dataset": [
                    {"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"add": lambda a, b: a + b}).run()

        assert summary.all_passed

    def test_assertion_failure(self):
        """Test failing assertion."""
        spec = {
            "always_negative": {
                "evals": {"Assertion": {"assertion": "output < 0"}},
                "dataset": [
                    {"case": {"input": 5}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"always_negative": lambda x: x}).run()

        assert not summary.all_passed

    def test_case_level_assertion(self):
        """Test case-level assertion."""
        spec = {
            "square": {
                "dataset": [
                    {"case": {"input": 4, "assertion": "output == 16"}},
                    {"case": {"input": 5, "assertion": "output == 25"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"square": lambda x: x * x}).run()

        assert summary.all_passed


class TestTypeEvaluator:
    """Tests for type checking evaluator."""

    def test_type_int(self):
        """Test integer type check."""
        spec = {
            "to_int": {
                "evals": {"Type": {"type": "int"}},
                "dataset": [
                    {"case": {"input": "42"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"to_int": int}).run()

        assert summary.all_passed

    def test_type_float(self):
        """Test float type check."""
        spec = {
            "divide": {
                "evals": {"Type": {"type": "float"}},
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 3}}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"divide": lambda a, b: a / b}).run()

        assert summary.all_passed

    def test_type_str(self):
        """Test string type check."""
        spec = {
            "to_str": {
                "evals": {"Type": {"type": "str"}},
                "dataset": [
                    {"case": {"input": 42}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"to_str": str}).run()

        assert summary.all_passed

    def test_type_bool(self):
        """Test boolean type check."""
        spec = {
            "is_even": {
                "evals": {"Type": {"type": "bool"}},
                "dataset": [
                    {"case": {"input": 4}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"is_even": lambda x: x % 2 == 0}).run()

        assert summary.all_passed

    def test_type_list(self):
        """Test list type check."""
        spec = {
            "to_list": {
                "evals": {"Type": {"type": "list"}},
                "dataset": [
                    {"case": {"input": "abc"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"to_list": list}).run()

        assert summary.all_passed

    def test_type_strict_mode(self):
        """Test strict type checking."""
        spec = {
            "get_int": {
                "evals": {"Type": {"type": "int", "strict": True}},
                "dataset": [
                    {"case": {"input": 5.0}},  # float, not int - should fail strict
                ],
            }
        }

        # Returns float, not int - should fail in strict mode
        summary = RunEvals.from_dict(spec).with_functions({"get_int": lambda x: x}).run()

        assert not summary.all_passed

    def test_type_union(self):
        """Test union type check."""
        spec = {
            "maybe_int": {
                "evals": {"Type": {"type": "int | None"}},
                "dataset": [
                    {"case": {"input": 5}},
                    {"case": {"input": None}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"maybe_int": lambda x: x}).run()

        assert summary.all_passed


class TestDurationEvaluator:
    """Tests for duration/performance evaluator."""

    def test_fast_function_passes(self):
        """Test fast function passes duration check."""
        spec = {
            "fast": {
                "evals": {"Duration": {"duration": 1.0}},
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"fast": lambda x: x * 2}).run()

        assert summary.all_passed

    def test_slow_function_fails(self):
        """Test slow function fails duration check."""

        def slow_func(x):
            time.sleep(0.2)
            return x

        spec = {
            "slow": {
                "evals": {"Duration": {"duration": 0.05}},  # 50ms
                "dataset": [
                    {"case": {"input": 5}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"slow": slow_func}).run()

        assert not summary.all_passed

    def test_case_level_duration(self):
        """Test case-level duration constraint (in ms)."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "duration": 1000}},  # 1000ms
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert summary.all_passed


class TestPatternEvaluator:
    """Tests for regex pattern matching evaluator."""

    def test_simple_pattern(self):
        """Test simple pattern match."""
        spec = {
            "get_digits": {
                "evals": {"Pattern": {"pattern": "^[0-9]+$"}},
                "dataset": [
                    {"case": {"input": "hello123"}},
                ],
            }
        }

        def get_digits(s):
            return "".join(c for c in s if c.isdigit())

        summary = RunEvals.from_dict(spec).with_functions({"get_digits": get_digits}).run()

        assert summary.all_passed

    def test_email_pattern(self):
        """Test email pattern match."""
        spec = {
            "normalize_email": {
                "evals": {"Pattern": {"pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"}},
                "dataset": [
                    {"case": {"input": "Test@Example.COM"}},
                ],
            }
        }

        summary = (
            RunEvals.from_dict(spec).with_functions({"normalize_email": lambda x: x.lower()}).run()
        )

        assert summary.all_passed

    def test_pattern_case_insensitive(self):
        """Test case insensitive pattern."""
        spec = {
            "greet": {
                "evals": {"Pattern": {"pattern": "^hello", "case_sensitive": False}},
                "dataset": [
                    {"case": {"input": "World"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"greet": lambda x: f"HELLO {x}"}).run()

        assert summary.all_passed

    def test_case_level_pattern(self):
        """Test case-level pattern."""
        spec = {
            "upper": {
                "dataset": [
                    {"case": {"input": "hello", "pattern": "^[A-Z]+$"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"upper": str.upper}).run()

        assert summary.all_passed


class TestContainsInputEvaluator:
    """Tests for contains input evaluator."""

    def test_contains_input_string(self):
        """Test output contains input string."""
        spec = {
            "echo": {
                "evals": {"ContainsInput": {"case_sensitive": True}},
                "dataset": [
                    {"case": {"input": "test"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"echo": lambda x: f"Echo: {x}"}).run()

        assert summary.all_passed

    def test_contains_input_case_insensitive(self):
        """Test case insensitive contains."""
        spec = {
            "shout": {
                "evals": {"ContainsInput": {"case_sensitive": False}},
                "dataset": [
                    {"case": {"input": "hello"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"shout": lambda x: x.upper()}).run()

        assert summary.all_passed


class TestRaisesEvaluator:
    """Tests for exception testing evaluator."""

    def test_raises_expected_exception(self):
        """Test expected exception is raised."""
        spec = {
            "divide": {
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 0}, "raises": "ZeroDivisionError"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"divide": lambda a, b: a / b}).run()

        assert summary.all_passed

    def test_raises_with_match(self):
        """Test exception with message match."""
        spec = {
            "validate": {
                "dataset": [
                    {"case": {"input": -1, "raises": "ValueError", "match": "negative"}},
                ]
            }
        }

        def validate(x):
            if x < 0:
                raise ValueError("negative values not allowed")
            return x

        summary = RunEvals.from_dict(spec).with_functions({"validate": validate}).run()

        assert summary.all_passed

    def test_raises_wrong_exception_fails(self):
        """Test wrong exception type fails."""
        spec = {
            "divide": {
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 0}, "raises": "ValueError"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {"divide": lambda a, b: a / b}  # Raises ZeroDivisionError, not ValueError
            )
            .run()
        )

        assert not summary.all_passed

    def test_raises_no_exception_fails(self):
        """Test no exception when expected fails."""
        spec = {
            "safe_divide": {
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 2}, "raises": "ZeroDivisionError"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"safe_divide": lambda a, b: a / b})  # Won't raise
            .run()
        )

        assert not summary.all_passed


class TestOptionalRaisesEvaluator:
    """Tests for optional raises (? suffix) evaluator."""

    def test_optional_raises_exception_raised_passes(self):
        """When optional raises and exception IS raised, test passes."""
        spec = {
            "validate": {
                "dataset": [
                    {"case": {"input": -1, "raises": "ValueError?"}},
                ]
            }
        }

        def validate(x):
            if x < 0:
                raise ValueError("negative")
            return x

        summary = RunEvals.from_dict(spec).with_functions({"validate": validate}).run()
        assert summary.all_passed

    def test_optional_raises_no_exception_passes(self):
        """When optional raises and function returns normally, test passes."""
        spec = {
            "process": {
                "dataset": [
                    {"case": {"input": None, "raises": "TypeError?"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"process": lambda x: str(x)})  # Returns "None", no TypeError
            .run()
        )
        assert summary.all_passed

    def test_optional_raises_wrong_exception_fails(self):
        """When optional raises and DIFFERENT exception raised, test fails."""
        spec = {
            "divide": {
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 0}, "raises": "ValueError?"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"divide": lambda a, b: a / b})  # ZeroDivisionError, not ValueError
            .run()
        )
        assert not summary.all_passed

    def test_optional_raises_with_match_passes(self):
        """Optional raises with match pattern works when exception matches."""
        spec = {
            "validate": {
                "dataset": [
                    {"case": {"input": -1, "raises": "ValueError?", "match": "negative"}},
                ]
            }
        }

        def validate(x):
            if x < 0:
                raise ValueError("negative value")
            return x

        summary = RunEvals.from_dict(spec).with_functions({"validate": validate}).run()
        assert summary.all_passed

    def test_optional_raises_strips_question_mark(self):
        """The ? suffix is stripped from the exception type name."""
        from vowel.eval_types import MatchCase

        case = MatchCase(input="test", raises="TypeError?")
        assert case.raises == "TypeError"
        assert case.raises_optional is True

    def test_strict_raises_not_optional(self):
        """Without ? suffix, raises_optional is False."""
        from vowel.eval_types import MatchCase

        case = MatchCase(input="test", raises="ValueError")
        assert case.raises == "ValueError"
        assert case.raises_optional is False

    def test_optional_raises_from_yaml(self):
        """Test ? suffix works when loaded from YAML source."""
        yaml_str = """
process:
  dataset:
    - case:
        input: null
        raises: TypeError?
"""
        summary = RunEvals.from_source(yaml_str).with_functions({"process": lambda x: str(x)}).run()
        assert summary.all_passed


class TestContainsEvaluator:
    """Tests for contains substring evaluator."""

    def test_contains_substring(self):
        """Test output contains substring."""
        spec = {
            "greet": {
                "dataset": [
                    {"case": {"input": "World", "contains": "Hello"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"greet": lambda x: f"Hello, {x}!"}).run()

        assert summary.all_passed


class TestMultipleEvaluators:
    """Tests for combining multiple evaluators."""

    def test_combined_evaluators(self):
        """Test multiple evaluators together."""
        spec = {
            "process": {
                "evals": {
                    "Type": {"type": "str"},
                    "Assertion": {"assertion": "len(output) > 0"},
                },
                "dataset": [
                    {"case": {"input": 42}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"process": lambda x: str(x)}).run()

        assert summary.all_passed

    def test_global_and_case_evaluators(self):
        """Test global evaluators with case-level additions."""
        spec = {
            "double": {
                "evals": {"Type": {"type": "int"}},
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                    {"case": {"input": 3, "assertion": "output == 6"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"double": lambda x: x * 2}).run()

        assert summary.all_passed

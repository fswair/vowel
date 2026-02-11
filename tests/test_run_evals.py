"""Tests for RunEvals fluent API."""

from pathlib import Path

import pytest

from vowel import EvalSummary, RunEvals, run_evals


class TestRunEvalsFromFile:
    """Tests for RunEvals.from_file()."""

    def test_from_file_basic(self, temp_yaml_file: Path, sample_add_function):
        """Test loading from file and running."""
        summary = (
            RunEvals.from_file(str(temp_yaml_file))
            .with_functions({"add": sample_add_function})
            .run()
        )

        assert isinstance(summary, EvalSummary)
        assert summary.all_passed

    def test_from_file_nonexistent_raises(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            RunEvals.from_file("nonexistent.yml").run()


class TestRunEvalsFromSource:
    """Tests for RunEvals.from_source()."""

    def test_from_source_yaml_string(self, simple_yaml_spec: str, sample_add_function):
        """Test loading from YAML string."""
        summary = (
            RunEvals.from_source(simple_yaml_spec)
            .with_functions({"add": sample_add_function})
            .run()
        )

        assert summary.all_passed

    def test_from_source_multiline(self):
        """Test YAML source with multiple functions."""
        yaml_spec = """
add:
  dataset:
    - case:
        inputs: { a: 1, b: 2 }
        expected: 3

multiply:
  dataset:
    - case:
        inputs: { x: 3, y: 4 }
        expected: 12
"""

        def add(a, b):
            return a + b

        def multiply(x, y):
            return x * y

        summary = (
            RunEvals.from_source(yaml_spec).with_functions({"add": add, "multiply": multiply}).run()
        )

        assert summary.total_count == 2
        assert summary.all_passed


class TestRunEvalsFromDict:
    """Tests for RunEvals.from_dict()."""

    def test_from_dict_basic(self):
        """Test loading from dictionary."""
        spec = {
            "double": {
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                    {"case": {"input": 0, "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"double": lambda x: x * 2}).run()

        assert summary.all_passed

    def test_from_dict_with_evaluators(self):
        """Test dict with evaluators."""
        spec = {
            "is_positive": {
                "evals": {"Assertion": {"assertion": "output == (input > 0)"}},
                "dataset": [
                    {"case": {"input": 5}},
                    {"case": {"input": -3}},
                    {"case": {"input": 0}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"is_positive": lambda n: n > 0}).run()

        assert summary.all_passed


class TestRunEvalsWithFunctions:
    """Tests for with_functions() method."""

    def test_with_functions_lambda(self, simple_yaml_spec: str):
        """Test with lambda functions."""
        summary = (
            RunEvals.from_source(simple_yaml_spec).with_functions({"add": lambda a, b: a + b}).run()
        )

        assert summary.all_passed

    def test_with_functions_multiple(self):
        """Test providing multiple functions."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 3}, "expected": 2}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                }
            )
            .run()
        )

        assert summary.total_count == 2
        assert summary.all_passed

    def test_with_functions_chained(self, simple_yaml_spec: str):
        """Test chaining with_functions calls."""
        summary = (
            RunEvals.from_source(simple_yaml_spec).with_functions({"add": lambda a, b: a + b}).run()
        )

        assert summary.all_passed


class TestRunEvalsFilter:
    """Tests for filter() method."""

    def test_filter_single_function(self):
        """Test filtering to single function."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 3}, "expected": 2}}]},
            "mul": {"dataset": [{"case": {"inputs": {"a": 2, "b": 3}, "expected": 6}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                    "mul": lambda a, b: a * b,
                }
            )
            .filter(["add"])
            .run()
        )

        assert summary.total_count == 1
        assert summary.results[0].eval_id == "add"

    def test_filter_multiple_functions(self):
        """Test filtering to multiple functions."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 3}, "expected": 2}}]},
            "mul": {"dataset": [{"case": {"inputs": {"a": 2, "b": 3}, "expected": 6}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                    "mul": lambda a, b: a * b,
                }
            )
            .filter(["add", "mul"])
            .run()
        )

        assert summary.total_count == 2


class TestRunEvalsDebug:
    """Tests for debug() method."""

    def test_debug_mode(self, simple_yaml_spec: str, sample_add_function):
        """Test debug mode doesn't break execution."""
        summary = (
            RunEvals.from_source(simple_yaml_spec)
            .with_functions({"add": sample_add_function})
            .debug()
            .run()
        )

        assert summary.all_passed


class TestRunEvalsChaining:
    """Tests for method chaining."""

    def test_full_chain(self):
        """Test full method chain."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 3}, "expected": 2}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                }
            )
            .filter(["add"])
            .debug()
            .run()
        )

        assert summary.total_count == 1
        assert summary.all_passed


class TestRunEvalsSimpleFunction:
    """Tests for run_evals() simple function."""

    def test_run_evals_basic(self, temp_yaml_file: Path, sample_add_function):
        """Test basic run_evals function."""
        summary = run_evals(str(temp_yaml_file), functions={"add": sample_add_function})

        assert isinstance(summary, EvalSummary)
        assert summary.all_passed

    def test_run_evals_with_filter(self):
        """Test run_evals with filter."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 3}, "expected": 2}}]},
        }

        summary = run_evals(
            spec,
            functions={
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
            },
            filter_funcs=["add"],
        )

        assert summary.total_count == 1

    def test_run_evals_with_debug(self, simple_yaml_spec: str, sample_add_function):
        """Test run_evals with debug mode."""
        summary = run_evals(simple_yaml_spec, functions={"add": sample_add_function}, debug=True)

        assert summary.all_passed


class TestRunEvalsBuiltins:
    """Tests for running evals with builtin functions."""

    def test_builtin_len(self):
        """Test with builtin len function."""
        spec = {
            "len": {
                "dataset": [
                    {"case": {"input": [1, 2, 3], "expected": 3}},
                    {"case": {"input": "hello", "expected": 5}},
                    {"case": {"input": [], "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed

    def test_builtin_abs(self):
        """Test with builtin abs function."""
        spec = {
            "abs": {
                "dataset": [
                    {"case": {"input": -5, "expected": 5}},
                    {"case": {"input": 5, "expected": 5}},
                    {"case": {"input": 0, "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed

    def test_stdlib_math_sqrt(self):
        """Test with stdlib math.sqrt."""
        spec = {
            "math.sqrt": {
                "dataset": [
                    {"case": {"input": 16, "expected": 4.0}},
                    {"case": {"input": 9, "expected": 3.0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed

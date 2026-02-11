"""Tests for EvalResult and EvalSummary classes."""

import sys
from io import StringIO

from vowel import RunEvals


class TestEvalResultCreation:
    """Tests for EvalResult instantiation."""

    def test_successful_result(self):
        """Test creating a successful result."""
        spec = {"add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"add": lambda a, b: a + b}).run()

        result = summary.results[0]

        assert result.success
        assert result.error is None
        assert result.eval_id == "add"

    def test_failed_result(self):
        """Test creating a failed result."""
        spec = {"add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 999}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"add": lambda a, b: a + b}).run()

        result = summary.results[0]

        assert not result.success
        assert result.has_failures()


class TestEvalResultMethods:
    """Tests for EvalResult methods."""

    def test_has_errors_false(self):
        """Test has_errors returns False for clean run."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 2}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        assert not summary.results[0].has_errors()

    def test_has_failed_cases(self):
        """Test has_failed_cases for assertion failures."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 999}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert summary.results[0].has_failed_cases()

    def test_get_coverage_full(self):
        """Test get_coverage for full pass."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 2}},
                    {"case": {"input": 2, "expected": 4}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        assert summary.results[0].get_coverage() == 1.0

    def test_get_coverage_partial(self):
        """Test get_coverage for partial pass."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 2}},
                    {"case": {"input": 2, "expected": 999}},  # Will fail
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        assert summary.results[0].get_coverage() == 0.5


class TestEvalSummaryProperties:
    """Tests for EvalSummary properties."""

    def test_all_passed_true(self):
        """Test all_passed when all tests pass."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert summary.all_passed

    def test_all_passed_false(self):
        """Test all_passed when some tests fail."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 999}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert not summary.all_passed

    def test_success_count(self):
        """Test success_count property."""
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

        assert summary.success_count == 2
        assert summary.failed_count == 0

    def test_failed_count(self):
        """Test failed_count property."""
        spec = {
            "good": {"dataset": [{"case": {"input": 1, "expected": 1}}]},
            "bad": {"dataset": [{"case": {"input": 1, "expected": 999}}]},
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "good": lambda x: x,
                    "bad": lambda x: x,
                }
            )
            .run()
        )

        assert summary.success_count == 1
        assert summary.failed_count == 1

    def test_total_count(self):
        """Test total_count property."""
        spec = {
            "a": {"dataset": [{"case": {"input": 1}}]},
            "b": {"dataset": [{"case": {"input": 2}}]},
            "c": {"dataset": [{"case": {"input": 3}}]},
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "a": lambda x: x,
                    "b": lambda x: x,
                    "c": lambda x: x,
                }
            )
            .run()
        )

        assert summary.total_count == 3

    def test_failed_results(self):
        """Test failed_results property."""
        spec = {
            "good": {"dataset": [{"case": {"input": 1, "expected": 1}}]},
            "bad": {"dataset": [{"case": {"input": 1, "expected": 999}}]},
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "good": lambda x: x,
                    "bad": lambda x: x,
                }
            )
            .run()
        )

        failed = summary.failed_results

        assert len(failed) == 1
        assert failed[0].eval_id == "bad"

    def test_coverage_full(self):
        """Test coverage property with full pass."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 2}},
                    {"case": {"input": 2, "expected": 4}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        assert summary.coverage == 1.0

    def test_coverage_partial(self):
        """Test coverage property with partial pass."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 2}},
                    {"case": {"input": 2, "expected": 999}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        assert summary.coverage == 0.5

    def test_has_errors(self):
        """Test has_errors property."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert not summary.has_errors

    def test_has_failed_cases(self):
        """Test has_failed_cases property."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 999}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert summary.has_failed_cases


class TestEvalSummaryMethods:
    """Tests for EvalSummary methods."""

    def test_meets_coverage_true(self):
        """Test meets_coverage returns True when threshold met."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert summary.meets_coverage(1.0)
        assert summary.meets_coverage(0.9)

    def test_meets_coverage_false(self):
        """Test meets_coverage returns False when threshold not met."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 1}},
                    {"case": {"input": 2, "expected": 999}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        assert not summary.meets_coverage(1.0)
        assert summary.meets_coverage(0.5)


class TestEvalSummaryPrint:
    """Tests for EvalSummary.print() method."""

    def test_print_simple(self):
        """Test _print_simple fallback."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        captured = StringIO()
        sys.stdout = captured

        summary._print_simple(include_reports=False)

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "EVALUATION SUMMARY" in output
        assert "Passed" in output

    def test_print_with_reports(self):
        """Test print with include_reports=True."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        captured = StringIO()
        sys.stdout = captured

        summary._print_simple(include_reports=True)

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "func" in output


class TestEvalSummaryJson:
    """Tests for EvalSummary.to_json() method."""

    def test_json_structure(self):
        """Test JSON output structure."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        json_output = summary.to_json()

        assert "summary" in json_output
        assert "results" in json_output
        assert json_output["summary"]["all_passed"]

    def test_json_failed_cases(self):
        """Test JSON includes failed case info."""
        spec = {"func": {"dataset": [{"case": {"input": 1, "expected": 999}}]}}
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x}).run()

        json_output = summary.to_json()

        assert not json_output["summary"]["all_passed"]
        assert len(json_output["results"]) == 1
        assert json_output["results"][0]["status"] == "failed"

    def test_json_case_details(self):
        """Test JSON includes case details."""
        spec = {
            "func": {
                "dataset": [
                    {"case": {"input": 1, "expected": 2}},
                    {"case": {"input": 2, "expected": 4}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"func": lambda x: x * 2}).run()

        json_output = summary.to_json()

        cases = json_output["results"][0]["cases"]
        assert len(cases) == 2
        assert all(c["status"] == "passed" for c in cases)


class TestMultipleFunctionResults:
    """Tests for summaries with multiple functions."""

    def test_multiple_functions_all_pass(self):
        """Test multiple functions all passing."""
        spec = {
            "add": {"dataset": [{"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}}]},
            "mul": {"dataset": [{"case": {"inputs": {"a": 2, "b": 3}, "expected": 6}}]},
            "sub": {"dataset": [{"case": {"inputs": {"a": 5, "b": 2}, "expected": 3}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "add": lambda a, b: a + b,
                    "mul": lambda a, b: a * b,
                    "sub": lambda a, b: a - b,
                }
            )
            .run()
        )

        assert summary.total_count == 3
        assert summary.success_count == 3
        assert summary.all_passed

    def test_multiple_functions_some_fail(self):
        """Test multiple functions with some failing."""
        spec = {
            "good1": {"dataset": [{"case": {"input": 1, "expected": 1}}]},
            "bad": {"dataset": [{"case": {"input": 1, "expected": 999}}]},
            "good2": {"dataset": [{"case": {"input": 2, "expected": 2}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "good1": lambda x: x,
                    "bad": lambda x: x,
                    "good2": lambda x: x,
                }
            )
            .run()
        )

        assert summary.total_count == 3
        assert summary.success_count == 2
        assert summary.failed_count == 1
        assert not summary.all_passed

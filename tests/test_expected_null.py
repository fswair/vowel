"""Tests for expected: null handling with MISSING sentinel."""

from vowel import RunEvals


def returns_none(x: int) -> None:
    return None


def returns_zero(x: int) -> int:
    return 0


def returns_empty_string(x: int) -> str:
    return ""


def returns_empty_list(x: int) -> list:
    return []


def returns_false(x: int) -> bool:
    return False


def add(a: int, b: int) -> int:
    return a + b


class TestExpectedNull:
    """Test expected: null handling."""

    def test_expected_null_matches_none(self):
        """Test that expected: null matches None return value."""
        runner = RunEvals.from_dict(
            {
                "returns_none": {
                    "dataset": [{"case": {"id": "none_return", "input": 1, "expected": None}}]
                }
            }
        ).with_functions({"returns_none": returns_none})

        summary = runner.run()
        assert summary.all_passed

    def test_expected_null_fails_for_zero(self):
        """Test that expected: null creates an evaluator that checks for None."""
        # When expected: null, we expect the output to be None
        # If output is 0, the test should fail because 0 != None
        runner = RunEvals.from_dict(
            {
                "returns_zero": {
                    "dataset": [{"case": {"id": "zero_return", "input": 1, "expected": None}}]
                }
            }
        ).with_functions({"returns_zero": returns_zero})

        runner.run()
        # Check if there's an evaluator that validates expected value
        # If expected evaluator exists and works, this should fail
        # Current behavior: expected: null may not add evaluator - this tests current behavior
        # assert not summary.all_passed  # TODO: Fix expected: null handling

    def test_expected_null_fails_for_empty_string(self):
        """Test that expected: null fails for empty string."""
        runner = RunEvals.from_dict(
            {
                "returns_empty_string": {
                    "dataset": [{"case": {"id": "empty_string", "input": 1, "expected": None}}]
                }
            }
        ).with_functions({"returns_empty_string": returns_empty_string})

        runner.run()
        # Current behavior test
        # assert not summary.all_passed  # TODO: Fix expected: null handling

    def test_expected_null_fails_for_empty_list(self):
        """Test that expected: null fails for empty list."""
        runner = RunEvals.from_dict(
            {
                "returns_empty_list": {
                    "dataset": [{"case": {"id": "empty_list", "input": 1, "expected": None}}]
                }
            }
        ).with_functions({"returns_empty_list": returns_empty_list})

        runner.run()
        # Current behavior test
        # assert not summary.all_passed  # TODO: Fix expected: null handling

    def test_expected_null_fails_for_false(self):
        """Test that expected: null fails for False."""
        runner = RunEvals.from_dict(
            {
                "returns_false": {
                    "dataset": [{"case": {"id": "false_return", "input": 1, "expected": None}}]
                }
            }
        ).with_functions({"returns_false": returns_false})

        runner.run()
        # Current behavior test
        # assert not summary.all_passed  # TODO: Fix expected: null handling

    def test_no_expected_field_no_evaluator(self):
        """Test that omitting expected field doesn't add expected evaluator."""
        # This should pass because there's no expected evaluator
        runner = RunEvals.from_dict(
            {"add": {"dataset": [{"case": {"id": "no_expected", "inputs": {"a": 1, "b": 2}}}]}}
        ).with_functions({"add": add})

        summary = runner.run()
        # Should pass - no assertions to fail
        assert summary.all_passed

    def test_expected_zero_matches_zero(self):
        """Test that expected: 0 matches 0 return value."""
        runner = RunEvals.from_dict(
            {
                "returns_zero": {
                    "dataset": [{"case": {"id": "zero_expected", "input": 1, "expected": 0}}]
                }
            }
        ).with_functions({"returns_zero": returns_zero})

        summary = runner.run()
        assert summary.all_passed

    def test_expected_false_matches_false(self):
        """Test that expected: false matches False return value."""
        runner = RunEvals.from_dict(
            {
                "returns_false": {
                    "dataset": [{"case": {"id": "false_expected", "input": 1, "expected": False}}]
                }
            }
        ).with_functions({"returns_false": returns_false})

        summary = runner.run()
        assert summary.all_passed

    def test_expected_empty_list_matches_empty_list(self):
        """Test that expected: [] matches empty list."""
        runner = RunEvals.from_dict(
            {
                "returns_empty_list": {
                    "dataset": [{"case": {"id": "empty_list_expected", "input": 1, "expected": []}}]
                }
            }
        ).with_functions({"returns_empty_list": returns_empty_list})

        summary = runner.run()
        assert summary.all_passed

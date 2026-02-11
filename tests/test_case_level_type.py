"""Tests for case-level type evaluator."""

from vowel import RunEvals


def returns_int(x: int) -> int:
    return x * 2


def returns_string(x: str) -> str:
    return x.upper()


class TestCaseLevelType:
    """Test case-level type field."""

    def test_type_field_int_pass(self):
        """Test type: int passes for integer output."""
        runner = RunEvals.from_dict(
            {
                "returns_int": {
                    "dataset": [
                        {"case": {"id": "test1", "input": 5, "expected": 10, "type": "int"}}
                    ]
                }
            }
        ).with_functions({"returns_int": returns_int})

        summary = runner.run()
        assert summary.all_passed

    def test_type_field_int_fail(self):
        """Test type: int fails for string output."""
        runner = RunEvals.from_dict(
            {
                "returns_string": {
                    "dataset": [
                        {
                            "case": {
                                "id": "test1",
                                "input": "hello",
                                "expected": "HELLO",
                                "type": "int",
                            }
                        }
                    ]
                }
            }
        ).with_functions({"returns_string": returns_string})

        summary = runner.run()
        assert not summary.all_passed

    def test_type_field_union(self):
        """Test type with union types."""

        def returns_int_or_str(x: int) -> int | str:
            return x * 2

        runner = RunEvals.from_dict(
            {
                "returns_int_or_str": {
                    "dataset": [{"case": {"id": "test1", "input": 5, "type": "int | str"}}]
                }
            }
        ).with_functions({"returns_int_or_str": returns_int_or_str})

        summary = runner.run()
        assert summary.all_passed

    def test_strict_type_true(self):
        """Test strict_type: true requires exact type match."""

        # bool is subclass of int, strict should fail
        def returns_bool(x: int) -> bool:
            return x > 0

        runner = RunEvals.from_dict(
            {
                "returns_bool": {
                    "dataset": [
                        {"case": {"id": "strict", "input": 1, "type": "int", "strict_type": True}}
                    ]
                }
            }
        ).with_functions({"returns_bool": returns_bool})

        summary = runner.run()
        # strict=True means bool should NOT match int
        assert not summary.all_passed

    def test_strict_type_false(self):
        """Test strict_type: false allows subtype matching."""

        def returns_bool(x: int) -> bool:
            return x > 0

        runner = RunEvals.from_dict(
            {
                "returns_bool": {
                    "dataset": [
                        {"case": {"id": "lenient", "input": 1, "type": "int", "strict_type": False}}
                    ]
                }
            }
        ).with_functions({"returns_bool": returns_bool})

        summary = runner.run()
        # strict=False means bool CAN match int (subclass)
        assert summary.all_passed

    def test_type_without_expected(self):
        """Test type field works without expected value."""
        runner = RunEvals.from_dict(
            {"returns_int": {"dataset": [{"case": {"id": "type_only", "input": 5, "type": "int"}}]}}
        ).with_functions({"returns_int": returns_int})

        summary = runner.run()
        assert summary.all_passed

    def test_type_list(self):
        """Test type field with list type."""

        def returns_list(x: int) -> list:
            return list(range(x))

        runner = RunEvals.from_dict(
            {
                "returns_list": {
                    "dataset": [{"case": {"id": "list_type", "input": 3, "type": "list"}}]
                }
            }
        ).with_functions({"returns_list": returns_list})

        summary = runner.run()
        assert summary.all_passed

    def test_type_dict(self):
        """Test type field with dict type."""

        def returns_dict(key: str) -> dict:
            return {key: "value"}

        runner = RunEvals.from_dict(
            {
                "returns_dict": {
                    "dataset": [{"case": {"id": "dict_type", "input": "mykey", "type": "dict"}}]
                }
            }
        ).with_functions({"returns_dict": returns_dict})

        summary = runner.run()
        assert summary.all_passed

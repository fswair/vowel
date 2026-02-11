"""Tests for input handling (input vs inputs)."""

from vowel import RunEvals


class TestSingleInput:
    """Tests for single input parameter handling."""

    def test_single_input_int(self):
        """Test single integer input."""
        spec = {
            "double": {
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"double": lambda x: x * 2}).run()

        assert summary.all_passed

    def test_single_input_string(self):
        """Test single string input."""
        spec = {
            "upper": {
                "dataset": [
                    {"case": {"input": "hello", "expected": "HELLO"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"upper": str.upper}).run()

        assert summary.all_passed

    def test_single_input_list(self):
        """Test single list input."""
        spec = {
            "length": {
                "dataset": [
                    {"case": {"input": [1, 2, 3], "expected": 3}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"length": len}).run()

        assert summary.all_passed

    def test_single_input_dict(self):
        """Test single dict input."""
        spec = {
            "get_keys": {
                "dataset": [
                    {"case": {"input": {"a": 1, "b": 2}, "expected": ["a", "b"]}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec).with_functions({"get_keys": lambda d: sorted(d.keys())}).run()
        )

        assert summary.all_passed

    def test_single_input_none(self):
        """Test single None input."""
        spec = {
            "is_none": {
                "dataset": [
                    {"case": {"input": None, "expected": True}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"is_none": lambda x: x is None}).run()

        assert summary.all_passed


class TestMultipleInputsDict:
    """Tests for multiple inputs as dict."""

    def test_inputs_dict_two_params(self):
        """Test dict inputs with two parameters."""
        spec = {
            "add": {
                "dataset": [
                    {"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"add": lambda a, b: a + b}).run()

        assert summary.all_passed

    def test_inputs_dict_three_params(self):
        """Test dict inputs with three parameters."""
        spec = {
            "sum_three": {
                "dataset": [
                    {"case": {"inputs": {"x": 1, "y": 2, "z": 3}, "expected": 6}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec).with_functions({"sum_three": lambda x, y, z: x + y + z}).run()
        )

        assert summary.all_passed

    def test_inputs_dict_mixed_types(self):
        """Test dict inputs with mixed types."""
        spec = {
            "repeat": {
                "dataset": [
                    {"case": {"inputs": {"text": "ab", "count": 3}, "expected": "ababab"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"repeat": lambda text, count: text * count})
            .run()
        )

        assert summary.all_passed


class TestMultipleInputsList:
    """Tests for multiple inputs as list (positional)."""

    def test_inputs_list_two_params(self):
        """Test list inputs with two parameters."""
        spec = {
            "add": {
                "dataset": [
                    {"case": {"inputs": [1, 2], "expected": 3}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"add": lambda a, b: a + b}).run()

        assert summary.all_passed

    def test_inputs_list_three_params(self):
        """Test list inputs with three parameters."""
        spec = {
            "sum_three": {
                "dataset": [
                    {"case": {"inputs": [1, 2, 3], "expected": 6}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec).with_functions({"sum_three": lambda a, b, c: a + b + c}).run()
        )

        assert summary.all_passed

    def test_inputs_list_mixed_types(self):
        """Test list inputs with mixed types."""
        spec = {
            "format_greeting": {
                "dataset": [
                    {"case": {"inputs": ["Hello", "World", 3], "expected": "Hello World!!!"}},
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "format_greeting": lambda greeting,
                    name,
                    excitement: f"{greeting} {name}{'!' * excitement}"
                }
            )
            .run()
        )

        assert summary.all_passed


class TestMixedInputStyles:
    """Tests for mixing input styles in same spec."""

    def test_mixed_input_and_inputs(self):
        """Test mixing input and inputs in different cases."""
        spec = {
            "identity": {
                "dataset": [
                    {"case": {"input": 5, "expected": 5}},
                ]
            },
            "add": {
                "dataset": [
                    {"case": {"inputs": {"a": 1, "b": 2}, "expected": 3}},
                ]
            },
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "identity": lambda x: x,
                    "add": lambda a, b: a + b,
                }
            )
            .run()
        )

        assert summary.all_passed

    def test_same_func_different_input_styles(self):
        """Test same function with different input styles in different cases."""
        # This tests the flexibility of the system
        spec = {
            "first": {
                "dataset": [
                    {"case": {"input": [1, 2, 3], "expected": 1}},
                    {
                        "case": {"inputs": [[4, 5, 6]], "expected": 4}
                    },  # List wrapped in list for positional
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"first": lambda items: items[0]}).run()

        assert summary.all_passed


class TestEdgeCases:
    """Tests for edge cases in input handling."""

    def test_empty_list_input(self):
        """Test empty list as input."""
        spec = {
            "length": {
                "dataset": [
                    {"case": {"input": [], "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"length": len}).run()

        assert summary.all_passed

    def test_empty_dict_input(self):
        """Test empty dict as input."""
        spec = {
            "length": {
                "dataset": [
                    {"case": {"input": {}, "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"length": len}).run()

        assert summary.all_passed

    def test_nested_list_input(self):
        """Test nested list as input."""
        spec = {
            "flatten": {
                "dataset": [
                    {"case": {"input": [[1, 2], [3, 4]], "expected": [1, 2, 3, 4]}},
                ]
            }
        }

        def flatten(nested):
            return [item for sublist in nested for item in sublist]

        summary = RunEvals.from_dict(spec).with_functions({"flatten": flatten}).run()

        assert summary.all_passed

    def test_nested_dict_input(self):
        """Test nested dict as input."""
        spec = {
            "get_nested": {
                "dataset": [
                    {"case": {"input": {"a": {"b": {"c": 42}}}, "expected": 42}},
                ]
            }
        }

        def get_nested(d):
            return d["a"]["b"]["c"]

        summary = RunEvals.from_dict(spec).with_functions({"get_nested": get_nested}).run()

        assert summary.all_passed

    def test_boolean_input(self):
        """Test boolean input."""
        spec = {
            "negate": {
                "dataset": [
                    {"case": {"input": True, "expected": False}},
                    {"case": {"input": False, "expected": True}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"negate": lambda x: not x}).run()

        assert summary.all_passed

    def test_float_input(self):
        """Test float input."""
        spec = {
            "round_val": {
                "dataset": [
                    {"case": {"input": 3.7, "expected": 4}},
                    {"case": {"input": 3.2, "expected": 3}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"round_val": round}).run()

        assert summary.all_passed

    def test_zero_input(self):
        """Test zero as input."""
        spec = {
            "is_zero": {
                "dataset": [
                    {"case": {"input": 0, "expected": True}},
                    {"case": {"input": 1, "expected": False}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"is_zero": lambda x: x == 0}).run()

        assert summary.all_passed

    def test_negative_input(self):
        """Test negative number input."""
        spec = {
            "absolute": {
                "dataset": [
                    {"case": {"input": -5, "expected": 5}},
                    {"case": {"input": -100, "expected": 100}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"absolute": abs}).run()

        assert summary.all_passed


class TestComplexInputCombinations:
    """Tests for complex input combinations."""

    def test_dict_with_list_values(self):
        """Test dict inputs with list values."""
        spec = {
            "process": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {"items": [1, 2, 3], "multiplier": 2},
                            "expected": [2, 4, 6],
                        }
                    },
                ]
            }
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"process": lambda items, multiplier: [x * multiplier for x in items]})
            .run()
        )

        assert summary.all_passed

    def test_list_inputs_with_dict_elements(self):
        """Test list inputs containing dicts."""
        spec = {
            "merge": {
                "dataset": [
                    {"case": {"inputs": [{"a": 1}, {"b": 2}], "expected": {"a": 1, "b": 2}}},
                ]
            }
        }

        def merge(d1, d2):
            result = d1.copy()
            result.update(d2)
            return result

        summary = RunEvals.from_dict(spec).with_functions({"merge": merge}).run()

        assert summary.all_passed

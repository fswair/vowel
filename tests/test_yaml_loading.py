"""Tests for YAML loading functionality."""

from pathlib import Path

import pytest

from vowel import (
    EvalsFile,
    load_evals,
    load_evals_file,
    load_evals_from_dict,
    load_evals_from_object,
    load_evals_from_yaml_string,
)


class TestLoadEvalsFromYamlString:
    """Tests for load_evals_from_yaml_string function."""

    def test_simple_yaml_loading(self, simple_yaml_spec: str):
        """Test loading a simple YAML spec."""
        evals = load_evals_from_yaml_string(simple_yaml_spec)

        assert "add" in evals
        assert len(evals["add"].dataset) == 2

    def test_yaml_with_evaluators(self, yaml_with_evaluators: str):
        """Test loading YAML with evaluators."""
        evals = load_evals_from_yaml_string(yaml_with_evaluators)

        assert "is_even" in evals
        assert evals["is_even"].evals is not None

    def test_yaml_with_type_check(self, yaml_with_type_check: str):
        """Test loading YAML with type checking."""
        evals = load_evals_from_yaml_string(yaml_with_type_check)

        assert "divide" in evals
        assert len(evals["divide"].dataset) == 2

    def test_yaml_with_raises(self, yaml_with_raises: str):
        """Test loading YAML with exception testing."""
        evals = load_evals_from_yaml_string(yaml_with_raises)

        assert "divide" in evals
        raises_cases = [c for c in evals["divide"].dataset if c.case.raises]
        assert len(raises_cases) == 1

    def test_empty_yaml_raises_error(self):
        """Test that empty YAML raises an error."""
        with pytest.raises(Exception):  # noqa: B017
            load_evals_from_yaml_string("")

    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises an error."""
        with pytest.raises(Exception):  # noqa: B017
            load_evals_from_yaml_string("invalid: [unclosed")


class TestLoadEvalsFromDict:
    """Tests for load_evals_from_dict function."""

    def test_dict_loading(self):
        """Test loading from a dictionary."""
        spec_dict = {
            "multiply": {
                "dataset": [
                    {"case": {"inputs": {"a": 2, "b": 3}, "expected": 6}},
                    {"case": {"inputs": {"a": 0, "b": 5}, "expected": 0}},
                ]
            }
        }

        evals = load_evals_from_dict(spec_dict)

        assert "multiply" in evals
        assert len(evals["multiply"].dataset) == 2

    def test_dict_with_evaluators(self):
        """Test loading dict with evaluators."""
        spec_dict = {
            "square": {
                "evals": {"Assertion": {"assertion": "output == input ** 2"}},
                "dataset": [
                    {"case": {"input": 3}},
                    {"case": {"input": 5}},
                ],
            }
        }

        evals = load_evals_from_dict(spec_dict)

        assert "square" in evals
        assert evals["square"].evals is not None


class TestLoadEvalsFile:
    """Tests for load_evals_file function."""

    def test_load_from_file(self, temp_yaml_file: Path):
        """Test loading from a YAML file."""
        evals = load_evals_file(str(temp_yaml_file))

        assert "add" in evals

    def test_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_evals_file("nonexistent_file.yml")


class TestLoadEvals:
    """Tests for the unified load_evals function."""

    def test_load_from_string(self, simple_yaml_spec: str):
        """Test load_evals with YAML string."""
        evals = load_evals(simple_yaml_spec)
        assert "add" in evals

    def test_load_from_dict(self):
        """Test load_evals with dict."""
        spec_dict = {"test": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        evals = load_evals(spec_dict)
        assert "test" in evals

    def test_load_from_path(self, temp_yaml_file: Path):
        """Test load_evals with Path object."""
        evals = load_evals(temp_yaml_file)
        assert "add" in evals

    def test_load_from_evals_file_object(self, simple_yaml_spec: str):
        """Test load_evals with EvalsFile object."""
        import yaml

        data = yaml.safe_load(simple_yaml_spec)
        evals_file = EvalsFile.model_validate(data)

        evals = load_evals_from_object(evals_file)
        assert "add" in evals

    def test_invalid_source_type_raises_error(self):
        """Test that invalid source type raises TypeError."""
        with pytest.raises(TypeError):
            load_evals(12345)  # type: ignore[arg-type]


class TestInputFormats:
    """Tests for different input formats in YAML."""

    def test_single_input(self):
        """Test single input format."""
        yaml_spec = """
double:
  dataset:
    - case:
        input: 5
        expected: 10
"""
        evals = load_evals_from_yaml_string(yaml_spec)
        case = evals["double"].dataset[0].case
        assert case.input == 5

    def test_inputs_dict(self):
        """Test inputs as dict format."""
        yaml_spec = """
add:
  dataset:
    - case:
        inputs: { x: 1, y: 2 }
        expected: 3
"""
        evals = load_evals_from_yaml_string(yaml_spec)
        case = evals["add"].dataset[0].case
        assert case.inputs == {"x": 1, "y": 2}

    def test_inputs_list(self):
        """Test inputs as list format."""
        yaml_spec = """
add:
  dataset:
    - case:
        inputs: [1, 2, 3]
        expected: 6
"""
        evals = load_evals_from_yaml_string(yaml_spec)
        case = evals["add"].dataset[0].case
        assert case.inputs == [1, 2, 3]

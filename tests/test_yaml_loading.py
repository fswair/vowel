"""Tests for YAML loading functionality."""

from pathlib import Path

import pytest

from vowel import (
    EvalsFile,
    load_bundle,
    load_bundle_file,
    load_bundle_from_dict,
    load_bundle_from_object,
    load_bundle_from_yaml_string,
)


class TestLoadBundleFromYamlString:
    """Tests for load_bundle_from_yaml_string function."""

    def test_simple_yaml_loading(self, simple_yaml_spec: str):
        """Test loading a simple YAML spec."""
        bundle = load_bundle_from_yaml_string(simple_yaml_spec)

        assert "add" in bundle.evals
        assert len(bundle.evals["add"].dataset) == 2

    def test_yaml_with_evaluators(self, yaml_with_evaluators: str):
        """Test loading YAML with evaluators."""
        bundle = load_bundle_from_yaml_string(yaml_with_evaluators)

        assert "is_even" in bundle.evals
        assert bundle.evals["is_even"].evals is not None

    def test_yaml_with_type_check(self, yaml_with_type_check: str):
        """Test loading YAML with type checking."""
        bundle = load_bundle_from_yaml_string(yaml_with_type_check)

        assert "divide" in bundle.evals
        assert len(bundle.evals["divide"].dataset) == 2

    def test_yaml_with_raises(self, yaml_with_raises: str):
        """Test loading YAML with exception testing."""
        bundle = load_bundle_from_yaml_string(yaml_with_raises)

        assert "divide" in bundle.evals
        raises_cases = [c for c in bundle.evals["divide"].dataset if c.case.raises]
        assert len(raises_cases) == 1

    def test_empty_yaml_raises_error(self):
        """Test that empty YAML raises an error."""
        with pytest.raises(Exception):  # noqa: B017
            load_bundle_from_yaml_string("")

    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises an error."""
        with pytest.raises(Exception):  # noqa: B017
            load_bundle_from_yaml_string("invalid: [unclosed")

        def test_yaml_with_top_level_serializers(self):
            """Test loading top-level serializer registry and eval references."""
            yaml_spec = """
serializers:
    user_schema:
        schema: tests.test_serializer.User

get_user_info:
    serializer: user_schema
    dataset:
        - case:
                input: {id: 1, name: Alice, email: a@a.com}
                expected: "User Alice has email a@a.com"
"""
            bundle = load_bundle_from_yaml_string(yaml_spec)

            assert "user_schema" in bundle.serializers
            assert bundle.evals["get_user_info"].serializer == "user_schema"

        def test_yaml_invalid_serializer_spec_raises_error(self):
            """Serializer specs cannot define both schema and serializer at once."""
            yaml_spec = """
serializers:
    invalid:
        schema: tests.test_serializer.User
        serializer: tests.test_serializer.yaml_serialize_user

get_user_info:
    serializer: invalid
    dataset:
        - case:
                input: {id: 1, name: Alice, email: a@a.com}
                expected: "User Alice has email a@a.com"
"""
            with pytest.raises(Exception):  # noqa: B017
                load_bundle_from_yaml_string(yaml_spec)


class TestLoadBundleFromDict:
    """Tests for load_bundle_from_dict function."""

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

        bundle = load_bundle_from_dict(spec_dict)

        assert "multiply" in bundle.evals
        assert len(bundle.evals["multiply"].dataset) == 2

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

        bundle = load_bundle_from_dict(spec_dict)

        assert "square" in bundle.evals
        assert bundle.evals["square"].evals is not None


class TestLoadBundleFile:
    """Tests for load_bundle_file function."""

    def test_load_from_file(self, temp_yaml_file: Path):
        """Test loading from a YAML file."""
        bundle = load_bundle_file(str(temp_yaml_file))

        assert "add" in bundle.evals

    def test_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_bundle_file("nonexistent_file.yml")


class TestLoadBundle:
    """Tests for the unified load_bundle function."""

    def test_load_from_string(self, simple_yaml_spec: str):
        """Test load_bundle with YAML string."""
        bundle = load_bundle(simple_yaml_spec)
        assert "add" in bundle.evals

    def test_load_from_dict(self):
        """Test load_bundle with dict."""
        spec_dict = {"test": {"dataset": [{"case": {"input": 1, "expected": 1}}]}}
        bundle = load_bundle(spec_dict)
        assert "test" in bundle.evals

    def test_load_from_path(self, temp_yaml_file: Path):
        """Test load_bundle with Path object."""
        bundle = load_bundle(temp_yaml_file)
        assert "add" in bundle.evals

    def test_load_from_evals_file_object(self, simple_yaml_spec: str):
        """Test load_bundle with EvalsFile object."""
        import yaml

        data = yaml.safe_load(simple_yaml_spec)
        evals_file = EvalsFile.model_validate(data)

        bundle = load_bundle_from_object(evals_file)
        assert "add" in bundle.evals

    def test_invalid_source_type_raises_error(self):
        """Test that invalid source type raises TypeError."""
        with pytest.raises(TypeError):
            load_bundle(12345)  # type: ignore[arg-type]


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
        bundle = load_bundle_from_yaml_string(yaml_spec)
        case = bundle.evals["double"].dataset[0].case
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
        bundle = load_bundle_from_yaml_string(yaml_spec)
        case = bundle.evals["add"].dataset[0].case
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
        bundle = load_bundle_from_yaml_string(yaml_spec)
        case = bundle.evals["add"].dataset[0].case
        assert case.inputs == [1, 2, 3]

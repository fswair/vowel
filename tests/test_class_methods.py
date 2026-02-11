"""Tests for class method evaluation support."""

import sys

import pytest

from vowel import RunEvals


@pytest.fixture(autouse=True)
def add_tmp_path_to_sys_path(tmp_path):
    """Add tmp_path to sys.path for all tests in this module."""
    sys.path.insert(0, str(tmp_path))
    yield
    # Clean up: remove tmp_path from sys.path
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


class TestClassMethodEvaluation:
    """Tests for evaluating class instance methods."""

    def test_class_method_with_cls_fixture(self, tmp_path):
        """Test evaluating a class method with cls-based fixture."""

        # Create a test module with a class
        test_module = tmp_path / "test_counter.py"
        test_module.write_text(
            """
class Counter:
    def __init__(self, start: int = 0):
        self.value = start

    def increment(self, by: int = 1) -> int:
        self.value += by
        return self.value

    def get_value(self) -> int:
        return self.value
"""
        )

        # Create YAML spec with class fixture
        yaml_spec = """
fixtures:
  Counter:
    cls: test_counter.Counter
    args: [10]
    scope: function

test_counter.Counter.increment:
  fixture: [Counter]
  dataset:
    - case:
        inputs: [1]
        expected: 11
    - case:
        inputs: [5]
        expected: 15

test_counter.Counter.get_value:
  fixture: [Counter]
  dataset:
    - case:
        expected: 10
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 2

    def test_class_method_with_kwargs_fixture(self, tmp_path):
        """Test evaluating a class method with kwargs in fixture."""

        # Create a test module with a class
        test_module = tmp_path / "test_calculator.py"
        test_module.write_text(
            """
class Calculator:
    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)
"""
        )

        # Create YAML spec with kwargs
        yaml_spec = """
fixtures:
  Calculator:
    cls: test_calculator.Calculator
    kwargs:
      precision: 3
    scope: function

test_calculator.Calculator.add:
  fixture: [Calculator]
  dataset:
    - case:
        inputs: [1.111, 2.222]
        expected: 3.333

test_calculator.Calculator.multiply:
  fixture: [Calculator]
  dataset:
    - case:
        inputs: [1.11, 2.22]
        expected: 2.464
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 2

    def test_class_method_module_scope_fixture(self, tmp_path):
        """Test evaluating class methods with module-scoped fixture."""

        # Create a test module with a class
        test_module = tmp_path / "test_stateful.py"
        test_module.write_text(
            """
class Stateful:
    def __init__(self):
        self.calls = []

    def record(self, value: str) -> str:
        self.calls.append(value)
        return f"Recorded: {value}"

    def get_calls(self) -> list:
        return list(self.calls)
"""
        )

        # Create YAML spec with module scope
        # Note: Module scope applies per eval, not across evals
        yaml_spec = """
fixtures:
  Stateful:
    cls: test_stateful.Stateful
    scope: module

test_stateful.Stateful.record:
  fixture: [Stateful]
  dataset:
    - case:
        inputs: ["first"]
        assertion: "'Recorded: first' in output"
    - case:
        inputs: ["second"]
        assertion: "'Recorded: second' in output"
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 1

    def test_class_method_with_teardown(self, tmp_path):
        """Test class method with fixture teardown."""

        # Create a test module with a class
        test_module = tmp_path / "test_resource.py"
        test_module.write_text(
            """
class Resource:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def use(self) -> str:
        if self.closed:
            raise RuntimeError("Resource is closed")
        return f"Using {self.name}"

    def close(self):
        self.closed = True

def close_resource(resource: Resource):
    resource.close()
"""
        )

        # Create YAML spec with teardown
        yaml_spec = """
fixtures:
  Resource:
    cls: test_resource.Resource
    args: ["db_connection"]
    teardown: test_resource.close_resource
    scope: function

test_resource.Resource.use:
  fixture: [Resource]
  dataset:
    - case:
        expected: "Using db_connection"
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 1

    def test_class_method_with_evaluators(self, tmp_path):
        """Test class method with global evaluators."""

        # Create a test module with a class
        test_module = tmp_path / "test_eval.py"
        test_module.write_text(
            """
class NumberOps:
    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier

    def multiply(self, x: int) -> int:
        return x * self.multiplier
"""
        )

        # Create YAML spec with evaluators
        yaml_spec = """
fixtures:
  NumberOps:
    cls: test_eval.NumberOps
    args: [3]

test_eval.NumberOps.multiply:
  fixture: [NumberOps]
  evals:
    IsInteger:
      type: int
    IsPositive:
      assertion: "output > 0"
  dataset:
    - case:
        input: 5
        expected: 15
    - case:
        input: 10
        expected: 30
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 1

    def test_class_method_multiple_instances(self, tmp_path):
        """Test multiple class instances in same eval."""

        # Create a test module with classes
        test_module = tmp_path / "test_multi.py"
        test_module.write_text(
            """
class Adder:
    def __init__(self, offset: int = 0):
        self.offset = offset

    def add(self, x: int) -> int:
        return x + self.offset

class Multiplier:
    def __init__(self, factor: int = 1):
        self.factor = factor

    def multiply(self, x: int) -> int:
        return x * self.factor
"""
        )

        # Create YAML spec with multiple instances
        yaml_spec = """
fixtures:
  Adder:
    cls: test_multi.Adder
    args: [10]
  Multiplier:
    cls: test_multi.Multiplier
    args: [5]

test_multi.Adder.add:
  fixture: [Adder]
  dataset:
    - case:
        input: 5
        expected: 15

test_multi.Multiplier.multiply:
  fixture: [Multiplier]
  dataset:
    - case:
        input: 3
        expected: 15
"""

        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text(yaml_spec)

        summary = RunEvals.from_file(str(yaml_file)).run()

        assert summary.all_passed
        assert summary.total_count == 2


class TestImportClass:
    """Tests for import_class function."""

    def test_import_class_basic(self, tmp_path):
        """Test importing a class from a module."""
        from vowel.utils import import_class

        # Create a test module
        test_module = tmp_path / "test_import_class.py"
        test_module.write_text(
            """
class TestClass:
    pass
"""
        )

        cls = import_class("test_import_class.TestClass")

        assert cls.__name__ == "TestClass"
        # Check it's actually a class (type)
        assert isinstance(cls, type)

    def test_import_class_builtin(self):
        """Test importing a builtin class."""
        from vowel.utils import import_class

        cls = import_class("builtins.dict")
        assert cls is dict

        cls = import_class("builtins.list")
        assert cls is list

    def test_import_class_invalid_format(self):
        """Test that invalid format raises error."""
        from vowel.utils import import_class

        with pytest.raises(ImportError, match="Invalid class path"):
            import_class("NoDots")

        with pytest.raises(ImportError, match="Invalid class path"):
            import_class("single.")

    def test_import_class_nonexistent_module(self):
        """Test that nonexistent module raises error."""
        from vowel.utils import import_class

        with pytest.raises(ImportError, match="Cannot import module"):
            import_class("nonexistent.module.ClassName")

    def test_import_class_not_a_class(self, tmp_path):
        """Test that importing a non-class raises error."""
        from vowel.utils import import_class

        # Create a test module with a function, not a class
        test_module = tmp_path / "test_not_class.py"
        test_module.write_text(
            """
def not_a_class():
    pass
"""
        )

        with pytest.raises(ImportError, match="is not a class"):
            import_class("test_not_class.not_a_class")

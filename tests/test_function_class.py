"""Tests for Function class functionality."""

import asyncio
import sys
from io import StringIO

import pytest

from vowel import Function


class TestFunctionCreation:
    """Tests for Function instantiation."""

    def test_create_function_with_code(self):
        """Test creating a Function with code string."""
        func = Function(
            name="add",
            description="Add two numbers",
            code="def add(a: int, b: int) -> int:\n    return a + b",
        )

        assert func.name == "add"
        assert func.description == "Add two numbers"
        assert "def add" in func.code

    def test_function_execute(self):
        """Test executing function code."""
        func = Function(
            name="multiply",
            description="Multiply two numbers",
            code="def multiply(x: int, y: int) -> int:\n    return x * y",
        )

        func.execute()

        assert func.func is not None
        assert callable(func.func)

    def test_function_call(self):
        """Test calling function directly."""
        func = Function(
            name="square",
            description="Square a number",
            code="def square(n: int) -> int:\n    return n * n",
        )

        result = func(5)

        assert result == 25

    def test_function_impl_property(self):
        """Test impl property auto-executes code."""
        func = Function(
            name="double",
            description="Double a number",
            code="def double(x: int) -> int:\n    return x * 2",
        )

        assert func.func is None

        impl = func.impl

        assert callable(impl)
        assert impl(10) == 20

    def test_function_name_property(self):
        """Test __name__ property."""
        func = Function(name="test_func", description="Test function", code="def test_func(): pass")

        assert func.__name__ == "test_func"


class TestFunctionFromCallable:
    """Tests for Function.from_callable static method."""

    def test_from_callable_simple(self, sample_add_function):
        """Test creating Function from a callable."""
        func = Function.from_callable(sample_add_function)

        assert func.name == "add"
        assert func.func is sample_add_function
        assert "def add" in func.code

    def test_from_callable_preserves_docstring(self):
        """Test that docstring becomes description."""

        def documented_func(x: int) -> int:
            """This is a documented function."""
            return x

        func = Function.from_callable(documented_func)

        assert "documented function" in func.description

    def test_from_callable_without_docstring(self):
        """Test function without docstring."""

        def no_doc(x: int) -> int:
            return x

        func = Function.from_callable(no_doc)

        assert func.description == "No description provided."

    def test_from_callable_execution(self, sample_is_even_function):
        """Test that from_callable function is executable."""
        func = Function.from_callable(sample_is_even_function)

        assert func(4)
        assert not func(7)


class TestFunctionPrint:
    """Tests for Function.print() method."""

    def test_print_simple(self):
        """Test _print_simple fallback."""
        func = Function(
            name="greet",
            description="Greet someone",
            code="def greet(name: str) -> str:\n    return f'Hello, {name}!'",
        )

        captured = StringIO()
        sys.stdout = captured

        func._print_simple()

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "greet" in output
        assert "Greet someone" in output

    def test_print_without_description(self):
        """Test print with show_description=False."""
        func = Function(name="test", description="Test description", code="def test(): pass")

        captured = StringIO()
        sys.stdout = captured

        func._print_simple(show_description=False)

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "test" in output
        assert "Test description" not in output


class TestFunctionWithComplexCode:
    """Tests for Function with complex code scenarios."""

    def test_multiline_function(self):
        """Test function with multiple lines."""
        code = """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

        func = Function(name="factorial", description="Calculate factorial", code=code)

        assert func(5) == 120
        assert func(0) == 1

    def test_function_with_imports(self):
        """Test function that needs imports."""
        code = """from math import sqrt
def hypotenuse(a: float, b: float) -> float:
    return sqrt(a**2 + b**2)"""

        func = Function(name="hypotenuse", description="Calculate hypotenuse", code=code)

        result = func(3.0, 4.0)
        assert abs(result - 5.0) < 0.001

    def test_function_with_type_hints(self):
        """Test function with complex type hints."""
        code = """def process_list(items: list[int]) -> dict[str, int]:
    return {"sum": sum(items), "count": len(items)}"""

        func = Function(name="process_list", description="Process a list of integers", code=code)

        result = func([1, 2, 3, 4, 5])
        assert result == {"sum": 15, "count": 5}

    def test_async_function(self):
        """Test async function execution."""
        code = """async def async_add(a: int, b: int) -> int:
    return a + b"""

        func = Function(name="async_add", description="Async add function", code=code)

        func.execute()
        assert func.func is not None
        result = asyncio.run(func.func(2, 3))
        assert result == 5

    def test_function_with_default_args(self):
        """Test function with default arguments."""
        code = '''def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"'''

        func = Function(name="greet", description="Greet with optional greeting", code=code)

        assert func("World") == "Hello, World!"
        assert func("World", "Hi") == "Hi, World!"


class TestFunctionFuncPath:
    """Tests for func_path property."""

    def test_func_path_without_file(self):
        """Test func_path when no file_path is set."""
        func = Function(name="test_func", description="Test", code="def test_func(): pass")

        assert func.func_path == "test_func"

    def test_func_path_with_file(self, tmp_path):
        """Test func_path with file_path set."""
        file_path = tmp_path / "module.py"
        file_path.write_text("def test(): pass")

        func = Function(
            name="test", description="Test", code="def test(): pass", file_path=str(file_path)
        )

        assert "test" in func.func_path


class TestFunctionErrorHandling:
    """Tests for Function error handling."""

    def test_invalid_code_raises_error(self):
        """Test that invalid code raises an error on execute."""
        func = Function(
            name="broken", description="Broken function", code="def broken(: invalid syntax"
        )

        with pytest.raises(RuntimeError, match="Error executing code"):
            func.execute()

    def test_missing_function_in_code(self):
        """Test when function name doesn't match code."""
        func = Function(name="expected_name", description="Test", code="def actual_name(): pass")

        with pytest.raises(KeyError):
            func.execute()

    def test_runtime_error_in_function(self):
        """Test function that raises at runtime."""
        func = Function(
            name="raises",
            description="Raises error",
            code="def raises():\n    raise ValueError('intentional')",
        )

        with pytest.raises(ValueError, match="intentional"):
            func()

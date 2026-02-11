"""Tests for type compatibility checking utilities."""

from collections.abc import Callable, Generator, Iterator
from typing import Any

from vowel import check_compatibility, get_unsupported_params, is_yaml_serializable_type


class TestIsYamlSerializableType:
    """Tests for is_yaml_serializable_type function."""

    def test_int_serializable(self):
        """Test int is serializable."""
        assert is_yaml_serializable_type(int)

    def test_float_serializable(self):
        """Test float is serializable."""
        assert is_yaml_serializable_type(float)

    def test_str_serializable(self):
        """Test str is serializable."""
        assert is_yaml_serializable_type(str)

    def test_bool_serializable(self):
        """Test bool is serializable."""
        assert is_yaml_serializable_type(bool)

    def test_none_serializable(self):
        """Test None is serializable."""
        assert is_yaml_serializable_type(None)
        assert is_yaml_serializable_type(type(None))

    def test_list_serializable(self):
        """Test list is serializable."""
        assert is_yaml_serializable_type(list)

    def test_dict_serializable(self):
        """Test dict is serializable."""
        assert is_yaml_serializable_type(dict)

    def test_tuple_serializable(self):
        """Test tuple is serializable."""
        assert is_yaml_serializable_type(tuple)

    def test_set_serializable(self):
        """Test set is serializable."""
        assert is_yaml_serializable_type(set)

    def test_list_int_serializable(self):
        """Test List[int] is serializable."""
        assert is_yaml_serializable_type(list[int])

    def test_dict_str_int_serializable(self):
        """Test Dict[str, int] is serializable."""
        assert is_yaml_serializable_type(dict[str, int])

    def test_optional_int_serializable(self):
        """Test Optional[int] is serializable."""
        assert is_yaml_serializable_type(int | None)

    def test_union_serializable(self):
        """Test Union types are serializable."""
        assert is_yaml_serializable_type(int | str)

    def test_any_serializable(self):
        """Test Any is serializable."""
        assert is_yaml_serializable_type(Any)

    def test_callable_not_serializable(self):
        """Test Callable is not serializable."""
        assert not is_yaml_serializable_type(Callable)
        assert not is_yaml_serializable_type(Callable[[int], int])

    def test_generator_not_serializable(self):
        """Test Generator is not serializable."""
        assert not is_yaml_serializable_type(Generator)

    def test_iterator_not_serializable(self):
        """Test Iterator is not serializable."""
        assert not is_yaml_serializable_type(Iterator)

    def test_nested_list_serializable(self):
        """Test nested list is serializable."""
        assert is_yaml_serializable_type(list[list[int]])

    def test_complex_dict_serializable(self):
        """Test complex dict is serializable."""
        assert is_yaml_serializable_type(dict[str, list[int]])


class TestGetUnsupportedParams:
    """Tests for get_unsupported_params function."""

    def test_all_params_supported(self):
        """Test function with all supported params."""

        def func(a: int, b: str, c: list[int]) -> int:
            return a

        unsupported = get_unsupported_params(func)

        assert len(unsupported) == 0

    def test_callable_param_unsupported(self):
        """Test function with Callable param."""

        def func(callback: Callable[[int], int]) -> int:
            return 0

        unsupported = get_unsupported_params(func)

        assert len(unsupported) == 1
        assert unsupported[0][0] == "callback"

    def test_multiple_unsupported_params(self):
        """Test function with multiple unsupported params."""

        def func(cb1: Callable[[int], int], cb2: Callable[[], None], good: int) -> int:
            return 0

        unsupported = get_unsupported_params(func)

        assert len(unsupported) == 2
        assert "cb1" in [u[0] for u in unsupported]
        assert "cb2" in [u[0] for u in unsupported]

    def test_no_annotations(self):
        """Test function without type annotations."""

        def func(a, b, c):
            return a

        unsupported = get_unsupported_params(func)

        assert len(unsupported) == 0

    def test_partial_annotations(self):
        """Test function with partial annotations."""

        def func(a: int, b, c: Callable) -> int:
            return a

        unsupported = get_unsupported_params(func)

        assert len(unsupported) == 1
        assert unsupported[0][0] == "c"


class TestCheckCompatibility:
    """Tests for check_compatibility function."""

    def test_compatible_function(self):
        """Test fully compatible function."""

        def add(a: int, b: int) -> int:
            return a + b

        is_compatible, issues = check_compatibility(add)

        assert is_compatible
        assert len(issues) == 0

    def test_incompatible_function(self):
        """Test incompatible function with callback."""

        def apply(func: Callable[[int], int], value: int) -> int:
            return func(value)

        is_compatible, issues = check_compatibility(apply)

        assert not is_compatible
        assert len(issues) == 1
        assert "func" in issues[0]

    def test_compatible_with_optional(self):
        """Test function with Optional param is compatible."""

        def greet(name: str, title: str | None = None) -> str:
            return f"{title or ''} {name}"

        is_compatible, issues = check_compatibility(greet)

        assert is_compatible

    def test_compatible_with_complex_types(self):
        """Test function with complex but serializable types."""

        def process(items: list[dict[str, int]], config: dict[str, Any]) -> list[int]:
            return []

        is_compatible, issues = check_compatibility(process)

        assert is_compatible


class TestRealWorldFunctions:
    """Tests with real-world function patterns."""

    def test_simple_math_function(self):
        """Test simple math function."""

        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        is_compatible, issues = check_compatibility(factorial)

        assert is_compatible

    def test_string_processing_function(self):
        """Test string processing function."""

        def process_text(text: str, max_length: int = 100, strip_whitespace: bool = True) -> str:
            if strip_whitespace:
                text = text.strip()
            return text[:max_length]

        is_compatible, issues = check_compatibility(process_text)

        assert is_compatible

    def test_data_transformation_function(self):
        """Test data transformation function."""

        def transform_data(
            records: list[dict[str, Any]], key_mapping: dict[str, str]
        ) -> list[dict[str, Any]]:
            return records

        is_compatible, issues = check_compatibility(transform_data)

        assert is_compatible

    def test_higher_order_function_incompatible(self):
        """Test higher-order function is incompatible."""

        def map_values(items: list[int], transform: Callable[[int], int]) -> list[int]:
            return [transform(x) for x in items]

        is_compatible, issues = check_compatibility(map_values)

        assert not is_compatible
        assert "transform" in issues[0]

    def test_async_compatible(self):
        """Test async function with simple types is compatible."""

        async def fetch_data(url: str, timeout: float = 30.0) -> dict:
            return {}

        is_compatible, issues = check_compatibility(fetch_data)

        assert is_compatible


class TestEdgeCases:
    """Tests for edge cases in type checking."""

    def test_empty_function(self):
        """Test function with no parameters."""

        def no_params() -> int:
            return 42

        is_compatible, issues = check_compatibility(no_params)

        assert is_compatible

    def test_var_args(self):
        """Test function with *args."""

        def var_func(*args: int) -> int:
            return sum(args)

        is_compatible, issues = check_compatibility(var_func)

        assert is_compatible

    def test_var_kwargs(self):
        """Test function with **kwargs."""

        def kwarg_func(**kwargs: int) -> int:
            return sum(kwargs.values())

        is_compatible, issues = check_compatibility(kwarg_func)

        assert is_compatible

    def test_mixed_args(self):
        """Test function with mixed parameter types."""

        def mixed(a: int, *args, b: str = "default", **kwargs) -> str:
            return ""

        is_compatible, issues = check_compatibility(mixed)

        assert is_compatible

    def test_lambda_compatibility(self):
        """Test lambda function compatibility."""

        def func(x):
            return x * 2

        is_compatible, issues = check_compatibility(func)

        assert is_compatible

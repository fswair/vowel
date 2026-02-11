"""Tests for async function support."""

import asyncio
import inspect

from vowel import Function, RunEvals


class TestAsyncFunctionExecution:
    """Tests for async function execution."""

    def test_async_function_basic(self):
        """Test basic async function evaluation."""

        async def async_double(x: int) -> int:
            return x * 2

        spec = {
            "async_double": {
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                    {"case": {"input": 0, "expected": 0}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_double": async_double}).run()

        assert summary.all_passed

    def test_async_function_with_await(self):
        """Test async function that awaits."""

        async def async_sum(items: list[int]) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return sum(items)

        spec = {
            "async_sum": {
                "dataset": [
                    {"case": {"input": [1, 2, 3], "expected": 6}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_sum": async_sum}).run()

        assert summary.all_passed

    def test_async_function_with_evaluator(self):
        """Test async function with evaluator."""

        async def async_process(text: str) -> str:
            await asyncio.sleep(0.001)
            return text.upper()

        spec = {
            "async_process": {
                "evals": {"Type": {"type": "str"}},
                "dataset": [
                    {"case": {"input": "hello", "expected": "HELLO"}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_process": async_process}).run()

        assert summary.all_passed


class TestAsyncFunctionClass:
    """Tests for Function class with async code."""

    def test_async_function_from_code(self):
        """Test creating async Function from code."""
        func = Function(
            name="async_add",
            description="Async add function",
            code="async def async_add(a: int, b: int) -> int:\n    return a + b",
        )

        func.execute()

        assert inspect.iscoroutinefunction(func.func)
        assert func.func is not None

        result = asyncio.run(func.func(2, 3))
        assert result == 5

    def test_async_function_detection(self):
        """Test detecting async function in code."""
        code = "async def my_async() -> int:\n    return 42"

        assert "async def" in code

    def test_async_function_from_callable(self):
        """Test Function.from_callable with async function."""

        async def original_async(x: int) -> int:
            """An async function."""
            return x * 2

        func = Function.from_callable(original_async)

        assert "async def" in func.code
        assert inspect.iscoroutinefunction(func.func)


class TestMixedSyncAsyncEvals:
    """Tests for mixed sync and async function evaluations."""

    def test_mixed_functions(self):
        """Test running both sync and async functions."""

        def sync_add(a: int, b: int) -> int:
            return a + b

        async def async_mul(a: int, b: int) -> int:
            return a * b

        spec = {
            "sync_add": {"dataset": [{"case": {"inputs": {"a": 2, "b": 3}, "expected": 5}}]},
            "async_mul": {"dataset": [{"case": {"inputs": {"a": 2, "b": 3}, "expected": 6}}]},
        }

        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {
                    "sync_add": sync_add,
                    "async_mul": async_mul,
                }
            )
            .run()
        )

        assert summary.total_count == 2
        assert summary.all_passed


class TestAsyncExceptionHandling:
    """Tests for async function exception handling."""

    def test_async_raises_exception(self):
        """Test async function that raises exception."""

        async def async_divide(a: float, b: float) -> float:
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return a / b

        spec = {
            "async_divide": {
                "dataset": [
                    {"case": {"inputs": {"a": 10, "b": 2}, "expected": 5.0}},
                    {"case": {"inputs": {"a": 10, "b": 0}, "raises": "ZeroDivisionError"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_divide": async_divide}).run()

        assert summary.all_passed

    def test_async_value_error(self):
        """Test async function raising ValueError."""

        async def async_validate(value: int) -> int:
            if value < 0:
                raise ValueError("negative value")
            return value

        spec = {
            "async_validate": {
                "dataset": [
                    {"case": {"input": 5, "expected": 5}},
                    {"case": {"input": -1, "raises": "ValueError", "match": "negative"}},
                ]
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_validate": async_validate}).run()

        assert summary.all_passed


class TestAsyncWithAssertions:
    """Tests for async functions with assertion evaluators."""

    def test_async_assertion(self):
        """Test async function with assertion."""

        async def async_square(x: int) -> int:
            return x * x

        spec = {
            "async_square": {
                "evals": {"Assertion": {"assertion": "output == input ** 2"}},
                "dataset": [
                    {"case": {"input": 4}},
                    {"case": {"input": 7}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_square": async_square}).run()

        assert summary.all_passed

    def test_async_multiple_assertions(self):
        """Test async function with multiple assertions."""

        async def async_abs(x: int) -> int:
            return abs(x)

        spec = {
            "async_abs": {
                "evals": {"Type": {"type": "int"}, "Assertion": {"assertion": "output >= 0"}},
                "dataset": [
                    {"case": {"input": -5}},
                    {"case": {"input": 5}},
                    {"case": {"input": 0}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"async_abs": async_abs}).run()

        assert summary.all_passed


class TestAsyncDuration:
    """Tests for async function duration constraints."""

    def test_async_fast_enough(self):
        """Test async function completes within duration."""

        async def fast_async(x: int) -> int:
            return x * 2

        spec = {
            "fast_async": {
                "evals": {"Duration": {"duration": 1.0}},
                "dataset": [
                    {"case": {"input": 5, "expected": 10}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"fast_async": fast_async}).run()

        assert summary.all_passed

    def test_async_too_slow(self):
        """Test async function exceeds duration."""

        async def slow_async(x: int) -> int:
            await asyncio.sleep(0.2)
            return x

        spec = {
            "slow_async": {
                "evals": {"Duration": {"duration": 0.05}},  # 50ms
                "dataset": [
                    {"case": {"input": 5}},
                ],
            }
        }

        summary = RunEvals.from_dict(spec).with_functions({"slow_async": slow_async}).run()

        assert not summary.all_passed

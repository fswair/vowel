"""Tests for executor backends, factory selection, and output parity."""

from __future__ import annotations

import asyncio
import importlib.util
from typing import TYPE_CHECKING

import pytest

from vowel.executor import (
    DefaultExecutor,
    Executor,
    get_executor,
    resolve_executors,
)

if TYPE_CHECKING:
    from vowel.executor import MontyExecutor

# MontyExecutor requires pydantic-monty; skip gracefully if unavailable.
_MONTY_AVAILABLE = importlib.util.find_spec("pydantic_monty") is not None

if _MONTY_AVAILABLE:
    from vowel.executor import MontyExecutor  # noqa: F811

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _binary_search(arr: list[int], target: int) -> int:
    """Reference binary search used across test classes."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _add(a, b):
    return a + b


def _build_executors() -> tuple[list[Executor], list[str]]:
    instances: list[Executor] = [DefaultExecutor()]
    ids = ["default"]
    if _MONTY_AVAILABLE:
        instances.insert(0, MontyExecutor())
        ids.insert(0, "monty")
    return instances, ids


EXECUTOR_INSTANCES, EXECUTOR_IDS = _build_executors()


@pytest.fixture(params=EXECUTOR_INSTANCES, ids=EXECUTOR_IDS)
def executor(request) -> Executor:
    """Parametrised fixture yielding each executor backend."""
    return request.param


# ---------------------------------------------------------------------------
# 1. External functions only
# ---------------------------------------------------------------------------


class TestExternalFunctions:
    """Snippet calls host-side callbacks via external_functions."""

    def test_single_function(self, executor: Executor):
        code = "_binary_search([1, 3, 5, 7, 9], 5)"
        r = asyncio.run(
            executor.execute(code, external_functions={"_binary_search": _binary_search})
        )
        assert r.success is True
        assert r.output == 2

    def test_multiple_calls(self, executor: Executor):
        code = (
            "results = []\n"
            "results.append(search([1, 3, 5, 7, 9], 5))\n"
            "results.append(search([1, 3, 5, 7, 9], 4))\n"
            "results.append(search([1], 1))\n"
            "results.append(search([], 1))\n"
            "results\n"
        )
        r = asyncio.run(executor.execute(code, external_functions={"search": _binary_search}))
        assert r.success is True
        assert r.output == [2, -1, 0, -1]

    def test_multiple_functions(self, executor: Executor):
        code = (
            "results = []\n"
            "results.append(search([10, 20, 30], 20))\n"
            "results.append(add(3, 4))\n"
            "results\n"
        )
        r = asyncio.run(
            executor.execute(
                code,
                external_functions={"search": _binary_search, "add": _add},
            )
        )
        assert r.success is True
        assert r.output == [1, 7]


# ---------------------------------------------------------------------------
# 2. Inputs only
# ---------------------------------------------------------------------------


class TestInputs:
    """Snippet uses injected values via inputs."""

    def test_arithmetic(self, executor: Executor):
        r = asyncio.run(executor.execute("x * y + z", inputs={"x": 10, "y": 3, "z": 5}))
        assert r.success is True
        assert r.output == 35

    def test_list_input(self, executor: Executor):
        r = asyncio.run(executor.execute("sorted(data)", inputs={"data": [3, 1, 2]}))
        assert r.success is True
        assert r.output == [1, 2, 3]

    def test_string_input(self, executor: Executor):
        r = asyncio.run(executor.execute("name.upper()", inputs={"name": "hello"}))
        assert r.success is True
        assert r.output == "HELLO"

    def test_dict_input(self, executor: Executor):
        r = asyncio.run(executor.execute("len(d)", inputs={"d": {"a": 1, "b": 2}}))
        assert r.success is True
        assert r.output == 2


# ---------------------------------------------------------------------------
# 3. Inputs + external functions combined
# ---------------------------------------------------------------------------


class TestCombined:
    """Snippet uses both inputs and external_functions."""

    def test_search_with_data(self, executor: Executor):
        r = asyncio.run(
            executor.execute(
                "search(data, query)",
                inputs={"data": [2, 4, 6, 8, 10], "query": 6},
                external_functions={"search": _binary_search},
            )
        )
        assert r.success is True
        assert r.output == 2

    def test_function_with_multiple_inputs(self, executor: Executor):
        code = (
            "results = []\n"
            "for item in items:\n"
            "    results.append(transform(item, factor))\n"
            "results\n"
        )
        r = asyncio.run(
            executor.execute(
                code,
                inputs={"items": [1, 2, 3], "factor": 10},
                external_functions={"transform": lambda x, f: x * f},
            )
        )
        assert r.success is True
        assert r.output == [10, 20, 30]


# ---------------------------------------------------------------------------
# 4. Pure code (no injection)
# ---------------------------------------------------------------------------


class TestPureCode:
    """Snippet needs no external injection."""

    def test_comprehension(self, executor: Executor):
        r = asyncio.run(executor.execute("[i**2 for i in range(5)]"))
        assert r.success is True
        assert r.output == [0, 1, 4, 9, 16]

    def test_arithmetic_expression(self, executor: Executor):
        r = asyncio.run(executor.execute("2 ** 10"))
        assert r.success is True
        assert r.output == 1024

    def test_multiline_with_last_expr(self, executor: Executor):
        code = "x = [1, 2, 3]\ny = [i * 2 for i in x]\nsum(y)\n"
        r = asyncio.run(executor.execute(code))
        assert r.success is True
        assert r.output == 12

    def test_no_trailing_expression(self, executor: Executor):
        """When the last statement is not an expression output should be None."""
        r = asyncio.run(executor.execute("x = 42"))
        assert r.success is True
        assert r.output is None


# ---------------------------------------------------------------------------
# 5. Stdout capture
# ---------------------------------------------------------------------------


class TestStdout:
    """print() output is captured in ExecutionResult.stdout."""

    def test_print_captured(self, executor: Executor):
        r = asyncio.run(executor.execute('print("hello")'))
        assert r.success is True
        assert "hello" in r.stdout


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Errors are returned as structured results, never raised."""

    def test_runtime_error(self, executor: Executor):
        r = asyncio.run(executor.execute("1 / 0"))
        assert r.success is False
        assert r.error_type == "ZeroDivisionError"
        assert r.output is None

    def test_type_error_in_external(self, executor: Executor):
        r = asyncio.run(
            executor.execute(
                'search("not_a_list", 5)',
                external_functions={"search": _binary_search},
            )
        )
        assert r.success is False
        assert r.error_type == "TypeError"

    def test_name_error(self, executor: Executor):
        r = asyncio.run(executor.execute("undefined_var + 1"))
        assert r.success is False
        assert r.error_type == "NameError"

    def test_syntax_error(self, executor: Executor):
        r = asyncio.run(executor.execute("def foo(:"))
        assert r.success is False
        assert r.error_type == "SyntaxError"

    def test_error_has_message(self, executor: Executor):
        r = asyncio.run(executor.execute("1 / 0"))
        assert r.error is not None
        assert len(r.error) > 0


# ---------------------------------------------------------------------------
# 7. ExecutionResult structure
# ---------------------------------------------------------------------------


class TestExecutionResult:
    """ExecutionResult fields are correctly populated."""

    def test_duration_is_positive(self, executor: Executor):
        r = asyncio.run(executor.execute("42"))
        assert r.duration_ms > 0

    def test_success_fields(self, executor: Executor):
        r = asyncio.run(executor.execute("42"))
        assert r.success is True
        assert r.error is None
        assert r.error_type is None

    def test_failure_fields(self, executor: Executor):
        r = asyncio.run(executor.execute("1/0"))
        assert r.success is False
        assert r.error is not None
        assert r.error_type is not None
        assert r.output is None


# ---------------------------------------------------------------------------
# 8. Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """Both executors satisfy the Executor protocol."""

    @pytest.mark.skipif(not _MONTY_AVAILABLE, reason="pydantic-monty not installed")
    def test_monty_is_executor(self):
        assert isinstance(MontyExecutor(), Executor)

    def test_default_is_executor(self):
        assert isinstance(DefaultExecutor(), Executor)


# ---------------------------------------------------------------------------
# 9. get_executor factory
# ---------------------------------------------------------------------------


class TestFactory:
    """get_executor returns the correct backend."""

    def test_auto(self):
        ex = get_executor("auto")
        assert isinstance(ex, Executor)

    @pytest.mark.skipif(not _MONTY_AVAILABLE, reason="pydantic-monty not installed")
    def test_monty(self):
        ex = get_executor("monty")
        assert isinstance(ex, MontyExecutor)

    def test_default(self):
        ex = get_executor("default")
        assert isinstance(ex, DefaultExecutor)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown executor backend"):
            get_executor("invalid")  # type: ignore


class _StaticSession:
    def __init__(self, value):
        self.value = value

    def feed(self, code):
        from vowel.executor import ExecutionResult

        return ExecutionResult(output=self.value, stdout="", success=True)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _RaisingExecutor:
    async def execute(self, code, **kwargs):
        raise RuntimeError("boom")

    def execute_sync(self, code, **kwargs):
        raise RuntimeError("boom")

    def create_session(self, setup_code, **kwargs):
        raise RuntimeError("boom")


class _StaticExecutor:
    def __init__(self, value):
        self.value = value

    async def execute(self, code, **kwargs):
        from vowel.executor import ExecutionResult

        return ExecutionResult(output=self.value, stdout="", success=True)

    def execute_sync(self, code, **kwargs):
        from vowel.executor import ExecutionResult

        return ExecutionResult(output=self.value, stdout="", success=True)

    def create_session(self, setup_code, **kwargs):
        return _StaticSession(self.value)


class TestResolveExecutors:
    def test_custom_executor_uses_default_fallback_on_session_failure(self):
        ex = resolve_executors(_RaisingExecutor())

        with ex.create_session("x = 1") as session:
            result = session.feed("x + 1")

        assert result.success is True
        assert result.output == 2

    def test_custom_fallback_executor_is_used(self):
        ex = resolve_executors(_RaisingExecutor(), _StaticExecutor("fallback"))

        with ex.create_session("ignored") as session:
            result = session.feed("ignored")

        assert result.success is True
        assert result.output == "fallback"


# ---------------------------------------------------------------------------
# 10. Parity — both executors produce the same output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MONTY_AVAILABLE, reason="pydantic-monty not installed")
class TestParity:
    """MontyExecutor and DefaultExecutor must agree on output."""

    CASES = [
        ("pure_arithmetic", "2 + 3", {}, {}),
        ("list_ops", "[1,2,3] + [4,5]", {}, {}),
        ("string_method", '"hello world".split()', {}, {}),
        ("with_inputs", "a + b", {"a": 10, "b": 20}, {}),
        ("with_ext_func", "f(3, 4)", {}, {"f": _add}),
        ("combined", "f(x, y)", {"x": 5, "y": 6}, {"f": _add}),
    ]

    @pytest.mark.parametrize(
        "label,code,inputs,ext_fns",
        CASES,
        ids=[c[0] for c in CASES],
    )
    def test_output_matches(self, label, code, inputs, ext_fns):
        monty = MontyExecutor()
        default = DefaultExecutor()
        kwargs: dict = {}
        if inputs:
            kwargs["inputs"] = inputs
        if ext_fns:
            kwargs["external_functions"] = ext_fns

        r_monty = asyncio.run(monty.execute(code, **kwargs))
        r_default = asyncio.run(default.execute(code, **kwargs))

        assert r_monty.success is True, f"Monty failed: {r_monty.error}"
        assert r_default.success is True, f"Default failed: {r_default.error}"
        assert r_monty.output == r_default.output, (
            f"Parity mismatch for '{label}': "
            f"monty={r_monty.output!r} vs default={r_default.output!r}"
        )

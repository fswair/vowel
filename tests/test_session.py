"""Tests for ExecutionSession API — DefaultSession and MontyReplSession.

Covers:
    - Basic feed() results (binary search)
    - Error handling (ZeroDivisionError)
    - Syntax error reporting
    - State preservation across feed() calls
    - Stdout capture through sessions
    - Context-manager lifecycle
    - Session isolation (fresh state per session)
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from vowel.executor import (
    DefaultExecutor,
    DefaultSession,
    ExecutionSession,
)

if TYPE_CHECKING:
    from vowel.executor import FallbackSession, MontyExecutor

# MontyExecutor requires pydantic-monty; skip gracefully if unavailable.
_MONTY_AVAILABLE = importlib.util.find_spec("pydantic_monty") is not None

if _MONTY_AVAILABLE:
    from vowel.executor import FallbackSession, MontyExecutor  # noqa: F811

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FUNC_CODE = """\
def binary_search(arr, target):
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
"""

SEARCH_CASES = [
    ("binary_search([1, 3, 5, 7, 9], 5)", 2),
    ("binary_search([], 1)", -1),
    ("binary_search([1, 2, 3], 4)", -1),
    ("binary_search([10, 20, 30], 10)", 0),
]


def _build_executor_params() -> tuple[list, list[str]]:
    params = [(DefaultExecutor, DefaultSession)]
    ids = ["default"]
    if _MONTY_AVAILABLE:
        params.insert(0, (MontyExecutor, FallbackSession))  # type: ignore
        ids.insert(0, "monty")
    return params, ids


EXECUTOR_CLASSES, EXECUTOR_IDS = _build_executor_params()


@pytest.fixture(params=EXECUTOR_CLASSES, ids=EXECUTOR_IDS)
def executor_and_session(request):
    """Yield (executor_instance, expected_session_class)."""
    cls, session_cls = request.param
    return cls(), session_cls


# ---------------------------------------------------------------------------
# Basic session correctness
# ---------------------------------------------------------------------------


class TestSessionBasic:
    """feed() returns correct outputs for a simple function."""

    def test_binary_search_cases(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session(FUNC_CODE) as session:
            for snippet, expected in SEARCH_CASES:
                r = session.feed(snippet)
                assert r.success, f"Failed: {snippet} => {r.error}"
                assert r.output == expected, f"{snippet}: got {r.output!r}, expected {expected!r}"

    def test_session_type(self, executor_and_session):
        """create_session() returns the correct session class."""
        executor, session_cls = executor_and_session
        with executor.create_session("x = 1") as session:
            assert isinstance(session, session_cls)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSessionErrors:
    """Errors are returned structured, not raised."""

    def test_zero_division(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("def foo(x): return 1/x") as session:
            r = session.feed("foo(0)")
            assert not r.success
            assert r.error_type == "ZeroDivisionError"
            assert r.error is not None

    def test_name_error(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("x = 1") as session:
            r = session.feed("undefined_var + 1")
            assert not r.success
            assert r.error_type == "NameError"

    def test_syntax_error(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("def foo(): return 42") as session:
            r = session.feed("foo(")
            assert not r.success
            assert r.error_type == "SyntaxError"

    def test_error_does_not_break_session(self, executor_and_session):
        """A single error in feed() should not corrupt the session."""
        executor, _ = executor_and_session
        with executor.create_session("def foo(x): return 1/x") as session:
            r_bad = session.feed("foo(0)")
            assert not r_bad.success
            # Session should still work after error:
            r_ok = session.feed("foo(2)")
            assert r_ok.success
            assert r_ok.output == 0.5


# ---------------------------------------------------------------------------
# State preservation
# ---------------------------------------------------------------------------


class TestStatePreservation:
    """State persists across feed() calls within a single session."""

    def test_mutation_persists(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("x = 10") as session:
            r1 = session.feed("x + 5")
            assert r1.output == 15

            session.feed("x = x * 2")

            r3 = session.feed("x")
            assert r3.output == 20

    def test_function_defined_in_session(self, executor_and_session):
        """Functions defined in one feed() are available in subsequent feeds."""
        executor, _ = executor_and_session
        with executor.create_session("y = 100") as session:
            session.feed("def double(n): return n * 2")
            r = session.feed("double(y)")
            assert r.success
            assert r.output == 200

    def test_list_accumulation(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("items = []") as session:
            session.feed("items.append(1)")
            session.feed("items.append(2)")
            session.feed("items.append(3)")
            r = session.feed("items")
            assert r.output == [1, 2, 3]


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------


class TestSessionStdout:
    """print() output is captured through the session."""

    def test_stdout_captured(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("def greet(name): print(f'Hello {name}')") as session:
            r = session.feed("greet('World')")
            assert "Hello World" in r.stdout


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    """Each session starts with a clean namespace."""

    def test_separate_sessions_isolated(self, executor_and_session):
        executor, _ = executor_and_session

        with executor.create_session("x = 42") as s1:
            r1 = s1.feed("x")
            assert r1.output == 42

        # A new session should NOT see x from the previous one:
        with executor.create_session("y = 99") as s2:
            r2 = s2.feed("y")
            assert r2.output == 99
            r_x = s2.feed("x")
            assert not r_x.success  # x should not exist


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestSessionProtocol:
    """Sessions satisfy the ExecutionSession protocol."""

    def test_protocol(self, executor_and_session):
        executor, _ = executor_and_session
        with executor.create_session("x = 1") as session:
            assert isinstance(session, ExecutionSession)

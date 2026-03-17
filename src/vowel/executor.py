"""Execution backends used by CodeMode for sandboxed and local code runs."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import logfire as _logfire

NEST_AVAILABLE = importlib.util.find_spec("nest_asyncio") is not None
MONTY_AVAILABLE = importlib.util.find_spec("pydantic_monty") is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, including running-loop environments."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "running event loop" not in str(exc) and "cannot be called from a running" not in str(
            exc
        ):
            raise
        # Already inside an event loop — patch and retry
        if not NEST_AVAILABLE:
            raise RuntimeError(
                "execute_sync() was called from inside a running event loop. "
                "Install nest-asyncio to support this: pip install nest-asyncio"
            ) from exc

        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of running a code snippet through an executor.

    Attributes
    ----------
    output:
        The value of the last expression evaluated in the snippet, or the
        value assigned to ``__result__`` in the namespace.  ``None`` when
        execution fails.
    stdout:
        Everything written to stdout during execution (via ``print()``).
    success:
        ``True`` when the snippet completed without raising an exception.
    error:
        Human-readable error message when ``success is False``.
    error_type:
        The Python exception class name (e.g. ``"ValueError"``) when
        ``success is False``.
    duration_ms:
        Wall-clock time spent executing the snippet, in milliseconds.
    """

    output: Any
    stdout: str
    success: bool
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Executor(Protocol):
    """Protocol for code execution backends used by CodeMode.

    Any callable object that matches this signature qualifies — concrete
    classes do *not* need to inherit from ``Executor``.

    Parameters
    ----------
    code:
        Python source code to execute.
    inputs:
        ``dict[str, Any]`` of values injected as top-level variables
        visible to the snippet.  For example ``{"x": 42}`` makes ``x``
        available inside the code.
    external_functions:
        ``dict[str, Callable]`` of host-side callbacks the snippet can
        call by name.  In the Monty backend each call exits the sandbox,
        runs on the host, and returns the result — so the real function
        can use any library.
    timeout:
        Maximum wall-clock seconds allowed for the snippet.  Execution is
        interrupted (or the result discarded) after this duration.
    max_memory:
        Maximum heap memory in bytes available to the sandbox.  Ignored by
        ``DefaultExecutor`` which has no memory isolation.

    Returns
    -------
    ExecutionResult
    """

    async def execute(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        raise NotImplementedError

    def execute_sync(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        raise NotImplementedError

    def create_session(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionSession:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ExecutionSession — compile once, feed many snippets
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionSession(Protocol):
    """A reusable execution context with pre-compiled setup code.

    The session compiles the *setup_code* (typically a function definition)
    once, then each ``feed()`` call runs a snippet against the preserved
    runtime state without re-parsing the setup code.

    This is the key optimisation for CodeMode exploration: when testing
    N edge-case snippets against the same function, the function is parsed
    and compiled only once instead of N times.

    The session is a context manager — use ``async with`` or ``with`` to
    ensure proper cleanup.
    """

    def feed(self, code: str) -> ExecutionResult:
        """Execute *code* against the session's pre-compiled state.

        Returns an ``ExecutionResult`` with the last expression value,
        stdout, and error info (if any).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources held by the session."""
        raise NotImplementedError

    def __enter__(self) -> ExecutionSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# MontyReplSession — sandboxed session using MontyRepl
# ---------------------------------------------------------------------------


class MontyReplSession:
    """Session backed by ``MontyRepl`` — compile once, feed many snippets.

    On construction the *setup_code* is parsed, compiled and executed once
    via ``MontyRepl.create()``.  Each subsequent ``feed()`` call runs a
    snippet against the preserved heap/globals without re-parsing the setup
    code.

    This is the recommended path for CodeMode exploration with Monty.  For
    a function with N edge-case snippets, the function source is compiled
    only once.
    """

    def __init__(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> None:
        import pydantic_monty

        self._pydantic_monty = pydantic_monty
        self._limits = pydantic_monty.ResourceLimits(
            max_duration_secs=timeout,
            max_memory=max_memory,
        )

        stdout_lines: list[str] = []

        def _print_callback(_stream: str, text: str) -> None:
            stdout_lines.append(text)

        # Create empty REPL and initialize it with setup code
        self._repl = pydantic_monty.MontyRepl(limits=self._limits)
        self._repl.feed_run(setup_code, print_callback=_print_callback)
        self._setup_stdout = "\n".join(stdout_lines)

    def feed(self, code: str) -> ExecutionResult:
        """Execute *code* against the REPL's preserved state."""
        stdout_lines: list[str] = []

        def _print_callback(_stream: str, text: str) -> None:
            stdout_lines.append(text)

        t0 = time.perf_counter()
        try:
            if not getattr(self, "_repl", None):
                # TODO: wrap with custom exception and detailed message
                raise ValueError("Repl not found.")
            else:
                output = self._repl.feed_run(code, print_callback=_print_callback)
                duration_ms = (time.perf_counter() - t0) * 1000
                return ExecutionResult(
                    output=output,
                    stdout="\n".join(stdout_lines),
                    success=True,
                    duration_ms=duration_ms,
                )

        except self._pydantic_monty.MontyRuntimeError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            inner = exc.exception()
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=exc.display(format="type-msg"),
                error_type=type(inner).__name__,
                duration_ms=duration_ms,
            )

        except self._pydantic_monty.MontySyntaxError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout="",
                success=False,
                error=str(exc),
                error_type="SyntaxError",
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

    def close(self) -> None:
        """Release the REPL instance."""
        # TODO: not sure about releasing the REPL instance is needed
        # self._repl = None  # type: ignore

    def __enter__(self) -> MontyReplSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# FallbackSession — Monty with auto-fallback to DefaultSession
# ---------------------------------------------------------------------------


class FallbackSession:
    """Session that tries MontyReplSession first, falls back to DefaultSession.

    Two fallback modes:

    1. **Session-level**: If ``MontyReplSession.__init__`` raises (e.g.
       ``MontySyntaxError`` for unsupported syntax like f-string ``!r``),
       the entire session transparently switches to ``DefaultSession``.

    2. **Snippet-level**: If a ``feed()`` call returns a
       ``ModuleNotFoundError`` (Monty doesn't have the module), that single
       snippet is re-executed via a ``DefaultSession``.  Subsequent Monty
       feeds continue normally — only the failing snippet falls back.
    """

    def __init__(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
        fallback_executor: Executor | None = None,
    ) -> None:
        self._setup_code = setup_code
        self._timeout = timeout
        self._max_memory = max_memory
        self._fallback_executor = fallback_executor or DefaultExecutor()
        self._monty_session: MontyReplSession | None = None
        self._fallback_session: ExecutionSession | None = None
        self._monty_failed_permanently = False

        try:
            self._monty_session = MontyReplSession(
                setup_code,
                timeout=timeout,
                max_memory=max_memory,
            )
        except Exception as exc:
            _logfire.info(
                "Monty session creation failed ({exc_type}: {exc_msg}), falling back to {fallback}",
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                fallback=type(self._fallback_executor).__name__,
            )
            self._monty_failed_permanently = True
            self._fallback_session = self._fallback_executor.create_session(
                setup_code,
                timeout=timeout,
                max_memory=max_memory,
            )

    def _get_fallback_session(self) -> ExecutionSession:
        """Lazily create the fallback session (only when first needed)."""
        if self._fallback_session is None:
            self._fallback_session = self._fallback_executor.create_session(
                self._setup_code,
                timeout=self._timeout,
                max_memory=self._max_memory,
            )
        return self._fallback_session

    def feed(self, code: str) -> ExecutionResult:
        """Execute *code*, falling back to the configured session on Monty gaps."""
        # Session-level fallback — Monty never worked
        if self._monty_failed_permanently:
            return self._get_fallback_session().feed(code)

        assert self._monty_session is not None
        result = self._monty_session.feed(code)

        # Snippet-level fallback — ModuleNotFoundError means Monty
        # doesn't have that stdlib module; retry with fallback session.
        if not result.success and result.error_type == "ModuleNotFoundError":
            _logfire.info(
                "Monty ModuleNotFoundError, retrying snippet with {fallback}: {error}",
                fallback=type(self._fallback_executor).__name__,
                error=result.error,
            )
            return self._get_fallback_session().feed(code)

        return result

    def close(self) -> None:
        if self._monty_session is not None:
            self._monty_session.close()
        if self._fallback_session is not None:
            self._fallback_session.close()

    def __enter__(self) -> FallbackSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# MontyExecutor — sandboxed, production-grade
# ---------------------------------------------------------------------------


class MontyExecutor:
    """Sandboxed executor backed by ``pydantic-monty`` (Rust interpreter).

    Monty provides strict isolation: no filesystem access, no network, no
    environment variables.  External functions are injected as host-side
    callbacks — they run on the *host* Python process with full access to
    stdlib and third-party libraries.

    Uses ``pydantic_monty.run_monty_async`` which implements Monty's step
    protocol (``start()`` → ``MontySnapshot`` → ``resume()``) with proper
    async support.  External functions can be sync or async — Monty handles
    both transparently.  The GIL is released during execution and Monty
    steps are offloaded to a thread pool.

    Requires the ``pydantic-monty`` package::

        pip install "vowel[monty]"   # or: pip install pydantic-monty

    Raises
    ------
    ImportError
        If ``pydantic-monty`` is not installed.
    """

    def __init__(self, fallback_executor: Executor | None = None) -> None:
        if not MONTY_AVAILABLE:
            raise ImportError(
                'MontyExecutor requires pydantic-monty. Install it with: pip install "vowel[monty]"'
            )
        self._fallback_executor = fallback_executor or DefaultExecutor()

    async def execute(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        """Execute *code* inside the Monty sandbox.

        Delegates to ``pydantic_monty.run_monty_async`` which handles the
        full step protocol (``start()`` → snapshot → ``resume()``).

        ``NameLookupSnapshot`` (undefined variables) is not handled by
        ``run_monty_async`` — it raises ``AssertionError``.  We catch that
        and use ``isinstance`` to detect the snapshot type cleanly.

        Parameters
        ----------
        code:
            Python source to run.
        inputs:
            Values injected as top-level variables (Monty ``inputs``).
        external_functions:
            Host-side callbacks the snippet can call by name.
        timeout / max_memory:
            Resource limits forwarded to Monty.
        """
        import pydantic_monty

        stdout_lines: list[str] = []

        def _print_callback(_stream: str, text: str) -> None:
            stdout_lines.append(text)

        input_names = list(inputs) if inputs else None

        limits = pydantic_monty.ResourceLimits(
            max_duration_secs=timeout,
            max_memory=max_memory,
        )

        t0 = time.perf_counter()
        try:
            m = pydantic_monty.Monty(
                code,
                inputs=input_names,
            )
            output = await pydantic_monty.run_monty_async(
                m,
                inputs=inputs,
                limits=limits,
                external_functions=external_functions,
                print_callback=_print_callback,
            )
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=output,
                stdout="\n".join(stdout_lines),
                success=True,
                duration_ms=duration_ms,
            )

        except pydantic_monty.MontyRuntimeError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            inner = exc.exception()
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=exc.display(format="type-msg"),
                error_type=type(inner).__name__,
                duration_ms=duration_ms,
            )

        except pydantic_monty.MontySyntaxError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout="",
                success=False,
                error=str(exc),
                error_type="SyntaxError",
                duration_ms=duration_ms,
            )

        except AssertionError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            # run_monty_async doesn't handle NameLookupSnapshot — it hits
            # `assert isinstance(progress, FutureSnapshot)` and the repr
            # of the snapshot is embedded in the assertion message.
            exc_msg = str(exc)
            if "NameLookupSnapshot" in exc_msg:
                marker = 'variable_name="'
                start = exc_msg.find(marker)
                if start != -1:
                    start += len(marker)
                    end = exc_msg.find('"', start)
                    var = exc_msg[start:end]
                    error = f"name '{var}' is not defined"
                else:
                    error = "name is not defined"
                return ExecutionResult(
                    output=None,
                    stdout="\n".join(stdout_lines),
                    success=False,
                    error=error,
                    error_type="NameError",
                    duration_ms=duration_ms,
                )
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=exc_msg,
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001 — catch-all for unexpected errors
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout="\n".join(stdout_lines),
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

    def execute_sync(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        """Synchronous wrapper around :meth:`execute`."""
        return run_sync(
            self.execute(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )
        )

    def create_session(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> FallbackSession:
        """Create a session that uses Monty with auto-fallback to DefaultSession.

        The *setup_code* (typically a function definition) is compiled and
        executed **once**.  If Monty cannot handle the code (e.g. unsupported
        syntax), the session transparently falls back to ``DefaultSession``.
        Individual ``feed()`` calls also fall back on ``ModuleNotFoundError``.
        """
        return FallbackSession(
            setup_code,
            timeout=timeout,
            max_memory=max_memory,
            fallback_executor=self._fallback_executor,
        )


# ---------------------------------------------------------------------------
# DefaultSession — unsandboxed session using persistent namespace
# ---------------------------------------------------------------------------


def _rewrite_last_expr(code: str) -> tuple[Any, bool]:
    """Parse *code* and rewrite the last expression to capture its value.

    Returns ``(compiled_code, has_result)`` where *has_result* is True when
    the last statement was an expression that was rewritten to assign to
    ``__result__``.
    """
    tree = ast.parse(code, "<vowel-session>", "exec")
    has_result = False
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body.pop()
        assign = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=last_expr.value,  # type: ignore[attr-defined]
        )
        ast.copy_location(last_expr, assign)
        tree.body.append(assign)
        ast.fix_missing_locations(tree)
        has_result = True
    return compile(tree, "<vowel-session>", "exec"), has_result


class DefaultSession:
    """Session backed by a persistent ``exec()`` namespace.

    The *setup_code* is executed once into a namespace dict on construction.
    Each ``feed()`` call executes a snippet in the **same** namespace, so
    functions and variables defined in the setup remain available.

    This mirrors ``MontyReplSession`` semantics for environments where Monty
    is not installed.
    """

    def __init__(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> None:
        self._namespace: dict[str, Any] = {}
        self._timeout = timeout
        # Execute setup code to define functions/variables
        exec(compile(setup_code, "<vowel-session-setup>", "exec"), self._namespace)  # noqa: S102

    def feed(self, code: str) -> ExecutionResult:
        """Execute *code* against the session's persistent namespace."""
        # Remove any leftover __result__ from previous feed
        self._namespace.pop("__result__", None)

        try:
            compiled, _has_result = _rewrite_last_expr(code)
        except SyntaxError as exc:
            return ExecutionResult(
                output=None,
                stdout="",
                success=False,
                error=str(exc),
                error_type="SyntaxError",
                duration_ms=0.0,
            )

        stdout_buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled, self._namespace)  # noqa: S102
            duration_ms = (time.perf_counter() - t0) * 1000
            output = self._namespace.get("__result__")
            return ExecutionResult(
                output=output,
                stdout=stdout_buf.getvalue(),
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout=stdout_buf.getvalue(),
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

    def close(self) -> None:
        """Clear the namespace."""
        self._namespace.clear()

    def __enter__(self) -> DefaultSession:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# DefaultExecutor — exec()-based, no sandbox
# ---------------------------------------------------------------------------


class DefaultExecutor:
    """Unsandboxed executor backed by Python's built-in ``exec()``.

    **WARNING: runs code with full host privileges.**  Only suitable for
    development, local testing, or environments where the code being executed
    is fully trusted.

    Both ``inputs`` and ``external_functions`` are merged into the execution
    namespace so the snippet can reference them as plain names.  The last
    assigned value of ``__result__``, or the module-level name ``results``
    if present, is returned as ``output``.

    No additional dependencies required — works with plain Python.

    Notes
    -----
    * ``timeout`` and ``max_memory`` parameters are accepted for API
      compatibility but are **not enforced**.
    * Stdout is captured via ``contextlib.redirect_stdout``.
    """

    async def execute(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        """Execute *code* using ``exec()`` — no sandbox, no isolation.

        To match Monty's behaviour, the value of the *last expression* in
        the snippet is captured automatically using ``ast`` rewriting.  If
        the snippet explicitly sets ``__result__``, that takes priority.
        """
        namespace: dict[str, Any] = {}
        if inputs:
            namespace.update(inputs)
        if external_functions:
            namespace.update(external_functions)

        # Rewrite the last expression statement to capture its value.
        try:
            tree = ast.parse(code, "<vowel-codemode>", "exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()
                assign = ast.Assign(
                    targets=[ast.Name(id="__result__", ctx=ast.Store())],
                    value=last_expr.value,  # type: ignore[attr-defined]
                )
                ast.copy_location(last_expr, assign)
                tree.body.append(assign)
                ast.fix_missing_locations(tree)
            compiled = compile(tree, "<vowel-codemode>", "exec")
        except SyntaxError as exc:
            return ExecutionResult(
                output=None,
                stdout="",
                success=False,
                error=str(exc),
                error_type="SyntaxError",
                duration_ms=0.0,
            )

        stdout_buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled, namespace)  # noqa: S102
            duration_ms = (time.perf_counter() - t0) * 1000

            output = namespace.get("__result__")

            return ExecutionResult(
                output=output,
                stdout=stdout_buf.getvalue(),
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExecutionResult(
                output=None,
                stdout=stdout_buf.getvalue(),
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )

    def execute_sync(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        """Synchronous wrapper around :meth:`execute`."""
        return run_sync(
            self.execute(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )
        )

    def create_session(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> DefaultSession:
        """Create an unsandboxed session with a persistent namespace.

        The *setup_code* is executed once into a shared namespace dict.
        Each ``session.feed(snippet)`` call runs in the same namespace,
        preserving functions and variables across calls.
        """
        return DefaultSession(
            setup_code,
            timeout=timeout,
            max_memory=max_memory,
        )


class ResolvedExecutor:
    """Executor wrapper that falls back when the primary executor raises."""

    def __init__(self, primary: Executor, fallback: Executor) -> None:
        self.primary = primary
        self.fallback = fallback

    async def execute(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        try:
            return await self.primary.execute(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )
        except Exception as exc:  # noqa: BLE001
            _logfire.info(
                "Primary executor {primary} raised {exc_type}; falling back to {fallback}",
                primary=type(self.primary).__name__,
                exc_type=type(exc).__name__,
                fallback=type(self.fallback).__name__,
            )
            return await self.fallback.execute(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )

    def execute_sync(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Callable[..., Any]] | None = None,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionResult:
        try:
            return self.primary.execute_sync(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )
        except Exception as exc:  # noqa: BLE001
            _logfire.info(
                "Primary executor {primary} raised {exc_type}; falling back to {fallback}",
                primary=type(self.primary).__name__,
                exc_type=type(exc).__name__,
                fallback=type(self.fallback).__name__,
            )
            return self.fallback.execute_sync(
                code,
                inputs=inputs,
                external_functions=external_functions,
                timeout=timeout,
                max_memory=max_memory,
            )

    def create_session(
        self,
        setup_code: str,
        *,
        timeout: float = 5.0,
        max_memory: int = 10 * 1024 * 1024,
    ) -> ExecutionSession:
        try:
            return self.primary.create_session(
                setup_code,
                timeout=timeout,
                max_memory=max_memory,
            )
        except Exception as exc:  # noqa: BLE001
            _logfire.info(
                "Primary executor {primary} session creation raised {exc_type}; "
                "falling back to {fallback}",
                primary=type(self.primary).__name__,
                exc_type=type(exc).__name__,
                fallback=type(self.fallback).__name__,
            )
            return self.fallback.create_session(
                setup_code,
                timeout=timeout,
                max_memory=max_memory,
            )


def resolve_executors(
    executor: Executor | None = None,
    fallback_executor: Executor | None = None,
) -> Executor:
    """Resolve primary/fallback executors while preserving Monty-first defaults."""
    fallback = fallback_executor or DefaultExecutor()

    if isinstance(executor, ResolvedExecutor):
        if fallback_executor is None:
            return executor
        return ResolvedExecutor(executor.primary, fallback)

    if executor is None:
        if MONTY_AVAILABLE:
            return MontyExecutor(fallback_executor=fallback)
        import warnings

        warnings.warn(
            "pydantic-monty not installed; using fallback executor "
            f'{type(fallback).__name__} (no sandboxing). Install with: pip install "vowel[monty]"',
            stacklevel=2,
        )
        return fallback

    if isinstance(executor, DefaultExecutor) and fallback_executor is None:
        return executor

    if isinstance(executor, MontyExecutor):
        executor._fallback_executor = fallback  # type: ignore[attr-defined]
        return executor

    if executor is fallback:
        return executor

    return ResolvedExecutor(executor, fallback)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_executor(backend: Literal["monty", "auto", "default"] = "auto") -> Executor:
    """Return a configured executor instance.

    Parameters
    ----------
    backend:
        ``"monty"``   — always use ``MontyExecutor`` (raises if not installed).
        ``"default"`` — always use ``DefaultExecutor``.
        ``"auto"``    — use ``MontyExecutor`` when available, fall back to
                        ``DefaultExecutor`` with a warning.

    Returns
    -------
    Executor
        A ready-to-use executor instance.
    """
    if backend == "monty":
        return MontyExecutor()

    if backend == "default":
        return DefaultExecutor()

    if backend == "auto":
        if MONTY_AVAILABLE:
            return MontyExecutor()
        import warnings

        warnings.warn(
            "pydantic-monty not installed; falling back to DefaultExecutor "
            '(no sandboxing). Install with: pip install "vowel[monty]"',
            stacklevel=2,
        )
        return DefaultExecutor()

    raise ValueError(
        f"Unknown executor backend: {backend!r}. Choose 'monty', 'default', or 'auto'."
    )

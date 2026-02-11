"""Utility functions for the vowel evaluation framework.

This module provides core utilities for:
- Loading and parsing YAML evaluation specifications
- Type compatibility checking for YAML serialization
- Function import and execution helpers
- Dataset creation and evaluation running
- Result aggregation and reporting

Key classes:
    EvalResult: Result of a single function evaluation
    EvalSummary: Aggregated results from multiple evaluations

Key functions:
    run_evals: Main entry point for running evaluations
    load_evals: Load evaluations from various sources
    to_dataset: Convert Evals to pydantic-evals Dataset
    is_yaml_serializable_type: Check if a type can be serialized to YAML
    check_compatibility: Validate function parameters for YAML compatibility
"""

import asyncio
import builtins
import collections.abc
import contextlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import wraps
from pathlib import Path, PurePath
from typing import Any, Literal, Optional, Union, get_args, get_origin

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import format_as_xml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected, Evaluator, MaxDuration
from pydantic_evals.reporting import EvaluationReport

from .eval_types import Evals, EvalsFile, FixtureDefinition
from .evals import (
    AssertionEvaluator,
    ContainsInputEvaluator,
    PatternMatchingEvaluator,
    RaisesEvaluator,
    TypeAdapterEvaluator,
    create_llm_judge,
)

logger = logging.getLogger(__name__)

_SYS_PATH_MODIFIED = False


# =============================================================================
# Evals Bundle - Container for evals and fixtures
# =============================================================================


class EvalsBundle(BaseModel):
    """Bundle containing evals and their associated fixtures."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    evals: dict[str, Evals] = Field(default_factory=dict)
    fixtures: dict[str, FixtureDefinition] = Field(default_factory=dict)


# =============================================================================
# YAML Serializable Types
# =============================================================================

YAML_SERIALIZABLE_TYPES = {
    int,
    float,
    str,
    bool,
    type(None),
    list,
    dict,
    tuple,
    set,
    frozenset,
    bytes,
    bytearray,
    datetime,
    date,
    time,
    timedelta,
    Decimal,
    Path,
    PurePath,
}

YAML_SERIALIZABLE_ORIGINS = {
    list,
    dict,
    tuple,
    set,
    frozenset,
    Optional,
    Union,
    collections.abc.Sequence,
    collections.abc.Mapping,
    collections.abc.MutableSequence,
    collections.abc.MutableMapping,
    collections.abc.Set,
    collections.abc.MutableSet,
    Sequence,
    Mapping,
    Literal,
}


def is_yaml_serializable_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents a YAML-serializable type.

    YAML can serialize:
    - Primitives: int, float, str, bool, None
    - Collections: list, dict, tuple, set (with serializable contents)
    - Optional and Union of serializable types

    YAML cannot serialize:
    - Callable types (functions, lambdas)
    - Custom classes without special handling
    - IO types (file handles, streams)
    - Generators, iterators
    - Context managers
    - Complex protocol types

    Args:
        type_hint: The type annotation to check

    Returns:
        True if the type can be represented in YAML, False otherwise
    """
    if type_hint is None or type_hint is type(None):
        return True

    if type_hint is Any:
        return True

    if type_hint in YAML_SERIALIZABLE_TYPES:
        return True

    origin = get_origin(type_hint)

    if origin is not None:
        if origin is Literal:
            args = get_args(type_hint)
            return all(isinstance(arg, (int, float, str, bool, type(None), bytes)) for arg in args)

        if origin in YAML_SERIALIZABLE_ORIGINS or origin in YAML_SERIALIZABLE_TYPES:
            args = get_args(type_hint)
            if not args:
                return True
            return all(is_yaml_serializable_type(arg) for arg in args)

        if origin is Union or origin is types.UnionType:
            args = get_args(type_hint)
            return all(is_yaml_serializable_type(arg) for arg in args)

        if origin is Callable or str(origin).startswith("typing.Callable"):
            return False

        return False

    if isinstance(type_hint, str):
        serializable_names = {
            "int",
            "float",
            "str",
            "bool",
            "None",
            "NoneType",
            "list",
            "dict",
            "tuple",
            "set",
            "frozenset",
            "List",
            "Dict",
            "Tuple",
            "Set",
            "FrozenSet",
            "Sequence",
            "Mapping",
            "Iterable",
            "Collection",
            "Any",
            "Optional",
            "Union",
            "Literal",
            "datetime",
            "date",
            "time",
            "timedelta",
            "Decimal",
            "Path",
            "PurePath",
            "bytes",
            "bytearray",
        }
        base_type = type_hint.split("[")[0].strip()
        if base_type in serializable_names:
            return True
        if "Callable" in type_hint or "callable" in type_hint.lower():
            return False
        return False

    try:
        from collections.abc import Callable as TypingCallable

        if type_hint is TypingCallable or type_hint is Callable:
            return False
    except ImportError:
        pass

    type_name = getattr(type_hint, "__name__", str(type_hint))
    non_serializable_patterns = [
        "Callable",
        "Generator",
        "Iterator",
        "Iterable",
        "Coroutine",
        "Awaitable",
        "AsyncGenerator",
        "AsyncIterator",
        "IO",
        "TextIO",
        "BinaryIO",
        "FileIO",
        "Protocol",
        "TypeVar",
        "Generic",
        "ContextManager",
        "AsyncContextManager",
    ]

    for pattern in non_serializable_patterns:
        if pattern in type_name:
            return False

    if isinstance(type_hint, type):
        return type_hint in YAML_SERIALIZABLE_TYPES

    return False


def get_unsupported_params(func: Callable) -> list[tuple[str, Any, str]]:
    """
    Get a list of function parameters that have non-YAML-serializable types.

    Args:
        func: The function to analyze

    Returns:
        List of tuples: (param_name, type_hint, reason)
        Empty list if all parameters are YAML-serializable
    """
    unsupported = []

    try:
        sig = inspect.signature(func)
        type_hints: dict[str, Any] = {}
        try:
            from typing import get_type_hints

            type_hints = get_type_hints(func)
        except Exception:
            type_hints = getattr(func, "__annotations__", {})
    except (ValueError, TypeError):
        return unsupported

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        type_hint = type_hints.get(param_name, param.annotation)

        if type_hint is inspect.Parameter.empty:
            continue

        if not is_yaml_serializable_type(type_hint):
            origin = get_origin(type_hint)
            if origin is Callable or "Callable" in str(type_hint):
                reason = "Callable/function types cannot be passed from YAML"
            elif "Generator" in str(type_hint) or "Iterator" in str(type_hint):
                reason = "Generator/Iterator types cannot be passed from YAML"
            elif "IO" in str(type_hint):
                reason = "IO/file types cannot be passed from YAML"
            elif "Protocol" in str(type_hint):
                reason = "Protocol types cannot be passed from YAML"
            else:
                reason = f"Complex type '{type_hint}' cannot be serialized to YAML"

            unsupported.append((param_name, type_hint, reason))

    return unsupported


def check_compatibility(func: Callable) -> tuple[bool, list[str]]:
    """
    Check if a function's parameters are compatible with YAML-based eval generation.

    Args:
        func: The function to check

    Returns:
        Tuple of (is_compatible, list_of_issues)
        - is_compatible: True if all params can be passed from YAML
        - list_of_issues: Human-readable descriptions of incompatible parameters
    """
    unsupported = get_unsupported_params(func)

    if not unsupported:
        return True, []

    issues = []
    for param_name, type_hint, reason in unsupported:
        issues.append(f"Parameter '{param_name}' ({type_hint}): {reason}")

    return False, issues


def _ensure_cwd_in_path() -> None:
    """Ensure current working directory is in sys.path (run once)."""
    global _SYS_PATH_MODIFIED
    if not _SYS_PATH_MODIFIED:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        _SYS_PATH_MODIFIED = True


def _apply_serializer(
    value: Any,
    serializer: type | Callable | dict[str, type | Callable] | None,
    param_name: str | None = None,
) -> Any:
    """
    Apply serializer to transform input value.

    Args:
        value: The input value to transform
        serializer: Type, callable, or dict of param-specific serializers
        param_name: Parameter name (for dict serializers)

    Returns:
        Transformed value
    """
    if serializer is None:
        return value

    # Dict of param-specific serializers
    if isinstance(serializer, dict):
        if param_name and param_name in serializer:
            return _apply_serializer(value, serializer[param_name])
        # If value is a dict, try to apply serializers to each key
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k in serializer:
                    result[k] = _apply_serializer(v, serializer[k])
                else:
                    result[k] = v
            return result
        return value

    # Type or callable serializer
    if isinstance(value, dict):
        # Try to construct from dict (Pydantic model style)
        try:
            return serializer(**value)
        except TypeError:
            # Not a constructor that takes **kwargs, try as callable
            return serializer(value)
    else:
        return serializer(value)


def _extract_args(
    arg_dict: dict | Any,
    serializer: type | Callable | dict[str, type | Callable] | None = None,
) -> tuple[tuple, dict]:
    """
    Extract positional and keyword args from input dict.

    Returns:
        (args, kwargs) tuple ready for function call
    """
    if not isinstance(arg_dict, dict):
        val = _apply_serializer(arg_dict, serializer)
        return (val,), {}

    if "inputs" in arg_dict:
        inputs = arg_dict["inputs"]
        if inputs is None:
            return (None,), {}
        if isinstance(inputs, dict):
            # Single type serializer -> serialize whole dict as one arg
            if serializer is not None and not isinstance(serializer, dict):
                val = _apply_serializer(inputs, serializer)
                return (val,), {}
            # Dict serializer -> apply per-param
            serialized = {
                k: _apply_serializer(v, serializer, param_name=k) for k, v in inputs.items()
            }
            return (), serialized
        else:
            # List inputs
            serialized = [_apply_serializer(v, serializer) for v in inputs]
            return tuple(serialized), {}

    elif "input" in arg_dict:
        val = _apply_serializer(arg_dict["input"], serializer)
        return (val,), {}

    # No input/inputs provided - return empty args for functions that take no arguments
    # (e.g., class methods with only self parameter)
    return (), {}


def unpack_inputs(
    func: Callable,
    has_raises: bool = False,
    serializer: type | Callable | dict[str, type | Callable] | None = None,
    serializer_fn: Callable[[dict], Any] | None = None,
) -> Callable:
    """
    Create a wrapper that unpacks input/inputs dict and calls function.

    Args:
        func: The function to wrap (sync or async)
        has_raises: If True, catches exceptions and returns them for validation
        serializer: Optional type/callable for automatic input conversion
        serializer_fn: Optional function for full control over input processing

    Returns:
        Wrapped function that accepts dict with 'input' or 'inputs' key
    """
    is_async = inspect.iscoroutinefunction(func)

    def _call_func(f, arg_dict):
        # serial_fn takes full control
        if serializer_fn is not None:
            result = serializer_fn(arg_dict)
            if isinstance(result, tuple):
                return f(*result)
            elif isinstance(result, dict):
                return f(**result)
            return f(result)

        args, kwargs = _extract_args(arg_dict, serializer)
        return f(*args, **kwargs)

    if is_async:

        @wraps(func)
        async def sync_wrapper(arg_dict):
            try:
                return await _call_func(func, arg_dict)
            except Exception as e:
                if has_raises:
                    return {"_exception": e, "_exception_type": type(e).__name__}
                raise

        return sync_wrapper
    else:

        @wraps(func)
        def async_wrapper(arg_dict):
            try:
                return _call_func(func, arg_dict)
            except Exception as e:
                if has_raises:
                    return {"_exception": e, "_exception_type": type(e).__name__}
                raise

        return async_wrapper


# =============================================================================
# Fixture Validation
# =============================================================================


class FixtureSignatureError(Exception):
    """Raised when a function signature doesn't comply with fixture requirements."""

    pass


def validate_fixture_signature(
    func: Callable[..., Any],
    fixture_names: list[str],
) -> None:
    """
    Validate that function signature complies with fixture requirements.

    Requirements:
    1. No *args or **kwargs (must be deterministic)
    2. Fixtures must be keyword-only arguments (after *)
    3. Fixtures must not be positional arguments (not at the start)

    Args:
        func: The function to validate
        fixture_names: List of fixture names this function uses

    Raises:
        FixtureSignatureError: If signature doesn't comply

    Examples:
        # ✅ Valid signatures:
        def fn(name: str, age: int, *, db: Any): ...
        def fn(x: int, *, db: Any, cache: Any): ...

        # ❌ Invalid signatures:
        def fn(*args, **kwargs): ...           # Not deterministic
        def fn(db: Any, name: str): ...        # Fixture not keyword-only
        def fn(name: str, db: Any): ...        # Fixture not after *
    """
    if not fixture_names:
        return  # No fixtures, no validation needed

    sig = inspect.signature(func)
    params = sig.parameters
    func_name = getattr(func, "__name__", "<unknown>")

    # Check for *args or **kwargs
    for _name, param in params.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise FixtureSignatureError(
                f"Function '{func_name}' has *args which is not allowed with fixtures. "
                f"Function signature must be deterministic."
            )
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise FixtureSignatureError(
                f"Function '{func_name}' has **kwargs which is not allowed with fixtures. "
                f"Function signature must be deterministic."
            )

    # Get keyword-only parameters (those after *)
    keyword_only_params = {
        name for name, param in params.items() if param.kind == inspect.Parameter.KEYWORD_ONLY
    }

    # Check that all fixtures are keyword-only
    for fixture_name in fixture_names:
        if fixture_name not in params:
            raise FixtureSignatureError(
                f"Function '{func_name}' uses fixture '{fixture_name}' but doesn't have "
                f"a parameter with that name. Add '*, {fixture_name}: Any' to the signature."
            )

        if fixture_name not in keyword_only_params:
            raise FixtureSignatureError(
                f"Function '{func_name}' has fixture '{fixture_name}' as a positional parameter. "
                f"Fixtures must be keyword-only arguments (after *). "
                f"Change signature to: def {func_name}(..., *, {fixture_name}: Any)"
            )


# =============================================================================
# Fixture Lifecycle Management
# =============================================================================


class FixtureManager:
    """
    Manages fixture lifecycle (setup/teardown) with proper scoping.

    Scopes:
    - function: Setup/teardown for each test case
    - module: Setup once per eval, teardown after all cases
    - session: Setup once per run_evals call, teardown at end

    Supports both:
    - YAML-defined fixtures (with import paths)
    - Programmatic fixtures (with actual callables)
    """

    def __init__(
        self,
        fixtures: dict[str, "FixtureDefinition"],
        fixture_funcs: dict[str, tuple[Callable, Callable | None]] | None = None,
    ):
        self.definitions = fixtures
        self._fixture_funcs = fixture_funcs or {}
        self._instances: dict[str, Any] = {}  # Cached fixture instances (all scopes)
        self._scope_counts: dict[str, int] = {}  # Reference counts for scoped fixtures
        self._generators: dict[str, Any] = {}  # Active generator fixtures for cleanup

    def setup(self, fixture_name: str) -> Any:
        """
        Setup a fixture and return its value.

        Args:
            fixture_name: Name of the fixture to setup

        Returns:
            The fixture instance (return value of setup function)
        """
        if fixture_name not in self.definitions:
            available = list(self.definitions.keys())
            raise ValueError(
                f"Unknown fixture: '{fixture_name}'. "
                f"Available fixtures: {available if available else '(none defined)'}"
            )

        defn = self.definitions[fixture_name]

        # For module/session scope, return cached instance if exists
        if defn.scope in ("module", "session") and fixture_name in self._instances:
            self._scope_counts[fixture_name] = self._scope_counts.get(fixture_name, 0) + 1
            return self._instances[fixture_name]

        # Class-based fixture
        if defn.cls:
            return self._setup_class_fixture(fixture_name, defn)

        # Function-based fixture
        return self._setup_function_fixture(fixture_name, defn)

    def _setup_class_fixture(self, fixture_name: str, defn: FixtureDefinition) -> Any:
        """Setup a class-based fixture by instantiating the class."""
        # Type assertion: defn.cls is guaranteed to be set here because
        # setup() only calls this function when cls is set
        assert defn.cls is not None, "cls must be set for class fixture"
        try:
            cls = import_class(defn.cls)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to import class for fixture '{fixture_name}': cannot import '{defn.cls}'. {e}"
            ) from e

        try:
            instance = cls(*defn.args, **defn.kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {defn.cls}: {e}") from e

        # Cache instance
        self._instances[fixture_name] = instance
        self._scope_counts[fixture_name] = self._scope_counts.get(fixture_name, 0) + 1

        return instance

    def _setup_function_fixture(self, fixture_name: str, defn: FixtureDefinition) -> Any:
        """Setup a function-based fixture (setup function or generator)."""
        # Get setup function - either programmatic or from import path
        if fixture_name in self._fixture_funcs:
            setup_func, teardown_func = self._fixture_funcs[fixture_name]
        else:
            # Type assertion: defn.setup is guaranteed to be set here because
            # setup() only calls this function when setup is set
            assert defn.setup is not None, "setup must be set for function fixture"
            try:
                setup_func = import_function(defn.setup)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import fixture '{fixture_name}': cannot import '{defn.setup}'. {e}"
                ) from e
            teardown_func = None

        # Call setup function - check if generator (pytest-style yield fixtures)
        try:
            if teardown_func is None and inspect.isgeneratorfunction(setup_func):
                # Generator fixture: extract yielded value, store generator for cleanup
                gen = setup_func(**defn.params) if defn.params else setup_func()
                instance = next(gen)
                self._generators[fixture_name] = gen
            else:
                # Normal fixture (tuple API or plain function)
                instance = setup_func(**defn.params) if defn.params else setup_func()
        except Exception as e:
            raise RuntimeError(f"Failed to setup fixture '{fixture_name}': {e}") from e

        # Cache instance (all scopes - function scope will be cleared on teardown)
        self._instances[fixture_name] = instance
        self._scope_counts[fixture_name] = self._scope_counts.get(fixture_name, 0) + 1

        return instance

    def teardown(self, fixture_name: str, scope_trigger: str = "function") -> None:
        """
        Teardown a fixture if appropriate for its scope.

        Args:
            fixture_name: Name of the fixture to teardown
            scope_trigger: The scope that triggered this teardown
        """
        if fixture_name not in self.definitions:
            return

        defn = self.definitions[fixture_name]

        # Only teardown if scope matches
        if defn.scope != scope_trigger:
            return

        # Decrement reference count
        if fixture_name in self._scope_counts:
            self._scope_counts[fixture_name] -= 1
            # For module/session scope, only teardown when count reaches 0
            if defn.scope in ("module", "session") and self._scope_counts[fixture_name] > 0:
                return  # Still in use

        # Perform teardown
        instance = self._instances.pop(fixture_name, None)
        if instance is not None:
            # Check if this is a generator fixture (pytest-style yield)
            gen = self._generators.pop(fixture_name, None)
            if gen is not None:
                # Continue generator to run cleanup code after yield
                with contextlib.suppress(StopIteration):
                    next(gen)
                # Generator cleanup handles its own teardown, skip explicit teardown func
                return

            # Get teardown function - programmatic, import path, or class method
            if fixture_name in self._fixture_funcs:
                _, teardown_func = self._fixture_funcs[fixture_name]
            elif defn.teardown:
                # Check if teardown is a class method (e.g., 'Connection.close')
                if "." in defn.teardown and defn.cls:
                    parts = defn.teardown.split(".")
                    if len(parts) == 2:
                        class_name, method_name = parts
                        if defn.cls.endswith(class_name):
                            teardown_func = getattr(instance, method_name, None)
                            if teardown_func:
                                teardown_func()
                                return
                # Fall back to regular function import
                teardown_func = import_function(defn.teardown)
            else:
                teardown_func = None

            if teardown_func:
                teardown_func(instance)

    def teardown_all(self, scope: str) -> None:
        """Teardown all fixtures of a given scope."""
        for fixture_name, defn in self.definitions.items():
            if defn.scope == scope and fixture_name in self._instances:
                self.teardown(fixture_name, scope)

    def get_fixtures_for_function(self, fixture_names: list[str]) -> dict[str, Any]:
        """
        Setup and return all fixtures for a function call.

        Args:
            fixture_names: List of fixture names needed

        Returns:
            Dict of {fixture_name: fixture_instance}
        """
        return {name: self.setup(name) for name in fixture_names}


def import_function(module_path: str) -> Any:
    """
    Import a function from a module path.
    Handles standard imports and file-based imports (when shadowing occurs).

    Args:
        module_path: Full module path like 'module.submodule.function', builtin name like 'len',
                    or builtin method like 'str.upper'

    Returns:
        The imported function or callable object

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function is not found in the module
    """
    _ensure_cwd_in_path()
    tried_combinations = []

    if "." not in module_path:
        try:
            return getattr(builtins, module_path)
        except AttributeError as e:
            raise ImportError(
                f"Function '{module_path}' not found in builtins. "
                f"Use full module path like 'module.function' or pass the function directly."
            ) from e

    parts = module_path.split(".")

    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        remaining_parts = parts[i:]
        tried_combinations.append(f"module='{module_name}', attr='{'.'.join(remaining_parts)}'")

        module = None

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.debug(f"Standard import failed for '{module_name}': {e}")
            relative_path = module_name.replace(".", os.sep) + ".py"
            file_path = os.path.join(os.getcwd(), relative_path)

            if os.path.exists(file_path):
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        logger.debug(f"File-based import succeeded for '{file_path}'")
                except Exception as e:
                    logger.debug(f"File-based import failed for '{file_path}': {e}")

        if module:
            try:
                obj: Any = module
                for part in remaining_parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError as e:
                logger.debug(f"Attribute lookup failed: {e}")
                continue

    try:
        obj = getattr(builtins, parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        pass

    raise ImportError(
        f"Cannot import '{module_path}'. Tried combinations:\n"
        + "\n".join(f"  - {c}" for c in tried_combinations)
    )


def import_class(class_path: str) -> type:
    """
    Import a class from a module path.

    Args:
        class_path: Full module path like 'myapp.Database' or 'module.submodule.ClassName'

    Returns:
        The imported class

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class is not found in the module
    """
    _ensure_cwd_in_path()

    parts = class_path.split(".")
    if len(parts) < 2 or any(not p for p in parts):
        raise ImportError(f"Invalid class path '{class_path}'. Expected format 'module.ClassName'.")

    module_name = ".".join(parts[:-1])
    class_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'") from e

    if not isinstance(cls, type):
        raise ImportError(f"'{class_name}' is not a class")

    return cls


def load_evals_file(yaml_path: str) -> dict[str, Evals]:
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)

    evals_file = EvalsFile.model_validate(loaded)
    return evals_file.get_evals()


def load_evals_from_yaml_string(yaml_content: str) -> dict[str, Evals]:
    loaded = yaml.safe_load(yaml_content)
    evals_file = EvalsFile.model_validate(loaded)
    return evals_file.get_evals()


def load_evals_from_dict(data: dict) -> dict[str, Evals]:
    evals_file = EvalsFile.model_validate(data)
    return evals_file.get_evals()


def load_evals_from_object(evals_obj: EvalsFile) -> dict[str, Evals]:
    return evals_obj.get_evals()


def load_evals(source: str | Path | dict | EvalsFile) -> dict[str, Evals]:
    if isinstance(source, EvalsFile):
        return load_evals_from_object(source)
    elif isinstance(source, dict):
        return load_evals_from_dict(source)
    elif isinstance(source, (str, Path)):
        source_str = str(source)
        # Check if it's an existing file path first, before YAML heuristics
        if os.path.exists(source_str):
            return load_evals_file(source_str)
        if "\n" in source_str or source_str.strip().startswith("{") or ":" in source_str:
            return load_evals_from_yaml_string(source_str)
        else:
            return load_evals_file(source_str)
    else:
        raise TypeError(
            f"source must be a file path (str/Path), YAML string (str), dict, "
            f"or EvalsFile object, got {type(source)}"
        )


# =============================================================================
# Bundle Loading Functions (with fixtures)
# =============================================================================


def load_bundle_file(yaml_path: str) -> EvalsBundle:
    """Load evals and fixtures from a YAML file."""
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)

    evals_file = EvalsFile.model_validate(loaded)
    return EvalsBundle(evals=evals_file.get_evals(), fixtures=evals_file.fixtures)


def load_bundle_from_yaml_string(yaml_content: str) -> EvalsBundle:
    """Load evals and fixtures from a YAML string."""
    loaded = yaml.safe_load(yaml_content)
    evals_file = EvalsFile.model_validate(loaded)
    return EvalsBundle(evals=evals_file.get_evals(), fixtures=evals_file.fixtures)


def load_bundle_from_dict(data: dict) -> EvalsBundle:
    """Load evals and fixtures from a dictionary."""
    evals_file = EvalsFile.model_validate(data)
    return EvalsBundle(evals=evals_file.get_evals(), fixtures=evals_file.fixtures)


def load_bundle_from_object(evals_obj: EvalsFile) -> EvalsBundle:
    """Load evals and fixtures from an EvalsFile object."""
    assert isinstance(evals_obj, EvalsFile)
    return EvalsBundle(evals=evals_obj.get_evals(), fixtures=evals_obj.fixtures)


def load_bundle(source: str | Path | dict | EvalsFile) -> EvalsBundle:
    """
    Load evals and fixtures from various sources.

    Args:
        source: Evaluation source (file path, YAML string, dict, or EvalsFile)

    Returns:
        EvalsBundle containing both evals and fixtures
    """
    if isinstance(source, EvalsFile):
        return load_bundle_from_object(source)
    elif isinstance(source, dict):
        return load_bundle_from_dict(source)
    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if "\n" in source_str or source_str.strip().startswith("{") or ":" in source_str:
            return load_bundle_from_yaml_string(source_str)
        else:
            return load_bundle_file(source_str)
    else:
        raise TypeError(
            f"source must be a file path (str/Path), YAML string (str), dict, "
            f"or EvalsFile object, got {type(source)}"
        )


def to_dataset(
    evals: Evals,
    *,
    name: str,
    ignore_duration: bool = False,
) -> Dataset:
    dataset_cases: list[Case] = []
    global_evaluators: list[Evaluator] = []

    for eval_case in evals.eval_cases:
        if eval_case.has_assertion:
            assertion = eval_case.case_data.assertion  # type: ignore[attr-defined]
            global_evaluators.append(AssertionEvaluator(assertion, evaluation_name=eval_case.id))
        elif eval_case.has_typecheck:
            type_str = eval_case.case_data.type  # type: ignore[attr-defined]
            strict = eval_case.case_data.strict  # type: ignore[attr-defined]
            global_evaluators.append(
                TypeAdapterEvaluator(type=type_str, evaluation_name=eval_case.id, strict=strict)
            )
        elif eval_case.has_duration and not ignore_duration:
            duration_seconds = eval_case.case_data.duration  # type: ignore[attr-defined]
            global_evaluators.append(MaxDuration(timedelta(seconds=duration_seconds)))
        elif eval_case.has_contains_input:
            case_sensitive = eval_case.case_data.case_sensitive  # type: ignore[attr-defined]
            as_strings = eval_case.case_data.as_strings  # type: ignore[attr-defined]
            global_evaluators.append(
                ContainsInputEvaluator(
                    evaluation_name=eval_case.id,
                    case_sensitive=case_sensitive,
                    as_strings=as_strings,
                )
            )
        elif eval_case.has_pattern_match:
            pattern = eval_case.case_data.pattern  # type: ignore[attr-defined]
            case_sensitive = eval_case.case_data.case_sensitive  # type: ignore[attr-defined]
            global_evaluators.append(
                PatternMatchingEvaluator(
                    pattern=pattern,
                    evaluation_name=eval_case.id,
                    case_sensitive=case_sensitive,
                )
            )
        elif eval_case.has_llm_judge:
            rubric = eval_case.case_data.rubric  # type: ignore[attr-defined]
            include = eval_case.case_data.include  # type: ignore[attr-defined]
            config = eval_case.case_data.config or {}  # type: ignore[attr-defined]
            judge = create_llm_judge(
                rubric=rubric,
                include=include,
                config=config,
            )
            judge.evaluation_name = eval_case.id  # type: ignore[attr-defined]
            global_evaluators.append(judge)

    for dataset_case in evals.dataset:
        match_case = dataset_case.case
        case_evaluators = []

        if match_case.has_expected:
            case_evaluators.append(EqualsExpected())

        if match_case.has_duration and not ignore_duration and match_case.duration is not None:
            case_evaluators.append(MaxDuration(timedelta(milliseconds=match_case.duration)))

        if match_case.has_contains:
            case_evaluators.append(Contains(match_case.contains))

        if match_case.has_assertion and match_case.assertion is not None:
            case_evaluators.append(
                AssertionEvaluator(
                    match_case.assertion,
                    evaluation_name=f"CaseAssertion: {match_case.assertion[:50]}",
                )
            )

        if match_case.has_pattern and match_case.pattern is not None:
            case_evaluators.append(
                PatternMatchingEvaluator(
                    pattern=match_case.pattern,
                    evaluation_name=f"CasePattern: {match_case.pattern[:50]}",
                    case_sensitive=match_case.case_sensitive,
                )
            )

        if match_case.has_raises and match_case.raises is not None:
            case_evaluators.append(
                RaisesEvaluator(
                    expected_exception_type=match_case.raises,
                    expected_exception_match=match_case.match,
                    optional=match_case.raises_optional,
                    evaluation_name=f"Raises: {match_case.raises}{'?' if match_case.raises_optional else ''}",
                )
            )

        if match_case.has_type and match_case.type is not None:
            case_evaluators.append(
                TypeAdapterEvaluator(
                    type=match_case.type,
                    strict=match_case.strict_type,
                    evaluation_name=f"CaseType: {match_case.type}",
                )
            )

        if not case_evaluators:
            case_evaluators = []

        case_metadata = {}
        if match_case.has_raises:
            case_metadata["_expects_exception"] = True
            case_metadata["_exception_type"] = match_case.raises
            if match_case.match:
                case_metadata["_exception_match"] = match_case.match

        if match_case.inputs is not None:
            display_input = f"inputs: {match_case.inputs}"
            input_value = {"inputs": match_case.inputs}
        else:
            display_input = f"input: {match_case.input}"
            input_value = {"input": match_case.input}

        if any(case for case in dataset_cases if input_value == case.inputs):
            logger.warning("Already exists in dataset, skipping duplicate case.")
            continue

        dataset_cases.append(
            Case(
                name=dataset_case.id or display_input,
                inputs=input_value,
                expected_output=match_case.expected,
                evaluators=tuple(case_evaluators),
                metadata=case_metadata if case_metadata else None,
            )
        )

    return Dataset(name=name, cases=dataset_cases, evaluators=global_evaluators)


def _merge_programmatic_fixtures(
    yaml_fixtures: dict[str, FixtureDefinition],
    programmatic_fixtures: dict[str, Callable | tuple[Callable, Callable | None]] | None,
) -> tuple[dict[str, FixtureDefinition], dict[str, tuple[Callable, Callable | None]]]:
    """
    Merge programmatic fixtures with YAML fixtures.

    Programmatic fixtures take precedence over YAML fixtures.

    Args:
        yaml_fixtures: Fixtures defined in YAML
        programmatic_fixtures: Fixtures provided as callables

    Returns:
        Tuple of (merged_fixtures, fixture_funcs)
    """
    merged_fixtures: dict[str, FixtureDefinition] = dict(yaml_fixtures)
    fixture_funcs: dict[str, tuple[Callable, Callable | None]] = {}

    if programmatic_fixtures:
        for name, fixture_spec in programmatic_fixtures.items():
            if isinstance(fixture_spec, tuple):
                setup_fn, teardown_fn = fixture_spec
            else:
                setup_fn, teardown_fn = fixture_spec, None

            # Store the actual functions for later use
            fixture_funcs[name] = (setup_fn, teardown_fn)  # type: ignore[assignment]

            # Create a placeholder FixtureDefinition (won't use setup path)
            merged_fixtures[name] = FixtureDefinition(
                setup=f"__programmatic__.{name}",  # Placeholder path
                teardown=f"__programmatic__.{name}_teardown" if teardown_fn else None,
                scope="function",  # Default scope for programmatic fixtures
            )

    return merged_fixtures, fixture_funcs


def _import_and_detect_class_method(
    eval_id: str,
    functions: dict[str, Callable] | None,
) -> tuple[Callable, str | None, str | None]:
    """
    Import a function and detect if it's a class method.

    Returns:
        Tuple of (function, class_path, class_name)
        - function: The callable
        - class_path: Full module.ClassName path for class methods, None otherwise
        - class_name: Class name for class methods, None otherwise
    """
    if functions and eval_id in functions:
        func = functions[eval_id]
        # Check if bound method (exclude builtin functions where __self__ is the module)
        if hasattr(func, "__self__") and not isinstance(func.__self__, types.ModuleType):
            class_name = func.__self__.__class__.__name__
            class_path = f"{func.__self__.__class__.__module__}.{class_name}"
            return func, class_path, class_name
        else:
            return func, None, None
    else:
        func = import_function(eval_id)

        # Detect unbound class methods by __qualname__ (e.g., 'ClassName.method')
        if inspect.isfunction(func) and hasattr(func, "__qualname__"):
            qualname = func.__qualname__
            if "." in qualname:
                parts = qualname.split(".")
                class_name = parts[0]
                # Only treat as class method if class_name is in eval_id path
                if class_name in eval_id.split("."):
                    id_parts = eval_id.split(".")
                    class_idx = id_parts.index(class_name)
                    class_path = ".".join(id_parts[: class_idx + 1])
                    return func, class_path, class_name

        return func, None, None


def _make_fixture_wrapper(
    original_func: Callable,
    fixture_names: list[str],
    fixture_manager: FixtureManager | None,
    module_fixtures: dict[str, Any],
    cls_name: str | None = None,
    cls_path: str | None = None,
) -> Callable:
    """Create a wrapper that injects fixtures into function calls."""
    is_async_func = inspect.iscoroutinefunction(original_func)

    if is_async_func:

        @wraps(original_func)
        async def async_fixture_wrapper(*args, **kwargs):
            if not fixture_names or not fixture_manager:
                return await original_func(*args, **kwargs)

            # Get fixtures for this call
            fixture_values = {}
            function_scoped = []
            instance = None
            instance_fixture_name = None

            for fname in fixture_names:
                defn = fixture_manager.definitions[fname]
                if defn.scope == "module":
                    fixture_values[fname] = module_fixtures[fname]
                elif defn.scope == "session":
                    fixture_values[fname] = fixture_manager.setup(fname)
                else:  # function scope
                    fixture_values[fname] = fixture_manager.setup(fname)
                    function_scoped.append(fname)

                # For class methods, save the instance for binding (check by name and class type)
                if cls_name and fname == cls_name:
                    instance = fixture_values[fname]
                    instance_fixture_name = fname
                elif cls_path and defn.cls:
                    try:
                        fixture_class = import_class(defn.cls)
                        method_class = import_class(cls_path)
                        if fixture_class is method_class:
                            instance = fixture_values[fname]
                            instance_fixture_name = fname
                    except (ImportError, AttributeError):
                        pass

            kwargs.update(fixture_values)

            try:
                # If class method with instance, bind the method
                if cls_name and instance is not None and inspect.isfunction(original_func):
                    bound_method = original_func.__get__(instance, type(instance))
                    kwargs_without_instance = {
                        k: v for k, v in kwargs.items() if k != instance_fixture_name
                    }
                    # For methods with no params besides self, filter out None args
                    sig = inspect.signature(bound_method)
                    num_params = sum(
                        1
                        for p in sig.parameters.values()
                        if p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
                    if num_params == 0 and len(args) == 1 and args[0] is None:
                        args = ()
                    return await bound_method(*args, **kwargs_without_instance)
                return await original_func(*args, **kwargs)
            finally:
                for fname in function_scoped:
                    fixture_manager.teardown(fname, "function")

        return async_fixture_wrapper
    else:

        @wraps(original_func)
        def fixture_wrapper(*args, **kwargs):
            if not fixture_names or not fixture_manager:
                return original_func(*args, **kwargs)

            # Get fixtures for this call
            fixture_values = {}
            function_scoped = []
            instance = None
            instance_fixture_name = None

            for fname in fixture_names:
                defn = fixture_manager.definitions[fname]
                if defn.scope == "module":
                    fixture_values[fname] = module_fixtures[fname]
                elif defn.scope == "session":
                    fixture_values[fname] = fixture_manager.setup(fname)
                else:  # function scope
                    fixture_values[fname] = fixture_manager.setup(fname)
                    function_scoped.append(fname)

                # For class methods, save the instance for binding (check by name and class type)
                if cls_name and fname == cls_name:
                    instance = fixture_values[fname]
                    instance_fixture_name = fname
                elif cls_path and defn.cls:
                    try:
                        fixture_class = import_class(defn.cls)
                        method_class = import_class(cls_path)
                        if fixture_class is method_class:
                            instance = fixture_values[fname]
                            instance_fixture_name = fname
                    except (ImportError, AttributeError):
                        pass

            kwargs.update(fixture_values)

            try:
                # If class method with instance, bind the method
                if cls_name and instance is not None and inspect.isfunction(original_func):
                    bound_method = original_func.__get__(instance, type(instance))
                    kwargs_without_instance = {
                        k: v for k, v in kwargs.items() if k != instance_fixture_name
                    }
                    # For methods with no params besides self, filter out None args
                    sig = inspect.signature(bound_method)
                    num_params = sum(
                        1
                        for p in sig.parameters.values()
                        if p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
                    if num_params == 0 and len(args) == 1 and args[0] is None:
                        args = ()
                    return bound_method(*args, **kwargs_without_instance)
                return original_func(*args, **kwargs)
            finally:
                for fname in function_scoped:
                    fixture_manager.teardown(fname, "function")

        return fixture_wrapper


def _evaluate_single_function(
    eval_id: str,
    evals: Evals,
    functions: dict[str, Callable] | None,
    merged_fixtures: dict[str, FixtureDefinition],
    fixture_manager: FixtureManager | None,
    schema: dict[str, type | Callable | dict[str, type | Callable]],
    serial_fn: dict[str, Callable[[dict], Any]],
    ignore_duration: bool,
) -> "EvalResult":
    """
    Evaluate a single function with its test cases.

    Args:
        eval_id: Function identifier (e.g., 'module.function')
        evals: Evaluation specification for this function
        functions: Optional dict of pre-imported functions
        merged_fixtures: Combined YAML and programmatic fixtures
        fixture_manager: Fixture lifecycle manager
        schema: Input type serializers
        serial_fn: Custom serializer functions
        ignore_duration: Skip duration constraints

    Returns:
        EvalResult with report or error
    """
    try:
        # Import and detect if it's a class method
        actual_func, class_path, class_name = _import_and_detect_class_method(eval_id, functions)
        click.echo(f"  📦 Imported: {click.style(eval_id, fg='cyan')}", err=True)

        # Get fixture names for this eval
        fixture_names = evals.fixture or []

        # For class methods, auto-add instance fixture if user hasn't specified any
        if class_name and not evals.fixture:
            instance_fixture_name = class_name
            fixture_names = [instance_fixture_name]

        # Validate fixture signature if fixtures are used
        if fixture_names:
            if not fixture_manager:
                raise ValueError(
                    f"Eval '{eval_id}' requires fixtures {fixture_names} but no fixtures "
                    f"are defined (neither in YAML nor via with_fixtures())."
                )
            missing = [f for f in fixture_names if f not in merged_fixtures]
            if missing:
                raise ValueError(
                    f"Eval '{eval_id}' requires undefined fixtures: {missing}. "
                    f"Available: {list(merged_fixtures.keys())}"
                )
            # For class methods, exclude fixtures that are instances of the method's class
            fixtures_to_validate = []
            for fname in fixture_names:
                if fname == class_name:
                    continue
                if class_path:
                    defn = merged_fixtures[fname]
                    if defn.cls:
                        try:
                            fixture_class = import_class(defn.cls)
                            method_class = import_class(class_path)
                            if fixture_class is method_class:
                                continue
                        except (ImportError, AttributeError):
                            pass
                fixtures_to_validate.append(fname)

            if fixtures_to_validate:
                validate_fixture_signature(actual_func, fixtures_to_validate)

        dataset = to_dataset(name=eval_id, evals=evals, ignore_duration=ignore_duration)

        has_any_raises = any(
            case.metadata and case.metadata.get("_expects_exception") for case in dataset.cases
        )

        # Get serializers for this function if defined
        func_schema = schema.get(eval_id)
        func_serial_fn = serial_fn.get(eval_id)

        # Setup module-scoped fixtures for this eval
        module_fixtures = {}
        if fixture_manager and fixture_names:
            for fname in fixture_names:
                defn = merged_fixtures[fname]
                if defn.scope == "module":
                    module_fixtures[fname] = fixture_manager.setup(fname)

        # Wrap function with fixture injection
        fixture_wrapped_func = _make_fixture_wrapper(
            actual_func,
            fixture_names,
            fixture_manager,
            module_fixtures,
            class_name,
            class_path,
        )

        wrapped_func = unpack_inputs(
            fixture_wrapped_func,
            has_raises=has_any_raises,
            serializer=func_schema,
            serializer_fn=func_serial_fn,
        )

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            def run_eval(ds=dataset, wf=wrapped_func):
                return ds.evaluate_sync(wf, progress=False)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_eval)
                report = future.result()
        except RuntimeError:
            report = dataset.evaluate_sync(wrapped_func, progress=False)

        # Teardown module-scoped fixtures after this eval completes
        if fixture_manager:
            for fname in fixture_names:
                fixture_manager.teardown(fname, "module")

        return EvalResult(eval_id, report=report)

    except (ImportError, AttributeError) as e:
        return EvalResult(eval_id, error=e)
    except Exception as e:
        return EvalResult(eval_id, error=e)


class EvalResult(BaseModel):
    """Result of a single function evaluation.

    Attributes:
        eval_id: Unique identifier for this evaluation
        report: Detailed evaluation report if successful
        error: Error message if evaluation failed
        success: Whether the evaluation passed all assertions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eval_id: str
    report: EvaluationReport | None = None
    error: str | None = None
    success: bool = Field(default=False)

    def __init__(
        self,
        eval_id: str,
        report: EvaluationReport | None = None,
        error: Exception | None = None,
        **data,
    ):
        error_str = str(error) if error else None
        success = error is None and not self._check_failures(report)
        super().__init__(eval_id=eval_id, report=report, error=error_str, success=success, **data)

    @staticmethod
    def _check_errors(report: EvaluationReport | None) -> bool:
        """Check if report has execution errors (crashes, evaluator failures, etc.)."""
        if report is None:
            return False

        if hasattr(report, "failures") and report.failures:
            return True

        for case in report.cases:
            if getattr(case, "error", None) is not None:
                return True

            if getattr(case, "evaluator_failures", None):
                return True

            for assertion in case.assertions.values():
                if getattr(assertion, "error", None) is not None:
                    return True
        return False

    @staticmethod
    def _check_failed_cases(report: EvaluationReport | None) -> bool:
        """Check if report has failed assertions (case didn't pass but no error)."""
        if report is None:
            return False

        for case in report.cases:
            for assertion in case.assertions.values():
                if not assertion.value:
                    return True
        return False

    @staticmethod
    def _check_failures(report: EvaluationReport | None) -> bool:
        """Check if report has any failures (errors OR failed cases)."""
        return EvalResult._check_errors(report) or EvalResult._check_failed_cases(report)

    def has_errors(self) -> bool:
        """Returns True if there are execution errors, crashes, or evaluator failures."""
        return self._check_errors(self.report)

    def has_failed_cases(self) -> bool:
        """Returns True if there are failed assertions but no errors."""
        return self._check_failed_cases(self.report) and not self._check_errors(self.report)

    def has_failures(self) -> bool:
        """Returns True if there are any failures (errors or failed cases)."""
        return self._check_failures(self.report)

    def get_coverage(self) -> float:
        """Calculate the coverage (percentage of passed cases)."""
        if self.report is None:
            return 0.0

        total_cases = len(self.report.cases)
        if total_cases == 0:
            return 1.0

        passed_cases = sum(
            1
            for case in self.report.cases
            if all(assertion.value for assertion in case.assertions.values())
        )
        return passed_cases / total_cases


class EvalSummary(BaseModel):
    """Aggregated results from multiple evaluations.

    Attributes:
        results: List of individual evaluation results
        success_count: Number of passed evaluations
        failed_count: Number of failed evaluations
        error_count: Number of evaluations that errored
        total_count: Total number of evaluations run
        func: Reference to the evaluated function
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[EvalResult]
    success_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    error_count: int = Field(default=0)
    total_count: int = Field(default=0)
    func: Any | None = None

    def __init__(self, results: list[EvalResult], **data):
        error_count = sum(1 for r in results if r.error is not None)

        failed_count = sum(1 for r in results if r.report and r.has_failures())

        success_count = sum(
            1 for r in results if r.report and not r.has_failures() and r.error is None
        )

        total_count = len(results)
        super().__init__(
            results=results,
            success_count=success_count,
            failed_count=failed_count,
            error_count=error_count,
            total_count=total_count,
            **data,
        )

    @property
    def all_passed(self) -> bool:
        return self.success_count == self.total_count

    @property
    def failed_results(self) -> list[EvalResult]:
        return [r for r in self.results if r.report and r.has_failures()]

    @property
    def has_errors(self) -> bool:
        """Returns True if any result has execution errors (crashes, evaluator failures)."""
        return any(r.has_errors() for r in self.results if r.report)

    @property
    def has_failed_cases(self) -> bool:
        """Returns True if any result has failed cases but no errors."""
        return any(r.has_failed_cases() for r in self.results if r.report)

    @property
    def coverage(self) -> float:
        """Calculate overall coverage (percentage of passed cases across all results)."""
        total_cases = 0
        passed_cases = 0

        for result in self.results:
            if result.report:
                for case in result.report.cases:
                    total_cases += 1
                    if all(assertion.value for assertion in case.assertions.values()):
                        passed_cases += 1

        return passed_cases / total_cases if total_cases > 0 else 1.0

    def meets_coverage(self, min_coverage: float = 1.0) -> bool:
        """Check if coverage meets the minimum threshold."""
        return self.coverage >= min_coverage

    def print(self, *, include_reports: bool = True, include_reasons: bool = True):
        """
        Print a formatted summary of evaluation results using rich tables.

        Args:
            include_reports: Whether to print individual evaluation reports
            include_reasons: Whether to include failure reasons in reports
        """
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
        except ImportError:
            self._print_simple(include_reports=include_reports)
            return

        console = Console()

        summary_table = Table(title="📊 Evaluation Summary", box=box.ROUNDED, show_header=True)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Total Functions", str(self.total_count))
        summary_table.add_row("✅ Passed", f"[green]{self.success_count}[/green]")
        summary_table.add_row(
            "❌ Failed",
            f"[red]{self.failed_count}[/red]" if self.failed_count > 0 else "0",
        )
        summary_table.add_row(
            "⚠️  Errors",
            f"[yellow]{self.error_count}[/yellow]" if self.error_count > 0 else "0",
        )
        summary_table.add_row(
            "Status",
            (
                "[green]✅ ALL PASSED[/green]"
                if self.all_passed
                else "[red]❌ FAILURES DETECTED[/red]"
            ),
        )

        console.print(summary_table)

        if self.results:
            console.print()
            results_table = Table(title="🔍 Function Results", box=box.SIMPLE, show_header=True)
            results_table.add_column("Function", style="cyan", no_wrap=True)
            results_table.add_column("Status", justify="center")
            results_table.add_column("Cases", justify="right")
            results_table.add_column("Success Rate", justify="right")
            results_table.add_column("Details", style="dim")

            for result in self.results:
                if result.error:
                    status = "⚠️"
                    cases = "N/A"
                    success_rate = "N/A"
                    details = f"Error: {str(result.error)[:50]}"
                elif result.report:
                    crashed_cases = len(getattr(result.report, "failures", []))
                    total_cases = len(result.report.cases) + crashed_cases

                    passed_cases = sum(
                        1
                        for case in result.report.cases
                        if all(assertion.value for assertion in case.assertions.values())
                    )

                    if result.success:
                        status = "✅"
                        success_rate = "[green]100%[/green]"
                    else:
                        status = "❌"
                        rate = (passed_cases / total_cases * 100) if total_cases > 0 else 0
                        success_rate = f"[yellow]{rate:.1f}%[/yellow]"

                    cases = f"{passed_cases}/{total_cases}"

                    failed_assertions = sum(
                        sum(1 for assertion in case.assertions.values() if not assertion.value)
                        for case in result.report.cases
                    )

                    details_parts = []
                    if failed_assertions > 0:
                        details_parts.append(f"{failed_assertions} failed assertions")
                    if crashed_cases > 0:
                        details_parts.append(f"{crashed_cases} crashed")
                    details = ", ".join(details_parts) if details_parts else "All passed"
                else:
                    status = "❓"
                    cases = "0"
                    success_rate = "N/A"
                    details = "No report"

                results_table.add_row(result.eval_id, status, cases, success_rate, details)

            console.print(results_table)

        if include_reports:
            for result in self.results:
                if result.report:
                    console.print()
                    console.print(Panel(f"[bold cyan]{result.eval_id}[/bold cyan]", expand=True))
                    result.report.print(include_reasons=include_reasons)

    def _print_simple(self, *, include_reports: bool = True):
        """Fallback simple print without rich library."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Functions: {self.total_count}")
        print(f"✅ Passed: {self.success_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"⚠️  Errors: {self.error_count}")
        print(f"Status: {'✅ ALL PASSED' if self.all_passed else '❌ FAILURES DETECTED'}")
        print("=" * 60)

        if self.results:
            print("\nFUNCTION RESULTS:")
            for result in self.results:
                status = "✅" if result.success else "❌" if result.report else "⚠️"
                print(f"  {status} {result.eval_id}")
                if result.error:
                    print(f"     Error: {result.error}")

        if include_reports:
            for result in self.results:
                if result.report:
                    print(f"\n{'=' * 60}")
                    print(f"{result.eval_id}")
                    print("=" * 60)
                    result.report.print()

    def to_json(self) -> dict:
        """
        Convert evaluation summary to detailed JSON format.
        Useful for LLM feedback - shows exactly which cases and evaluations failed.

        Returns:
            dict
        """
        result_dict: dict[str, Any] = {
            "summary": {
                "total_functions": self.total_count,
                "passed": self.success_count,
                "failed": self.failed_count,
                "errors": self.error_count,
                "all_passed": self.all_passed,
            },
            "results": [],
        }

        for result in self.results:
            result_data: dict[str, Any] = {
                "function": result.eval_id,
                "status": ("error" if result.error else ("passed" if result.success else "failed")),
            }

            if result.error:
                result_data["error"] = result.error
                result_data["error_type"] = "Error"

            if result.report:
                result_data["cases"] = []

                for case in result.report.cases:
                    case_passed = all(assertion.value for assertion in case.assertions.values())

                    case_output = case.output
                    if isinstance(case_output, dict) and "_exception" in case_output:
                        case_output = {
                            "_exception_type": case_output.get("_exception_type"),
                            "_exception_message": str(case_output.get("_exception")),
                        }

                    case_data: dict[str, Any] = {
                        "case_id": case.name,
                        "status": "passed" if case_passed else "failed",
                        "input": case.inputs,
                        "output": case_output,
                        "expected_output": case.expected_output,
                        "duration_ms": (
                            round(case.total_duration * 1000, 2)
                            if hasattr(case, "total_duration") and case.total_duration
                            else None
                        ),
                        "evaluations": [],
                    }

                    for eval_name, assertion in case.assertions.items():
                        eval_data = {
                            "name": eval_name,
                            "passed": assertion.value,
                        }

                        if hasattr(assertion, "reason") and assertion.reason:
                            eval_data["reason"] = assertion.reason

                        case_data["evaluations"].append(eval_data)

                    result_data["cases"].append(case_data)

                for failure in getattr(result.report, "failures", []):
                    failure_data = {
                        "case_id": failure.name,
                        "status": "error",
                        "input": failure.inputs,
                        "output": None,
                        "expected_output": failure.expected_output,
                        "error_message": failure.error_message,
                        "error_stacktrace": failure.error_stacktrace,
                        "evaluations": [],
                    }
                    result_data["cases"].append(failure_data)

            result_dict["results"].append(result_data)

        return result_dict

    def xml(self) -> str:
        """
        Convert evaluation summary to XML format.
        Uses pydantic_ai's format_as_xml for conversion.

        Returns:
            str: XML representation of the evaluation summary
        """
        return format_as_xml(self.model_dump_json())


def run_evals(
    source: str | Path | dict | EvalsFile,
    *,
    filter_funcs: list[str] | None = None,
    functions: dict[str, Callable] | None = None,
    debug: bool = False,
    schema: dict[str, type | Callable | dict[str, type | Callable]] | None = None,
    serial_fn: dict[str, Callable[[dict], Any]] | None = None,
    fixtures: dict[str, Callable | tuple[Callable, Callable | None]] | None = None,
    ignore_duration: bool = False,
) -> EvalSummary:
    """
    Run evaluations from various sources.

    Args:
        source: Evaluation source (file path, YAML string, dict, or EvalsFile)
        filter_funcs: Optional list of function names to evaluate
        functions: Optional dict of {name: function} to use instead of importing
        debug: Enable debug mode
        schema: Optional dict of input types for automatic conversion
        serial_fn: Optional dict of serializer functions (receive full input dict)
        fixtures: Optional dict of fixture functions {name: setup_fn} or {name: (setup_fn, teardown_fn)}
        ignore_duration: If True, skip duration constraints

    Returns:
        EvalSummary with aggregated results
    """
    # Load both evals and fixtures from YAML
    bundle = load_bundle(source)
    all_evals = bundle.evals
    yaml_fixtures = bundle.fixtures

    # Merge programmatic fixtures with YAML fixtures
    merged_fixtures, fixture_funcs = _merge_programmatic_fixtures(yaml_fixtures, fixtures)

    schema = schema or {}
    serial_fn = serial_fn or {}

    if filter_funcs:
        filtered_evals = {k: v for k, v in all_evals.items() if k in filter_funcs}
        if not filtered_evals:
            available = list(all_evals.keys())
            raise ValueError(
                f"No functions found matching filters: {', '.join(filter_funcs)}. "
                f"Available: {', '.join(available)}"
            )
        all_evals = filtered_evals

    # Create fixture manager if fixtures are defined
    fixture_manager = FixtureManager(merged_fixtures, fixture_funcs) if merged_fixtures else None

    results = []

    try:
        for eval_id, evals in all_evals.items():
            try:
                result = _evaluate_single_function(
                    eval_id,
                    evals,
                    functions,
                    merged_fixtures,
                    fixture_manager,
                    schema,
                    serial_fn,
                    ignore_duration,
                )
                results.append(result)
            except Exception as e:
                if debug:
                    raise
                results.append(EvalResult(eval_id, error=e))
    finally:
        # Teardown session-scoped fixtures at the end
        if fixture_manager:
            fixture_manager.teardown_all("session")

    return EvalSummary(results)

"""RunEvals - A fluent API for running evaluations.

This module provides:
- Function: Pydantic model representing a function with code and metadata
- RunEvals: Fluent API for loading and running evaluations

Example:
    # Run from YAML file
    from vowel import RunEvals

    summary = RunEvals.from_file("evals.yml").run()
    print(f"All passed: {summary.all_passed}")

    # Run with custom functions
    def my_func(x):
        return x * 2

    summary = (
        RunEvals.from_file("evals.yml")
        .with_functions({"my_func": my_func})
        .filter(["my_func"])
        .debug()
        .run()
    )
"""

import ast
import codecs
import contextlib
import inspect
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field

from .eval_types import Evals, EvalsFile, FixtureDefinition
from .utils import EvalSummary
from .utils import run_evals as _run_evals

_T = TypeVar("_T", bound=Any)
_RT = TypeVar("_RT", bound=Any)


class Function(BaseModel, Generic[_RT]):
    name: str = Field(description="The name of the function to generate evals for.")
    description: str = Field(description="A brief description of the function's purpose.")
    cases: list[str] = Field(
        default_factory=list,
        description="List of unique specialized case scenario descriptions for generating comprehensive test cases for the function to cover edge cases and diverse inputs.",
        min_length=5,
    )
    code: str = Field(
        description="The complete implementation code of the function with full type annotations. "
        "Must include type hints for all parameters and return type. "
        "Use built-in generics (list[int], dict[str, Any]) for Python 3.9+"
        " or add 'from typing import ...' import if needed."
    )
    file_path: str | None = Field(
        None,
        description="The file path where the function is defined, if applicable.",
        examples=["/path/to/module.py"],
    )

    func: Any | None = Field(
        default=None,
        exclude=True,
        description="The actual function implementation as a callable.",
    )

    @property
    def __name__(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.name

    @property
    def impl(self) -> Callable[..., _RT]:
        """
        Get the function implementation as a callable.

        Returns:
            Callable: The function implementation.
        """
        if not self.func:
            self.execute()
        return cast(Callable, self.func)

    def execute(self) -> None:
        """Execute the function code and store the callable.

        Raises:
            RuntimeError: If the code cannot be executed.
        """
        if self.func:
            return

        local_scope: dict[str, object] = {}
        try:
            code = self.code
            try:
                exec(code, local_scope, local_scope)
            except Exception:
                if "\\n" in code or '\\"' in code or "\\'" in code:
                    code = codecs.decode(code, "unicode_escape")
                    exec(code, local_scope, local_scope)
                else:
                    raise

        except Exception as e:
            raise RuntimeError(f"Error executing code for function '{self.name}'.") from e

        self.func = local_scope[self.name]

    def __call__(self, *args, **kwargs) -> _RT:
        """
        Call the function implementation with the provided arguments.
        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        Returns:
            _RT: The return value of the function.
        """
        return self.impl(*args, **kwargs)

    @property
    def func_path(self) -> str:
        if self.file_path:
            file_path = Path(self.file_path).resolve()
            current_dir = Path(os.getcwd()).resolve()

            try:
                relative_path = file_path.relative_to(current_dir)

                module_path = relative_path.with_suffix("")
                module_path_str = ".".join(module_path.parts)

                return f"{module_path_str}.{self.name}"
            except ValueError:
                return f"{file_path.stem}.{self.name}"
        return self.name

    @staticmethod
    def from_callable(func: Callable[..., _T]) -> "Function[_T]":
        """
        Create a Function instance from a callable.

        Args:
            func: The callable to wrap.

        Returns:
            Function[_RT]: The created Function instance.
        """
        name = getattr(func, "__name__", "unknown")
        description = getattr(func, "__doc__", None) or "No description provided."
        try:
            source = inspect.getsource(func)
        except Exception as e:
            raise RuntimeError(f"Could not retrieve source code for function '{name}'.") from e

        file_path = inspect.getfile(func)

        return Function(
            name=name, description=description, code=source, file_path=file_path, func=func
        )

    def print(self, *, theme: str = "monokai", show_description: bool = True) -> None:
        """Pretty print the function with syntax highlighting.

        Args:
            theme: Syntax highlighting theme (monokai, dracula, github-dark, etc.)
            show_description: Whether to show the function description
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
        except ImportError:
            self._print_simple(show_description=show_description)
            return

        console = Console()

        subtitle = (
            f"[dim]{self.description[:80]}{'...' if len(self.description) > 80 else ''}[/dim]"
            if show_description
            else None
        )

        console.print(
            Panel(
                Syntax(self.code, "python", theme=theme, line_numbers=True),
                title=f"[green]🐍 {self.name}[/green]",
                subtitle=subtitle,
            )
        )

    def _print_simple(self, *, show_description: bool = True) -> None:
        """Fallback simple print without rich library."""
        print(f"\n{'=' * 60}")
        print(f"🐍 {self.name}")
        if show_description:
            print(f"   {self.description}")
        print("=" * 60)
        print(self.code)

    def __eq__(self, fn):
        """
        Compare this Function's implementation with another callable.

        Args:
            fn: A callable or Function instance to compare against.

        Returns:
            bool: True if the provided callable is the same as this Function's implementation.
        """
        return fn == self.impl


class RunEvals:
    """
    Fluent API for running evaluations.

    Examples:
        RunEvals.from_file("evals.yml").run()
        RunEvals.from_evals(evals_obj, functions={"func": func}).run()
        RunEvals.from_evals([evals1, evals2], functions={...}).run()
        RunEvals.from_source(yaml_spec).run()
        RunEvals.from_file("evals.yml").filter(["func1", "func2"]).debug().run()
    """

    def __init__(
        self,
        source: str | Path | dict | EvalsFile | Evals | Sequence[Evals],
        *,
        functions: dict[str, Callable] | None = None,
        filter_funcs: list[str] | None = None,
        debug_mode: bool = False,
        schema: dict[str, type | Callable | dict[str, type | Callable]] | None = None,
        serial_fn: dict[str, Callable[[dict], Any]] | None = None,
        fixtures: (
            dict[str, Callable | tuple[Callable, Callable | None] | FixtureDefinition] | None
        ) = None,
        ignore_duration: bool = False,
    ):
        self._source = source
        self._functions = functions or {}
        self._filter_funcs = filter_funcs or []
        self._debug_mode = debug_mode
        self._schema = schema or {}
        self._serial_fn = serial_fn or {}
        self._fixtures = fixtures or {}
        self._ignore_duration = ignore_duration

    @classmethod
    def from_file(cls, path: str | Path) -> "RunEvals":
        """
        Create from a YAML file path.

        Args:
            path: Path to YAML file

        Returns:
            RunEvals instance

        Example:
            RunEvals.from_file("evals.yml").run()
        """
        return cls(str(path))

    @classmethod
    def from_source(cls, source: str | dict | EvalsFile) -> "RunEvals":
        """
        Create from a YAML string, dict, or EvalsFile object.

        Args:
            source: YAML string, dict, or EvalsFile

        Returns:
            RunEvals instance

        Example:
            yaml_spec = "func: {evals: {...}, dataset: [...]}"
            RunEvals.from_source(yaml_spec).run()
        """
        return cls(source)

    @classmethod
    def from_evals(
        cls,
        evals: Evals | Sequence[Evals],
        *,
        functions: dict[str, Callable] | None = None,
    ) -> "RunEvals":
        """
        Create from one or more Evals objects.

        Args:
            evals: Single Evals or sequence of Evals objects
            functions: Optional dict of {name: function} to evaluate

        Returns:
            RunEvals instance

        Examples:
            RunEvals.from_evals(evals_obj, functions={"func": func}).run()
            RunEvals.from_evals([evals1, evals2], functions={...}).run()
        """
        if isinstance(evals, Evals):
            for i, case in enumerate(evals.dataset):
                if case.case.input is not None and isinstance(case.case.input, str):
                    with contextlib.suppress(ValueError, SyntaxError):
                        evals.dataset[i].case.input = ast.literal_eval(case.case.input)

                if case.case.inputs is not None:
                    for j, inp in enumerate(case.case.inputs):
                        if isinstance(inp, str):
                            with contextlib.suppress(ValueError, SyntaxError):
                                inputs = evals.dataset[i].case.inputs
                                if inputs is not None and not isinstance(inputs, dict):
                                    inputs[j] = ast.literal_eval(inp)  # type: ignore[index]

            source_dict = {evals.id: evals.model_dump(exclude={"id"})}
        else:
            source_dict = {}
            for eval_obj in evals:
                if not isinstance(eval_obj, Evals):
                    raise TypeError(f"Expected Evals object, got {type(eval_obj)}")
                source_dict[eval_obj.id] = eval_obj.model_dump(exclude={"id"})

        return cls(source_dict, functions=functions)

    @classmethod
    def from_dict(cls, data: dict) -> "RunEvals":
        """
        Create from a dictionary.

        Args:
            data: Dictionary with eval specifications

        Returns:
            RunEvals instance

        Example:
            spec = {"func": {"evals": {...}, "dataset": [...]}}
            RunEvals.from_dict(spec).run()
        """
        return cls(data)

    def with_functions(self, functions: dict[str, Callable]) -> "RunEvals":
        """
        Add or update functions to use for evaluation.

        Args:
            functions: Dict of {name: function}

        Returns:
            Self for chaining

        Example:
            RunEvals.from_file("evals.yml").with_functions({"func": func}).run()
        """
        self._functions.update(functions)
        return self

    def filter(self, func_names: str | list[str]) -> "RunEvals":
        """
        Filter to only evaluate specific functions.

        Args:
            func_names: Single function name or list of names

        Returns:
            Self for chaining

        Example:
            RunEvals.from_file("evals.yml").filter(["func1", "func2"]).run()
        """
        if isinstance(func_names, str):
            func_names = [func_names]
        self._filter_funcs.extend(func_names)
        return self

    def debug(self, enabled: bool = True) -> "RunEvals":
        """
        Enable or disable debug mode.

        Args:
            enabled: Whether to enable debug mode

        Returns:
            Self for chaining

        Example:
            RunEvals.from_file("evals.yml").debug().run()
        """
        self._debug_mode = enabled
        return self

    def with_serializer(
        self,
        schema: dict[str, type | Callable | dict[str, type | Callable]] | None = None,
        *,
        serial_fn: dict[str, Callable[[dict], Any]] | None = None,
    ) -> "RunEvals":
        """
        Add input serializers to transform inputs before passing to functions.

        Two modes:
        1. schema: Type/callable applied to each input automatically
        2. serial_fn: Full control - receives raw {input:..} or {inputs:..} dict

        Args:
            schema: Dict mapping function names to types/callables.
                - Single type: Applied to single input or all inputs
                - Dict of types: Applied to specific named parameters
            serial_fn: Dict mapping function names to functions that receive
                the raw input dict and return serialized value(s).
                Signature: fn(input_dict: dict) -> Any

        Returns:
            Self for chaining

        Examples:
            # Mode 1: Auto serialization with type
            class User(BaseModel):
                name: str
                age: int

            RunEvals.from_file("evals.yml").with_serializer({
                "greet": User  # User(**input) called automatically
            }).run()

            # Mode 2: Full control with serial_fn
            def serialize_user(d: dict) -> User:
                data = d.get("input") or d.get("inputs")
                return User(**data)

            RunEvals.from_file("evals.yml").with_serializer(
                serial_fn={"greet": serialize_user}
            ).run()
        """
        if schema:
            self._schema.update(schema)
        if serial_fn:
            self._serial_fn.update(serial_fn)
        return self

    def with_fixtures(
        self,
        fixtures: dict[str, Callable | tuple[Callable, Callable | None] | FixtureDefinition],
    ) -> "RunEvals":
        """
        Add fixture functions for dependency injection.

        Fixtures are injected as keyword-only arguments to eval functions.
        In YAML, just specify fixture names; provide actual functions here.

        Args:
            fixtures: Dict mapping fixture names to:
                - Callable: Setup function (no teardown)
                - Tuple[Callable, Callable | None]: (setup_fn, teardown_fn)

        Returns:
            Self for chaining

        Examples:
            # Simple fixture (setup only)
            def create_db():
                return {"connected": True, "host": "localhost"}

            RunEvals.from_file("evals.yml").with_fixtures({
                "db": create_db
            }).run()

            # Fixture with teardown
            def create_db():
                db = connect_to_db()
                return db

            def close_db(db):
                db.close()

            RunEvals.from_file("evals.yml").with_fixtures({
                "db": (create_db, close_db)
            }).run()

            # YAML file:
            # my_func:
            #   fixture:
            #     - db
            #   dataset:
            #     - case: {input: {x: 1}, expected_output: 2}
        """
        self._fixtures.update(fixtures)
        return self

    def run(self) -> EvalSummary:
        """
        Execute the evaluations.

        Returns:
            EvalSummary with results

        Example:
            summary = RunEvals.from_file("evals.yml").run()
            print(f"Passed: {summary.all_passed}")
        """
        return _run_evals(
            self._source,  # type: ignore[arg-type]
            filter_funcs=self._filter_funcs,
            functions=self._functions,
            debug=self._debug_mode,
            schema=self._schema,
            serial_fn=self._serial_fn,
            fixtures=self._fixtures,
            ignore_duration=self._ignore_duration,
        )

    def ignore_duration(self) -> "RunEvals":
        """
        Ignore duration constraints during evaluation.

        Returns:
            Self for chaining

        Example:
            summary = RunEvals.from_file("evals.yml").ignore_duration().run()
        """
        self._ignore_duration = True
        return self

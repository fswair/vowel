"""
RunEvals - A fluent API for running evaluations
"""

import os
import ast
from pathlib import Path
from typing import (
    Any,
    TypeVar,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)
from pydantic import BaseModel, Field

from .eval_types import Evals, EvalsFile
from .utils import EvalSummary
from .utils import run_evals as _run_evals

_RT = TypeVar("_RT", bound=Any)


class Function(BaseModel, Generic[_RT]):
    name: str = Field(description="The name of the function to generate evals for.")
    description: str = Field(description="A brief description of the function's purpose.")
    code: str = Field(description="The complete implementation code of the function.")
    file_path: Optional[str] = Field(
        description="The file path where the function is defined, if applicable.",
        examples=["/path/to/module.py"],
    )

    func: Optional[Any] = Field(
        default=None, exclude=True, description="The actual function implementation as a callable."
    )

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def impl(self) -> Callable:
        """
        Get the function implementation as a callable.

        Returns:
            Callable: The function implementation.
        """
        if not self.func:
            self.execute()
        return cast(Callable, self.func)

    def execute(self):
        local_scope = {}
        try:
            exec(self.code, local_scope, local_scope)
        except Exception as e:
            raise e from RuntimeError(f"Error executing code for function '{self.name}'.")

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


class RunEvals:
    """
    Fluent API for running evaluations.

    Examples:
        RunEvals.from_file("evals.yml").run()
        RunEvals.from_evals(evals_obj, functions={"func": func}).run()
        RunEvals.from_evals([evals1, evals2], functions={...}).run()
        RunEvals.from_source(yaml_str).run()
        RunEvals.from_file("evals.yml").filter(["func1", "func2"]).debug().run()
    """

    def __init__(
        self,
        source: Union[str, Path, dict, EvalsFile, Evals, Sequence[Evals]],
        *,
        functions: Optional[Dict[str, Callable]] = None,
        filter_funcs: Optional[List[str]] = None,
        debug_mode: bool = False,
    ):
        self._source = source
        self._functions = functions or {}
        self._filter_funcs = filter_funcs or []
        self._debug_mode = debug_mode

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RunEvals":
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
    def from_source(cls, source: Union[str, dict, EvalsFile]) -> "RunEvals":
        """
        Create from a YAML string, dict, or EvalsFile object.

        Args:
            source: YAML string, dict, or EvalsFile

        Returns:
            RunEvals instance

        Example:
            yaml_str = "func: {evals: {...}, dataset: [...]}"
            RunEvals.from_source(yaml_str).run()
        """
        return cls(source)

    @classmethod
    def from_evals(
        cls,
        evals: Union[Evals, Sequence[Evals]],
        *,
        functions: Optional[Dict[str, Callable]] = None,
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
                    try:
                        evals.dataset[i].case.input = ast.literal_eval(case.case.input)
                    except (ValueError, SyntaxError):
                        pass

                if case.case.inputs is not None:
                    for j, inp in enumerate(case.case.inputs):
                        if isinstance(inp, str):
                            try:
                                evals.dataset[i].case.inputs[j] = ast.literal_eval(inp)
                            except (ValueError, SyntaxError):
                                pass

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

    def with_functions(self, functions: Dict[str, Callable]) -> "RunEvals":
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

    def filter(self, func_names: Union[str, List[str]]) -> "RunEvals":
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

    def run(self) -> EvalSummary:
        """
        Execute the evaluations.

        Returns:
            EvalSummary with results

        Example:
            summary = RunEvals.from_file("evals.yml").run()
            print(f"Passed: {summary.all_passed}")
        """
        kwargs = {
            "debug": self._debug_mode,
        }

        if self._filter_funcs:
            kwargs["filter_funcs"] = self._filter_funcs

        if self._functions:
            kwargs["functions"] = self._functions

        return _run_evals(self._source, **kwargs)

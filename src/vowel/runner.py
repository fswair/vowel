"""
RunEvals - A fluent API for running evaluations
"""

import ast
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from .eval_types import Evals, EvalsFile
from .utils import EvalSummary
from .utils import run_evals as _run_evals


class RunEvals:
    """
    Fluent API for running evaluations.

    Examples:
        # From file
        RunEvals.from_file("evals.yml").run()

        # From Evals object
        RunEvals.from_evals(evals_obj, functions={"func": func}).run()

        # From multiple Evals
        RunEvals.from_evals([evals1, evals2], functions={...}).run()

        # From dict/YAML string
        RunEvals.from_source(yaml_str).run()

        # With filtering and debug
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
            # Single Evals
            RunEvals.from_evals(evals_obj, functions={"func": func}).run()

            # Multiple Evals
            RunEvals.from_evals([evals1, evals2], functions={...}).run()
        """
        # Convert Evals object(s) to dict format
        if isinstance(evals, Evals):
            for i, case in enumerate(evals.dataset):
                # Handle string inputs - convert to proper Python objects
                if case.case.input is not None and isinstance(case.case.input, str):
                    # Use ast.literal_eval to safely parse literal expressions
                    # and avoid executing arbitrary code or referencing
                    # undefined names (which caused NameError previously).
                    try:
                        evals.dataset[i].case.input = ast.literal_eval(case.case.input)
                    except (ValueError, SyntaxError):
                        # If it's not a literal (shouldn't happen if LLM obeys
                        # instructions), leave it as-is so downstream code can
                        # handle/report the invalid case.
                        pass

                # Handle string inputs list - convert each element
                if case.case.inputs is not None:
                    for j, inp in enumerate(case.case.inputs):
                        if isinstance(inp, str):
                            try:
                                evals.dataset[i].case.inputs[j] = ast.literal_eval(inp)
                            except (ValueError, SyntaxError):
                                pass

            source_dict = {evals.id: evals.model_dump(exclude={"id"})}
        else:
            # Sequence of Evals objects
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

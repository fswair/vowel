"""
vowel - A modular evaluation framework for testing functions with YAML-based specifications.

This package provides a comprehensive evaluation framework for testing Python functions
using YAML-based specifications. It supports various evaluation types including:

- Type checking (isinstance validation)
- Custom assertions (Python expressions)
- Performance constraints (duration limits)
- Input containment checks
- Regex pattern matching
- Exception validation
- LLM-based semantic evaluation

Quick Start:
    # Run evaluations from a YAML file
    from vowel import run_evals
    summary = run_evals("evals.yml")

    # Generate evals for a function using LLM
    from vowel import EvalGenerator, Function
    gen = EvalGenerator(model="openai:gpt-4o")
    func = Function(name="add", code="def add(a, b): return a + b", description="Add two numbers")
    summary = gen.generate_and_run(func, auto_retry=True)

For more information, see the documentation at:
https://github.com/fswair/vowel
"""

import importlib.metadata

__version__ = importlib.metadata.version("vowel")

from .ai import EvalGenerator, GenerationResult, UnsupportedParameterTypeError
from .context import EVAL_SPEC_CONTEXT
from .errors import FixturePathError, SignatureError
from .eval_types import EvalsFile
from .runner import Function, RunEvals
from .utils import (
    EvalResult,
    EvalSummary,
    check_compatibility,
    get_unsupported_params,
    is_yaml_serializable_type,
    load_evals,
    load_evals_file,
    load_evals_from_dict,
    load_evals_from_object,
    load_evals_from_yaml_string,
    run_evals,
    to_dataset,
)

__all__ = [
    "load_evals_file",
    "load_evals_from_yaml_string",
    "load_evals_from_dict",
    "load_evals_from_object",
    "load_evals",
    "to_dataset",
    "run_evals",
    "RunEvals",
    "EvalResult",
    "EvalSummary",
    "EvalsFile",
    "EvalGenerator",
    "GenerationResult",
    "Function",
    "EVAL_SPEC_CONTEXT",
    "UnsupportedParameterTypeError",
    "SignatureError",
    "FixturePathError",
    "check_compatibility",
    "get_unsupported_params",
    "is_yaml_serializable_type",
]

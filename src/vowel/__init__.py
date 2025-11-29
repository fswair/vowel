"""
vowel - A modular evaluation framework for testing functions with YAML-based specifications.
"""
import importlib.metadata

__version__ = importlib.metadata.version("vowel")

from .eval_types import EvalsFile
from .runner import RunEvals
from .utils import (
    EvalResult,
    EvalSummary,
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
]

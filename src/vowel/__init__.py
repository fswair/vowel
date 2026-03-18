"""Public package exports for the vowel evaluation framework."""

import importlib.metadata
from contextlib import suppress

__version__ = importlib.metadata.version("vowel")

from .ai import EvalGenerator, GenerationResult, UnsupportedParameterTypeError
from .codemode import CodeModeGenerator, CodeModeResult, ExplorationPlan, SnippetResult
from .context import EVAL_SPEC_CONTEXT
from .costs import CostManager
from .errors import FixturePathError, SignatureError
from .eval_types import EvalsFile
from .executor import (
    DefaultExecutor,
    DefaultSession,
    ExecutionResult,
    ExecutionSession,
    Executor,
    MontyExecutor,
    MontyReplSession,
    get_executor,
    resolve_executors,
)
from .runner import Function, RunEvals
from .schema import ensure_cached_schema
from .utils import (
    EvalResult,
    EvalSummary,
    check_compatibility,
    get_unsupported_params,
    is_yaml_serializable_type,
    load_bundle,
    load_bundle_file,
    load_bundle_from_dict,
    load_bundle_from_object,
    load_bundle_from_yaml_string,
    run_evals,
    to_dataset,
)

__all__ = [
    "load_bundle_file",
    "load_bundle_from_yaml_string",
    "load_bundle_from_dict",
    "load_bundle_from_object",
    "load_bundle",
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
    # CodeMode executor
    "Executor",
    "ExecutionResult",
    "ExecutionSession",
    "MontyExecutor",
    "MontyReplSession",
    "DefaultExecutor",
    "DefaultSession",
    "get_executor",
    "resolve_executors",
    # CodeMode pipeline
    "CodeModeGenerator",
    "CodeModeResult",
    "ExplorationPlan",
    "SnippetResult",
    "CostManager",
]


with suppress(Exception):
    ensure_cached_schema(__version__)

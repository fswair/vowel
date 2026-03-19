# CHANGELOG

## codemode_driven_generation

This document summarizes the main features added or improved on this branch.

## 1) Executor and ExecutionSession protocols

- The code execution interface was formalized using Protocols.
- The Executor async/sync API was standardized:
  - execute(...)
  - execute_sync(...)
  - create_session(...)
- ExecutionSession now compiles/executes setup code once and supports multi-snippet feed execution.
- This reduces repeated parse/compile overhead while exploring the same function.
- The run_sync helper was hardened for running-loop environments via nest-asyncio.

## 2) MontyExecutor, DefaultExecutor, MontySession, FallbackSession structures

- MontyExecutor was added:
  - sandboxed execution via pydantic-monty,
  - ResourceLimits support (timeout/memory),
  - stdout capture and normalized error typing/messages,
- DefaultExecutor was added/improved:
  - pure Python exec-based fallback execution,
  - last-expression capture (__result__) and stdout capture.
- MontyReplSession (MontySession role) was added:
  - one-time setup load, reusable feed-run model.
- FallbackSession was added:
  - Session-level fallback: if Monty session initialization fails, switch entirely to DefaultSession.
  - Snippet-level fallback: if Monty returns ModuleNotFoundError for a snippet, rerun that snippet via fallback executor.
- Executor/fallback wiring was simplified through resolve_executors.

## 3) Main implementation: CodeModeGenerator

- Two-phase exploration-guided generation flow:
  - Phase 1: behavior exploration (exploration snippets + error snippets)
  - Phase 2: spec generation from verified observations
- Lazy Agent architecture:
  - explorer_agent (ExplorationPlan)
  - spec_agent (EvalsSource or EvalsBundle)
- Prompt layers were clearly separated:
  - exploration prompt: coverage, diversity, duplicate prevention
  - spec prompt: expected values from verified outputs only
- A refinement loop was added:
  - generate -> run -> failure_context -> regenerate
- Optional duration injection and a final summary run were added at the end.

## 4) Runtime hierarchy and utility usage

CodeMode hierarchy:

1. explore()
2. generate_spec()
3. validate_and_fix_spec()
4. validate_expected_values()
5. inject_missing_error_cases()
6. inject_durations() (optional)
7. validation/refinement with RunEvals

Utilities used:

- build_call_code
- build_failure_context
- validate_and_fix_spec
- validate_expected_values
- inject_missing_error_cases
- inject_durations

## 5) Cost Manager

- Generation/run cost tracking was added for CodeMode.
- Features:
  - generation_id and run_id lifecycle management,
  - step-level usage/cost recording,
  - model price resolution (genai-prices or costs.yml),
  - atomic/locked JSON persistence,
  - generation-level and run-level totals,
  - status tracking: running/completed/failed.
- The CLI costs command now supports list/by-generation/by-run views.

## 6) Serializer syntax and YAML-native serializer registry

- Top-level serializers registry support was added at EvalsFile level.
- Per-eval serializer references are now supported via serializer:.
- SerializerSpec was clarified with one-of behavior:
  - schema (string or dict)
  - serializer (callable import path)
  - not both at the same time.
- Runtime resolver additions:
  - import-path resolution,
  - cached imports (_import_path_cached),
  - per-eval resolution (_resolve_yaml_serializer_entry).
- Precedence between programmatic serializer maps and YAML serializer registry was defined.

## 7) Spec model / Exploration model separation

- Model separation in CodeModeGenerator constructor was formalized:
  - spec_model
  - exploration_model
- use_model_spec output mode was clarified:
  - use_model_spec=True: structured output mode (schema/model output via EvalsBundle)
  - use_model_spec=False: YAML string output mode (via EvalsSource.yaml_spec)
- HIGHLY RECOMMENDED TO KEEP use_model_spec=False.
- Model resolution order and env fallback logic were added.
- Cost tracking now supports separate model usage across separate steps.

## 8) Adding executor/fallback executor to utilities

- Utility flows were updated to accept executor and fallback executor parameters.
- Monty -> Default fallback behavior was generalized in execution-aware paths.
- Executor behavior was centralized across run_evals and validation stages.

## 9) YAML schema generator

- Runtime-model-driven schema generation was improved:
  - supports top-level fixtures + serializers,
  - preserves function-level EvalsMapValue behavior.
- Schema cache strategy was updated:
  - content-hash-based filename (reduces stale editor cache issues).
- File header updates are handled safely via materialize_yaml_with_schema_header.

## 10) CLI komutları: schema, costs

- vowel schema <file>:
  - update schema header after YAML + pydantic validation
- vowel schema --create [path]:
  - direct schema JSON generation
- vowel costs:
  - --list
  - --by-generation
  - --by-run
  - --generation <id>
  - --run <id>

## 11) module.function -> function alias support

- Alias support was added for programmatic mapping resolution:
  - function map
  - serializer schema map
  - serializer function map
- Behavior:
  - exact match first,
  - short-name fallback,
  - explicit error for ambiguous reverse short-name mapping.

## 12) Feedback-guided exploration

- A targeted Round-2 exploration flow was added:
  - build cluster summaries from Round-1 results,
  - generate snippets focused on uncovered behavior classes.
- Duplicate/semantic repetition minimization was reinforced at prompt level.
- Distinct failure-mode coverage was improved for error snippets.
- Additional rounds now measure value via new-behavior counting.

## 13) Assertion + serializer integration

- AssertionEvaluator input context is now serializer-aware.
- Assertions now see serialized input for schema, serial_fn, and nested/dict schema modes.
- This behavior is covered by regression tests.

## 14) LLM Judge env-ref improvements

- create_llm_judge now supports $ENV_VAR resolution for rubric/model fields.
- Missing env refs now produce clearer errors.

## 15) Examples, documentation, and test coverage

- A runnable native serializer + fixture example was added.
- README and serializer docs were updated with serializer/assertion context notes.
- Meaningful id fields were added to eval cases under examples.
- New/updated tests include:
  - test_schema
  - test_llm_judge_env_refs
  - serializer assertion regressions
  - YAML/native serializer parsing tests

## 16) Fixture scope alias support

- Fixture scopes now support clearer canonical names:
  - case
  - eval
  - file
- Backward-compatible aliases are still accepted:
  - function (alias of case)
  - module (alias of eval)
  - session (alias of file)
- At parse time, canonical names are normalized to legacy internal runtime values:
  - case -> function
  - eval -> module
  - file -> session
- This keeps existing runtime lifecycle behavior unchanged while allowing more descriptive scope names in YAML.

Note: Old names would be deprecated after v1.0.0

## Note

This changelog is based on features observed and validated in code on this branch, without using git history.

# CodeMode

CodeMode is Vowel's exploration-guided evaluation spec generator.

Instead of generating test specs from description only, CodeMode runs exploration snippets against real function code first, then generates and refines eval specs using verified outputs and observed errors.

## Pipeline Overview

CodeMode runs in phases:

1. Explore behavior
- Generates normal snippets and error snippets.
- Executes snippets against the target function.
- Collects real outputs, exceptions, and timings.

2. Generate spec
- Builds a spec prompt from verified execution results.
- Produces either YAML text or structured bundle output.

3. Validate and refine
- Runs generated evals against the function.
- If coverage is below target, builds failure context and retries.
- Repeats up to max refinement rounds.

4. Optional duration injection
- Measures runtime and injects duration thresholds into cases.

5. Final summary
- Returns a CodeModeResult with exploration artifacts, final YAML spec, and optional EvalSummary.

## Core API

CodeMode class:
- `vowel.codemode.CodeModeGenerator`

Result type:
- `vowel.codemode.CodeModeResult`

Main entrypoint:
- `await CodeModeGenerator.generate(...)`

## Model Configuration

Constructor model resolution order:

- `spec_model` argument, else `SPEC_MODEL`, else fallback `model`/`MODEL_NAME`
- `exploration_model` argument, else `EXPLORATION_MODEL`, else fallback `model`/`MODEL_NAME`

Both models must resolve to non-empty values.

## Output Modes (`use_model_spec`)

- `use_model_spec=False` (default)
  - Spec agent output type: `EvalsSource`
  - Generates YAML string via `yaml_spec`

- `use_model_spec=True`
  - Spec agent output type: `EvalsBundle`
  - Generates structured model output first, then can be converted to YAML

Recommendation used in this repository benchmark flow:
- HIGHLY RECOMMENDED TO KEEP `use_model_spec=False`.

## Minimal Example

```python
import asyncio

from vowel.codemode import CodeModeGenerator
from vowel.runner import Function

func = Function(
    name="flatten",
    description="Recursively flatten an arbitrarily nested list.",
    code="""
def flatten(lst: list) -> list:
    if not isinstance(lst, list):
        raise TypeError(f'Expected list, got {type(lst).__name__}')
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
)

async def main() -> None:
    gen = CodeModeGenerator(
        spec_model="openrouter:google/gemini-3-flash-preview",
        exploration_model="openrouter:google/gemini-3.1-flash-lite-preview",
        use_model_spec=False,
    )

    result = await gen.generate(
        func,
        run_evals=True,
        max_refinement_rounds=2,
        min_coverage=1.0,
        inject_durations=False,
        save_to_file=True,
    )

    print(result.yaml_spec)
    if result.summary:
        result.summary.print()

asyncio.run(main())
```

## `generate(...)` Parameters

Important flags in `CodeModeGenerator.generate`:

- `run_id`: optional run identifier for cost tracking
- `run_evals`: run generated spec after generation
- `save_to_file`: write `<function_name>_evals.yml`
- `max_refinement_rounds`: retry/refinement budget
- `min_coverage`: stop threshold (default 1.0)
- `inject_durations`: inject measured duration checks

## What `CodeModeResult` Contains

- `exploration_results`: snippet execution results
- `yaml_spec`: final YAML eval spec
- `summary`: EvalSummary when `run_evals=True`
- `refinement_rounds`: number of refinement retries used

## Benchmark Integration (`python -m codemode_benchmark`)

Benchmark runner path:
- `codemode_benchmark/run_benchmark.py`

Typical usage:

```bash
python -m codemode_benchmark
python -m codemode_benchmark --only flatten group_by
python -m codemode_benchmark --show-config
python -m codemode_benchmark --replay codemode_benchmark/run_20260312_181510
```

If you use Python launcher on your machine:

```bash
py -m codemode_benchmark
```

Benchmark runner compares model pairs (`spec_model`, `exploration_model`) across built-in scenarios and stores artifacts under `codemode_benchmark/run_<timestamp>/`.

## Troubleshooting

- Error: spec/exploration model not set
  - Set constructor args or env vars (`SPEC_MODEL`, `EXPLORATION_MODEL`, `MODEL_NAME`).

- Low coverage after generation
  - Increase `max_refinement_rounds`.
  - Provide clearer function descriptions.
  - Check whether the function has non-deterministic behavior.

- YAML parse/validation failures
  - Keep `use_model_spec=False` for YAML-first flow in this repo.
  - Let refinement run (`run_evals=True`) so failure context can repair issues.

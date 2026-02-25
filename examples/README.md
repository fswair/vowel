# Vowel Examples

Production-ready examples demonstrating every major vowel feature.

## Structure

```
examples/
├── basic_usage/        # Fundamentals: YAML specs, input formats, run_evals()
│   ├── evals.yml
│   ├── functions.py
│   └── run.py
├── evaluators/         # Every built-in evaluator in action
│   ├── evals.yml
│   ├── functions.py
│   └── run.py
├── fixtures/           # Dependency injection with keyword-only arg fixtures
│   ├── evals.yml
│   ├── functions.py
│   └── run.py
├── fluent_api/         # RunEvals builder: file / string / dict, filter, debug
│   ├── evals.yml
│   └── run.py
├── ai_generation/      # EvalGenerator: LLM-powered spec & function generation
│   └── run.py
└── evals/              # CLI-runnable YAML files (no Python needed)
    ├── functions.py    # Shared functions referenced by module path
    ├── builtins.yml    # Built-in & stdlib functions
    ├── strings.yml     # String utilities with evaluators
    ├── math.yml        # Math & numeric functions with assertions
    └── validation.yml  # Validation & parsing with pattern matching
```

## Running

### Programmatic (Python)

Each example is a standalone module you can run from the project root:

```bash
# Basic usage — input formats, expected values, raises
python -m examples.basic_usage.run

# Evaluators — Type, Assertion, Pattern, Duration, raises
python -m examples.evaluators.run

# Fixtures — keyword-only arg dependency injection
python -m examples.fixtures.run

# Fluent API — RunEvals builder pattern
python -m examples.fluent_api.run

# AI generation (requires MODEL_NAME env var)
python -m examples.ai_generation.run
```

### CLI (no Python code needed)

Run individual YAML files or the entire directory:

```bash
# Run a single file
vowel examples/evals/builtins.yml
vowel examples/evals/strings.yml --quiet
vowel examples/evals/math.yml --debug

# Run all YAML files in the directory
vowel -d examples/evals

# Run in CI environment
vowel -d examples/evals --ci --cov 95
vowel -d examples/evals --ci --ignore-duration

# Filter by function name
vowel -d examples/evals --filter fibonacci

# Export results as JSON
vowel -d examples/evals --export-json results.json

# Verbose summary with spec overview and case breakdown
vowel examples/evals/math.yml -v

# Verbose without pydantic_evals report
vowel examples/evals/math.yml -v --hide-report
```

## What each example covers

### basic_usage
- Running evaluations from a YAML file with `run_evals()`
- Running from a Python `dict` (no YAML needed)
- Single input (`input:`), named inputs (`inputs: {}`), positional inputs (`inputs: []`)
- Using built-in (`len`) and stdlib (`math.sqrt`, `os.path.join`) functions
- Testing exceptions with `raises:` + `match:`
- Reading `EvalSummary` attributes

### evaluators
- **Type** — strict and lenient type checking
- **Assertion** — arbitrary Python expressions (`output >= 0`, `abs(output - expected) < 0.01`)
- **Pattern** — regex validation on stringified output
- **Duration** — per-function and per-case performance limits
- **ContainsInput** — verify output contains the input value
- **Raises** — exception class + optional message matching
- Combining multiple evaluators on a single function

### fixtures
- Fixtures declared via `fixture: [name]` in YAML
- Injected as keyword-only args (after `*`) to functions
- Setup-only fixtures (callable) or setup+teardown (tuple)
- Provided via `.with_fixtures()` in Python

### fluent_api
- `RunEvals.from_file()`, `.from_source()`, `.from_dict()`
- Chaining: `.with_functions()`, `.filter()`, `.debug()`
- Coverage thresholds with `summary.meets_coverage()`
- JSON/XML export for LLM feedback loops

### ai_generation
- `EvalGenerator.generate_function()` — create a function from a description
- `EvalGenerator.generate_spec()` — auto-generate YAML eval spec
- `EvalGenerator.generate_and_run()` — full pipeline with auto-retry & healing
- Requires `MODEL_NAME` environment variable

### evals/ (CLI)
- **builtins.yml** — built-in functions (`len`, `abs`, `sorted`, `max`, `min`, `sum`, `round`, `bool`) and stdlib (`math.sqrt`)
- **strings.yml** — custom string functions with Type and Pattern evaluators
- **math.yml** — numeric functions with Assertion evaluators and `raises:` for error cases
- **validation.yml** — validation/parsing functions with Pattern matching, `raises:`, and strict type checking

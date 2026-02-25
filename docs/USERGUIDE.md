# Vowel User Guide

A YAML-based evaluation framework for testing Python functions with LLM-powered generation.

## Table of Contents

- [Quick Start](#quick-start)
- [YAML Specification](#yaml-specification)
- [Input vs Inputs](#input-vs-inputs)
- [Evaluators](#evaluators)
- [Assertion Context](#assertion-context)
- [Duration Units](#duration-units)
- [CLI Usage](#cli-usage)
- [Best Practices](#best-practices)

---

## Quick Start

### From YAML File

```python
from vowel import RunEvals

summary = RunEvals.from_file("evals.yml").run()
summary.print()
```

### From Code

```python
from vowel import EvalGenerator

generator = EvalGenerator()
func = generator.generate_function("Calculate factorial of n")
summary = generator.generate_and_run(func, auto_retry=True)
```

---

## YAML Specification

### Basic Structure

```yaml
function_name:
  evals:                    # Global evaluators (apply to ALL cases)
    EvaluatorName:
      type: int             # or assertion, duration, pattern, rubric
  
  dataset:                  # Test cases
    - case:
        id: optional_id     # Optional case identifier
        input: <value>      # For single-param functions
        expected: <value>   # Expected output (optional)
        duration: 100       # Max duration in MILLISECONDS
```

### Multiple Functions

```yaml
add:
  dataset:
    - case:
        inputs: [2, 3]
        expected: 5

multiply:
  dataset:
    - case:
        inputs: [4, 5]
        expected: 20
```

---

## Input vs Inputs

Understanding when to use `input` vs `inputs` is crucial.

### `input` - Single Parameter

Use when your function takes **one parameter**:

```yaml
# Function: def square(x): return x * x
square:
  dataset:
    - case:
        input: 5           # square(5)
        expected: 25

# Function: def process_list(items): return len(items)
process_list:
  dataset:
    - case:
        input: [1, 2, 3]   # process_list([1, 2, 3]) - list is ONE param
        expected: 3
```

### `inputs` - Multiple Parameters

Use when your function takes **multiple parameters** (as list or dict):

```yaml
# Function: def add(a, b): return a + b
add:
  dataset:
    - case:
        inputs: [2, 3]     # add(2, 3) - unpacked as two args
        expected: 5

# Named parameters
greet:
  dataset:
    - case:
        inputs:
          name: "Alice"
          greeting: "Hello"
        expected: "Hello, Alice!"
```

### `input: null` - Pass None

```yaml
handle_none:
  dataset:
    - case:
        input: null        # handle_none(None)
        expected: "default"
```

### Common Pitfall

```yaml
# ❌ WRONG - tries to call process_list(1, 2, 3)
process_list:
  dataset:
    - case:
        inputs: [1, 2, 3]

# ✅ CORRECT - calls process_list([1, 2, 3])
process_list:
  dataset:
    - case:
        input: [1, 2, 3]
```

---

## Evaluators

### Type Evaluator

Check output type:

```yaml
evals:
  IsInteger:
    type: int
  IsNumeric:
    type: "int | float"
  IsOptionalString:
    type: "str | None"
```

### Assertion Evaluator

Python expression that must evaluate to `True`:

```yaml
evals:
  IsPositive:
    assertion: "output > 0"
  CorrectSquare:
    assertion: "output == input ** 2"

# Case-level assertions
dataset:
  - case:
      input: 5
      assertion: "output % 5 == 0"
```

### Duration Evaluator

> ⚠️ **Units differ between global and case level!**

```yaml
evals:
  FastEnough:
    duration: 0.1          # Global: SECONDS (100ms)

dataset:
  - case:
      input: 100
      duration: 50         # Case: MILLISECONDS (50ms)
```

### Expected

Check exact equality:

```yaml
dataset:
  - case:
      input: 5
      expected: 25
```

### Contains

Check substring presence:

```yaml
dataset:
  - case:
      input: "hello world"
      contains: "world"
```

### Pattern Match

Regex pattern matching:

```yaml
evals:
  ValidEmail:
    pattern: "^[\\w.-]+@[\\w.-]+\\.\\w+$"
    case_sensitive: false

dataset:
  - case:
      input: "test"
      pattern: "^TEST$"
      case_sensitive: false
```

### Raises

Check exception raising:

```yaml
dataset:
  - case:
      input: -1
      raises: ValueError
      match: "must be positive"    # Optional: regex for message
```

### LLM Judge

Use an LLM to evaluate quality:

```yaml
evals:
  IsGoodSummary:
    rubric: "Is the output a good summary of the input?"
    include:
      - input
      - expected_output
    config:
      model: "openai:gpt-4o"      # Or JUDGE_MODEL env var
      temperature: 0.0
```

---

## Assertion Context

Available variables in assertions:

| Variable | Type | Description |
|----------|------|-------------|
| `input` | Any | The input value(s) passed to the function |
| `output` | Any | The actual function output |
| `expected` | Any | Expected value (if provided) |
| `duration` | float | Execution time in seconds |
| `metadata` | dict | Additional metadata (if provided) |

### Examples

```python
# Numeric
"output > 0"
"output == input ** 2"
"abs(output - expected) < 0.001"

# String
"output.isupper()"
"input in output"
"len(output) > 0"

# Collection
"len(output) == len(input)"
"all(x > 0 for x in output)"
"set(output) == set(expected)"

# Dict
"output['key'] == expected"
"'key' in output"

# Performance
"duration < 0.1"
```

### Multi-Parameter Functions

When using `inputs`, the `input` variable contains the parameters:

```python
# For inputs: [a, b, c]
"len(input) == 3"
"input[0] < input[1]"

# For inputs: {x: 1, y: 2}
"input['x'] + input['y'] == output"
```

---

## Duration Units

| Scope | Unit | Example |
|-------|------|---------|
| **Global** (`evals:`) | **Seconds** | `0.5` = 500ms |
| **Case** (`case:`) | **Milliseconds** | `500` = 500ms |

### Example

```yaml
fibonacci:
  evals:
    GlobalLimit:
      duration: 1.0        # 1 SECOND max for all cases
  
  dataset:
    - case:
        input: 5
        duration: 10       # 10 MILLISECONDS
    
    - case:
        input: 20
        duration: 100      # 100 MILLISECONDS
```

### Common Mistake

```yaml
# ❌ This is 100 SECONDS, not 100ms!
evals:
  FastEnough:
    duration: 100

# ✅ Correct - 100 milliseconds
evals:
  FastEnough:
    duration: 0.1
```

---

## Input Serializers

Input serializers handle complex YAML input types (like JSON strings) before passing to functions.

### JSON Deserialization

```yaml
# YAML string input -> Python dict
parse_config:
  schema:
    input:
      type: object
      format: json
  dataset:
    - case:
        input: '{"key": "value"}'
        expected: {key: "value"}
```

### Custom Serializer Functions

```python
from vowel import RunEvals

def custom_parser(data: str):
    return json.loads(data)

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"func": func})
    .with_serializer({"func": custom_parser})
    .run()
)
```

---

## CLI Usage

```bash
# Run single file
vowel evals.yml

# Run all YAML files in directory (recursive)
vowel --dir tests/
vowel -d tests/

# Debug mode
vowel evals.yml --debug

# Filter specific functions
vowel evals.yml --filter func1,func2
vowel evals.yml -f add,multiply

# Quiet mode (only summary)
vowel evals.yml --quiet
vowel evals.yml -q

# Disable colors (for CI logs)
vowel evals.yml --no-color

# CI mode with coverage threshold
vowel evals.yml --ci --cov 90

# Coverage warning (no exit code)
vowel evals.yml --cov 80

# Combine options
vowel -d tests/ -q --ci --cov 80
```

### CLI Options Reference

#### Core Options

| Option | Short | Description |
|--------|-------|-------------|
| `--debug` | | Enable debug mode with stack traces |
| `--dir` | `-d` | Run all YAML files in directory (recursive) |
| `--filter` | `-f` | Only run specific function(s) |
| `--quiet` | `-q` | Minimal output, only show summary |
| `--no-color` | | Disable colored output |
| `--ci` | | CI mode, exit 1 on failure |
| `--cov` / `--coverage` | | Required coverage percentage (default: 100) |

#### Information Options

| Option | Description |
|--------|-------------|
| `--list-fixtures` | List fixtures with setup/teardown/scope/usage |
| `--dry-run` | Show test plan without running tests |
| `--fixture-tree` | Show fixture dependency tree |
| `--export-json PATH` | Export results to JSON file |
| `--verbose` / `-v` | Detailed summary: spec overview, per-case breakdown, pass rate |
| `--hide-report` | Hide pydantic_evals report output |

#### Examples

```bash
# List all fixtures with details
vowel evals.yml --list-fixtures

# Dry run - see what will be tested
vowel evals.yml --dry-run

# Show fixture dependency tree
vowel evals.yml --fixture-tree

# Export results to JSON
vowel evals.yml --export-json results.json

# Verbose summary (spec overview, case breakdown, pass rate)
vowel evals.yml -v

# Verbose without pydantic_evals report output
vowel evals.yml -v --hide-report

# Combine options
vowel -d tests/ -q --ci --cov 80 --export-json ci-results.json
```

### Coverage Behavior

- `--cov 90`: Shows warning if coverage < 90%, does NOT exit
- `--ci --cov 90`: Exits with code 1 if coverage < 90%

---

## Best Practices

### 1. Keep Assertions Simple

```yaml
# ❌ Too complex
evals:
  Everything:
    assertion: "output > 0 and output < 100 and isinstance(output, int)"

# ✅ Split into multiple
evals:
  IsInteger:
    type: int
  IsPositive:
    assertion: "output > 0"
  InRange:
    assertion: "0 < output < 100"
```

### 2. Use Descriptive Names

```yaml
evals:
  IsPositive: ...           # ✅ Clear
  Check1: ...               # ❌ Unclear
```

### 3. Use Expected When Available

```yaml
# ❌ Hardcoded
assertion: "output == 25"

# ✅ Dynamic
assertion: "output == expected"
assertion: "output == input ** 2"
```

### 4. No Side Effects in Assertions

```yaml
# ❌ Modifies input
assertion: "input.append(5) and True"

# ✅ Read-only
assertion: "5 in input"
```

### 5. Handle Edge Cases

```yaml
dataset:
  - case:
      input: 0              # Zero
  - case:
      input: -1             # Negative
  - case:
      input: null           # None
  - case:
      input: []             # Empty
```

---

## Complete Example

```yaml
# calculator.yml
add:
  evals:
    IsNumber:
      type: "int | float"
    FastEnough:
      duration: 0.01        # 10ms (seconds)
  
  dataset:
    - case:
        id: basic
        inputs: [2, 3]
        expected: 5
    
    - case:
        id: negative
        inputs: [-5, 3]
        expected: -2
    
    - case:
        id: floats
        inputs: [1.5, 2.5]
        expected: 4.0

divide:
  evals:
    IsFloat:
      type: float
  
  dataset:
    - case:
        inputs: [10, 2]
        expected: 5.0
    
    - case:
        id: division_by_zero
        inputs: [10, 0]
        raises: ZeroDivisionError
```

```python
from vowel import RunEvals

summary = RunEvals.from_file("calculator.yml").run()
summary.print()
```

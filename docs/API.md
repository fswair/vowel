# Vowel API Reference

## Table of Contents

- [RunEvals](#runevals)
- [EvalGenerator](#evalgenerator)
- [Data Types](#data-types)

---

## RunEvals

Fluent API for running evaluations.

### Factory Methods

#### `from_file(path)`

Load evaluations from a YAML file.

```python
from vowel import RunEvals

summary = RunEvals.from_file("evals.yml").run()
```

#### `from_source(source)`

Load from YAML string, dict, or EvalsFile object.

```python
yaml_str = """
square:
  dataset:
    - case:
        input: 5
        expected: 25
"""
summary = RunEvals.from_source(yaml_str).run()
```

#### `from_dict(data)`

Load from a dictionary.

```python
spec = {
    "func": {
        "evals": {"IsInteger": {"type": "int"}},
        "dataset": [{"case": {"input": 2, "expected": 4}}]
    }
}
summary = RunEvals.from_dict(spec).run()
```

#### `from_evals(evals, *, functions=None)`

Load from Evals object(s). Best for LLM-generated specs.

```python
from vowel.eval_types import Evals

evals_obj = Evals(
    id="my_func",
    evals={"IsInteger": {"type": "int"}},
    dataset=[{"case": {"input": 2, "expected": 4}}]
)

summary = RunEvals.from_evals(
    evals_obj,
    functions={"my_func": my_func}
).run()

# Multiple Evals
summary = RunEvals.from_evals(
    [evals1, evals2],
    functions={"func1": func1, "func2": func2}
).run()
```

### Chaining Methods

#### `.filter(func_names)`

Filter to specific functions.

```python
RunEvals.from_file("evals.yml").filter(["func1", "func2"]).run()
RunEvals.from_file("evals.yml").filter("func1").run()
```

#### `.with_functions(functions)`

Provide custom function implementations.

```python
def my_func(x):
    return x * 2

RunEvals.from_file("evals.yml").with_functions({"my_func": my_func}).run()
```

#### `.debug(enabled=True)`

Enable debug mode for detailed traces.

```python
RunEvals.from_file("evals.yml").debug().run()
```

#### `.run()`

Execute evaluations and return `EvalSummary`.

```python
summary = RunEvals.from_file("evals.yml").run()
print(f"Passed: {summary.all_passed}")
print(f"Coverage: {summary.coverage * 100}%")
```

### Result Methods

```python
summary = RunEvals.from_file("evals.yml").run()

summary.print()                        # Print results
summary.print(include_reasons=True)    # With detailed reasons
summary.to_json()                      # JSON for LLM feedback
summary.xml()                          # XML export
```

---

## EvalGenerator

LLM-powered function and eval generation.

### Constructor

```python
from vowel import EvalGenerator

generator = EvalGenerator(
    model="openrouter:google/gemini-2.0-flash-001",  # Or MODEL_NAME env var
    additional_context="Include edge cases",
)
```

### Methods

#### `generate_function(prompt, async_func=True)`

Generate a function from description.

```python
func = generator.generate_function("Calculate factorial of n")
print(func.code)
```

Returns: `Function` object

#### `generate_spec(func, **kwargs)`

Generate eval specification for a function.

```python
runner = generator.generate_spec(
    func,
    additional_context="Test edge cases",
    save_to_file=True,
    retries=5,
)
summary = runner.run()
```

Returns: `RunEvals` instance

#### `generate_and_run(func, **kwargs)`

Generate spec and run with optional retry/healing.

```python
summary = generator.generate_and_run(
    func,
    additional_context="Extra context",
    auto_retry=True,
    max_retries=3,
    retry_delay=5.0,
    min_coverage=0.9,
    heal_function=True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | Function | required | Function to test |
| `additional_context` | str | None | Extra context for LLM |
| `auto_retry` | bool | False | Retry on failure |
| `max_retries` | int | 3 | Max retry attempts |
| `retry_delay` | float | 5.0 | Seconds between retries |
| `min_coverage` | float | 0.9 | Required pass rate (0-1) |
| `heal_function` | bool | False | Fix buggy functions |

Returns: `EvalSummary`

### Retry Logic Flow

```
generate_and_run()
    │
    ├─► Run evals
    │
    ├─► Check results:
    │   │
    │   ├─► ERRORS (crashes)
    │   │   └─► Regenerate eval spec
    │   │
    │   ├─► FAILURES (wrong output)
    │   │   └─► Heal function code
    │   │
    │   └─► Coverage < threshold
    │       └─► Regenerate better evals
    │
    └─► Repeat until success or max_retries
```

### Example: Complete Workflow

```python
from vowel import EvalGenerator

generator = EvalGenerator(debug=True)

# Generate function
func = generator.generate_function(
    "Check if a number is prime"
)

# Generate and run with healing
summary = generator.generate_and_run(
    func,
    auto_retry=True,
    heal_function=True,
    min_coverage=0.9,
)

# Check results
if summary.all_passed:
    print("✅ All tests passed!")
else:
    print(f"Coverage: {summary.coverage * 100:.1f}%")
```

---

## Data Types

### Function

```python
from vowel import Function

func = Function(
    name="add",
    description="Add two numbers",
    code="def add(a, b): return a + b",
    func=lambda a, b: a + b,  # Optional callable
)
```

### Evals

```python
from vowel.eval_types import Evals

evals = Evals(
    id="my_function",
    evals={
        "IsInteger": {"type": "int"},
        "IsPositive": {"assertion": "output > 0"},
    },
    dataset=[
        {"case": {"input": 5, "expected": 25}},
        {"case": {"inputs": [2, 3], "expected": 5}},
    ]
)
```

### EvalSummary

Returned by `.run()`:

```python
summary.all_passed      # bool: All tests passed
summary.success_count   # int: Number passed
summary.failed_count    # int: Number failed
summary.total_count     # int: Total tests
summary.coverage        # float: Pass rate (0-1)

summary.meets_coverage(0.9)  # bool: Check threshold
summary.print()              # Print formatted results
summary.to_json()            # JSON representation
summary.xml()                # XML representation
```

---

## Quick Reference

### Load & Run

```python
# From file
RunEvals.from_file("evals.yml").run()

# From string
RunEvals.from_source(yaml_str).run()

# From dict
RunEvals.from_dict(spec_dict).run()

# From Evals object
RunEvals.from_evals(evals_obj, functions={"f": f}).run()
```

### With Options

```python
RunEvals.from_file("evals.yml") \
    .filter(["func1", "func2"]) \
    .with_functions({"func1": impl}) \
    .debug() \
    .run()
```

### Generate & Test

```python
generator = EvalGenerator()
func = generator.generate_function("description")
summary = generator.generate_and_run(func, auto_retry=True)
```

### CI/CD Usage

```python
import sys
from vowel import RunEvals

summary = RunEvals.from_file("evals.yml").run()

if not summary.all_passed:
    print(f"❌ {summary.failed_count} tests failed")
    sys.exit(1)

print("✅ All tests passed")
```

---

## MCP Server

Vowel provides an MCP (Model Context Protocol) server for integration with AI assistants.

### Running the Server

```bash
python -m vowel.mcp_server
```

### MCP Configuration

Add to your MCP client config:

```json
{
    "mcpServers": {
        "vowel": {
            "command": "python",
            "args": ["-m", "vowel.mcp_server"]
        }
    }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `run_evals_from_file` | Run evals from a YAML file |
| `run_evals_from_yaml` | Run evals from YAML content string |
| `generate_function` | Generate a function from description |
| `generate_eval_spec` | Generate eval spec for a function |
| `generate_and_run` | Full workflow: description → function → evals → results |
| `list_yaml_files` | List YAML files in a directory |

### Available Resources

| Resource | Description |
|----------|-------------|
| `vowel://context` | Eval specification documentation |
| `vowel://example` | Example YAML eval specification |

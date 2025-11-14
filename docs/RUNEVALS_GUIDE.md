# RunEvals - Fluent API Guide

## Overview

`RunEvals` provides a clean, fluent interface for running evaluations with method chaining support.

## Quick Examples

```python
from vowel import RunEvals

# From file
RunEvals.from_file("evals.yml").run()

# From Evals object with custom function
RunEvals.from_evals(evals_obj, functions={"func": func}).run()

# With filtering and debug
RunEvals.from_file("evals.yml").filter(["func1", "func2"]).debug().run()
```

## Input vs Inputs

Understanding when to use `input` vs `inputs` is crucial for correct test specification:

### Use `input` for Single-Parameter Functions

When your function accepts **one parameter**, use the `input` field:

```python
# Function with one parameter
def square(x: int) -> int:
    return x * x

def process_list(items: list) -> int:
    return len(items)
```

```yaml
square:
  dataset:
    - case:
        input: 5 # square(5)
        expected: 25

process_list:
  dataset:
    - case:
        input: [1, 2, 3] # process_list([1, 2, 3]) - list is ONE parameter
        expected: 3
```

### Use `inputs` for Multi-Parameter Functions

When your function accepts **multiple parameters**, use the `inputs` field with a list:

```python
# Function with multiple parameters
def add(x: int, y: int) -> int:
    return x + y

def max_of_three(a: int, b: int, c: int) -> int:
    return max(a, b, c)
```

```yaml
add:
  dataset:
    - case:
        inputs: [2, 3] # add(2, 3) - unpacks as two arguments
        expected: 5

max_of_three:
  dataset:
    - case:
        inputs: [10, 5, 8] # max_of_three(10, 5, 8) - unpacks as three arguments
        expected: 10
```

### Built-in Examples

```yaml
# len() - single parameter (takes one collection)
len:
  dataset:
    - case:
        input: [1, 2, 3, 4] # len([1, 2, 3, 4])
        expected: 4

# max() - multiple parameters (variadic)
max:
  dataset:
    - case:
        inputs: [5, 10, 3] # max(5, 10, 3)
        expected: 10

# pow() - two parameters
pow:
  dataset:
    - case:
        inputs: [2, 3] # pow(2, 3)
        expected: 8
```

### Common Pitfall

❌ **Wrong** - Using `inputs` for a single-parameter function that takes a list:

```yaml
# WRONG! This tries to call process_items(1, 2, 3)
process_items:
  dataset:
    - case:
        inputs: [1, 2, 3]
```

✅ **Correct** - Using `input` because the function expects ONE list parameter:

```yaml
# Correct! This calls process_items([1, 2, 3])
process_items:
  dataset:
    - case:
        input: [1, 2, 3]
```

## API Methods

### Factory Methods

#### `RunEvals.from_file(path)`

Load evaluations from a YAML file.

```python
summary = RunEvals.from_file("evals.yml").run()
```

#### `RunEvals.from_source(source)`

Load from YAML string, dict, or EvalsFile object.

```python
yaml_str = """
str:
  evals:
    NonString:
        assertion: not isinstance(input, str)
  dataset:
    - case:
        input: 1
        contains: "1"
"""
summary = RunEvals.from_source(yaml_str).run()
```

#### `RunEvals.from_dict(data)`

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

#### `RunEvals.from_evals(evals, *, functions=None)`

Load from one or more `Evals` objects. **Most powerful for LLM-generated specs!**

```python
from vowel.eval_types import Evals

# Single Evals
evals_obj = Evals(
    id="my_func",
    evals={"IsInteger": {"type": "int"}},
    dataset=[{"case": {"input": 2, "expected": 4}}]
)
summary = RunEvals.from_evals(evals_obj, functions={"my_func": my_func}).run()

# Multiple Evals
summary = RunEvals.from_evals(
    [evals1, evals2],
    functions={"func1": func1, "func2": func2}
).run()
```

### Chaining Methods

#### `.filter(func_names)`

Filter to only evaluate specific functions.

```python
RunEvals.from_file("evals.yml").filter(["func1", "func2"]).run()

# Single function
RunEvals.from_file("evals.yml").filter("func1").run()
```

#### `.with_functions(functions)`

Add custom functions to use instead of importing.

```python
def custom_func(x):
    return x * 2

RunEvals.from_file("evals.yml").with_functions({"custom_func": custom_func}).run()
```

#### `.debug(enabled=True)`

Enable debug mode to see detailed error traces.

```python
RunEvals.from_file("evals.yml").debug().run()

# Disable debug
RunEvals.from_file("evals.yml").debug(False).run()
```

#### `.run()`

Execute the evaluations and return `EvalSummary`.

```python
summary = RunEvals.from_file("evals.yml").run()
print(f"Passed: {summary.all_passed}")
```

## Complete Examples

### Example 1: LLM-Generated Function Testing

```python
from vowel import RunEvals
from vowel.eval_types import Evals
from pydantic_ai import Agent

# Generate function spec with LLM
eval_generator = Agent("groq:qwen/qwen3-32b")
result = eval_generator.run_sync(
    "Generate an Evals spec for is_prime function",
    output_type=Evals
)

# Define the function
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Test it - super clean!
summary = (
    RunEvals.from_evals(result.output, functions={"is_prime": is_prime})
    .debug()
    .run()
)

print(f"All passed: {summary.all_passed}")
```

### Example 2: Testing Multiple Custom Functions

```python
from vowel import RunEvals
from vowel.eval_types import Evals

def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

evals1 = Evals(
    id="add",
    evals={"IsInteger": {"type": "int"}},
    dataset=[{"case": {"inputs": [2, 3], "expected": 5}}]  # add(2, 3)
)

evals2 = Evals(
    id="multiply",
    evals={"IsInteger": {"type": "int"}},
    dataset=[{"case": {"inputs": [2, 3], "expected": 6}}]  # multiply(2, 3)
)

summary = RunEvals.from_evals(
    [evals1, evals2],
    functions={"add": add, "multiply": multiply}
).run()

print(f"Success: {summary.success_count}/{summary.total_count}")
```

### Example 3: Production CI/CD Pipeline

```python
from vowel import RunEvals
import sys

summary = (
    RunEvals.from_file("tests/evals.yml")
    .filter(["critical_func1", "critical_func2"])
    .run()
)

if not summary.all_passed:
    print(f"❌ {summary.failed_count} tests failed")
    for result in summary.failed_results:
        print(f"  - {result.eval_id}")
    sys.exit(1)

print("✅ All tests passed")
sys.exit(0)
```

### Example 4: Dynamic Function Registry

```python
from vowel import RunEvals

# Build function registry dynamically
functions = {}
for name, func in my_module.__dict__.items():
    if callable(func) and not name.startswith("_"):
        functions[name] = func

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions(functions)
    .run()
)
```

## Comparison: Old vs New API

### Old Way (Still Supported)

```python
from vowel.eval_types import Evals
from vowel import run_evals

evals_obj = Evals(...)

evals_dict = {evals_obj.id: evals_obj.model_dump(exclude={"id"})}
summary = run_evals(evals_dict, functions={"is_prime": is_prime}, debug=True)
```

### New Way (Cleaner!)

```python
from vowel import RunEvals

summary = (
    RunEvals.from_evals(Evals(...), functions={"is_prime": is_prime})
    .debug()
    .run()
)
```

## Benefits

- ✨ **Fluent Interface**: Chain methods for readable code
- 🎯 **Type-Safe**: Full IDE autocomplete support
- 🚀 **Flexible**: Multiple factory methods for different use cases
- 🔧 **LLM-Friendly**: Perfect for dynamic function generation
- 📦 **No Side Effects**: Doesn't modify global state
- 🧪 **Testable**: Easy to mock and test

## When to Use What

| Use Case               | Method                       |
| ---------------------- | ---------------------------- |
| Load from file         | `RunEvals.from_file()`       |
| YAML string content    | `RunEvals.from_source()`     |
| Dictionary spec        | `RunEvals.from_dict()`       |
| LLM-generated Evals    | `RunEvals.from_evals()`      |
| Multiple Evals objects | `RunEvals.from_evals([...])` |
| Custom functions       | Use `functions=` parameter   |
| Filter specific funcs  | Use `.filter()`              |
| Debug mode             | Use `.debug()`               |

## Advanced Patterns

### Pattern 1: Conditional Debugging

```python
import os

summary = (
    RunEvals.from_file("evals.yml")
    .debug(os.getenv("DEBUG") == "1")
    .run()
)
```

### Pattern 2: Progressive Filtering

```python
runner = RunEvals.from_file("evals.yml")

if quick_test:
    runner = runner.filter(["fast_func1", "fast_func2"])

summary = runner.run()
```

### Pattern 3: Function Injection

```python
def create_runner(evals_obj, func_impl):
    return (
        RunEvals.from_evals(evals_obj, functions={evals_obj.id: func_impl})
        .debug()
    )

summary = create_runner(llm_generated_evals, my_function).run()

summary.print(include_reasons=True)

summary.json() # You can give eval report to LLM as json
```

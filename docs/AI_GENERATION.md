# AI-Powered Generation

vowel provides two AI generators for automated test spec generation and function creation.

| Generator | Input | Flow | Use Case |
|-----------|-------|------|----------|
| `EvalGenerator` | Existing function code | Code → Tests | Test existing functions |
| `TDDGenerator` | Description (intent) | Signature → Tests → Code | New function from scratch |

---

## EvalGenerator

Uses LLMs to automatically generate test specs and heal buggy functions.

### Basic Usage

```python
from vowel import EvalGenerator, Function

generator = EvalGenerator(
    model="openai:gpt-4o",
    load_env=True
)

func = Function(
    name="is_palindrome",
    description="Check if a string is a palindrome (case-insensitive)",
    code="""
def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]
"""
)

# Generate and run evals — returns GenerationResult
result = generator.generate_and_run(func)

print(f"Coverage: {result.summary.coverage * 100:.1f}%")
print(f"Was healed: {result.was_healed}")
print(f"Final YAML:\n{result.yaml_spec}")
print(f"Final function code:\n{result.func.code}")
```

### GenerationResult

```python
result = generator.generate_and_run(func)

# Properties
result.yaml_spec    # str: Final YAML specification
result.func         # Function: Final function (healed or original)
result.summary      # EvalSummary: Test results
result.was_healed   # bool: Whether function was healed

# Pretty print with rich formatting
result.print()                           # Full output
result.print(show_yaml=False)            # Hide YAML
result.print(show_func=False)            # Hide function code
result.print(show_summary=False)         # Hide summary table
result.print(theme="dracula")            # Change syntax theme
```

### Auto-Retry and Healing

```python
result = generator.generate_and_run(
    func,
    auto_retry=True,      # Retry on failures
    max_retries=3,        # Maximum retry attempts
    retry_delay=5.0,      # Delay between retries (seconds)
    min_coverage=0.9,     # Target 90% coverage
    heal_function=True,   # Auto-fix buggy functions
)

if result.was_healed:
    print("Function was fixed!")
    print(f"New code:\n{result.func.code}")
```

**Retry Flow:**

1. Generate eval spec → Run tests
2. If errors occur → Regenerate spec with error context
3. If tests fail but no errors:
   - If `heal_function=True` → LLM fixes the function
   - If heal improves coverage → Continue with healed function
   - If not → Regenerate test cases
4. Repeat until `min_coverage` met or `max_retries` exhausted

### Generate Spec Only

```python
# Returns (RunEvals, yaml_spec)
runner, yaml_spec = generator.generate_spec(
    func,
    additional_context="Focus on edge cases",
    save_to_file=True,  # Saves to {func.name}_evals.yml
    retries=5
)

summary = runner.run()
```

### Generate Function from Prompt

```python
func = generator.generate_function(
    prompt="Create a function that calculates Fibonacci numbers",
    async_func=False
)

print(func.code)
# def fibonacci(n: int) -> int:
#     if n <= 1:
#         return n
#     return fibonacci(n - 1) + fibonacci(n - 2)
```

### Function from Callable

```python
from vowel import Function

def existing_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

func = Function.from_callable(existing_function)
result = generator.generate_and_run(func)
```

### EvalGenerator Parameters

```python
generator = EvalGenerator(
    model="openai:gpt-4o",       # LLM model (or MODEL_NAME env var)
    additional_context="...",    # Extra context for generation
    load_env=True,               # Load .env file on init
)
```

---

## TDDGenerator (True TDD Flow)

`TDDGenerator` provides a true Test-Driven Development approach where tests are generated **before** the implementation.

**Flow:** Description → Signature → Evals → Implementation

### Basic Usage

```python
from vowel.tdd import TDDGenerator

generator = TDDGenerator(
    model="gemini-3-flash-preview",
    load_env=True
)

result = generator.generate_all(
    description="Binary search for target in sorted list. Returns index or -1.",
    name="binary_search"
)

# Access all components
print(result.signature.to_signature_str())  # Function signature
print(result.yaml_spec)                      # Generated test cases
print(result.func.code)                      # Implementation
print(result.summary.all_passed)             # Test results

# Pretty print with rich formatting
result.print()
```

### TDDResult

```python
result = generator.generate_all(description="...", name="my_func")

# Properties
result.signature    # FunctionSignature: Generated signature with types
result.yaml_spec    # str: Generated eval specification (tests)
result.func         # Function: Generated implementation
result.summary      # EvalSummary: Test results

# Pretty print
result.print()      # Shows signature → tests → code → results
```

### Step-by-Step Generation

Generate each step separately for more control:

```python
from vowel.tdd import TDDGenerator

generator = TDDGenerator(model="gemini-3-flash-preview")

# Step 1: Generate signature from description
signature = generator.generate_signature(
    description="Calculate factorial of a non-negative integer",
    name="factorial"
)
print(signature.to_signature_str())
# def factorial(n: int) -> int

# Step 2: Generate eval spec from signature
runner, yaml_spec = generator.generate_evals_from_signature(signature)
print(yaml_spec)
# factorial:
#   dataset:
#     - case: { input: 0, expected: 1 }
#     - case: { input: 5, expected: 120 }
#     ...

# Step 3: Generate implementation from signature + evals
func = generator.generate_implementation(signature, yaml_spec)
print(func.code)
# def factorial(n: int) -> int:
#     if n <= 1:
#         return 1
#     return n * factorial(n - 1)

# Step 4: Run evals
summary = runner.with_functions({"factorial": func.impl}).run()
print(f"Coverage: {summary.coverage * 100:.1f}%")
```

### With Auto-Retry and Healing

```python
result = generator.generate_all(
    description="Merge two sorted lists into one sorted list",
    name="merge_sorted",
    auto_retry=True,
    max_retries=3,
    min_coverage=0.95,
    heal_function=True,
)

if result.summary.all_passed:
    print("All tests passed!")
else:
    print(f"Coverage: {result.summary.coverage * 100:.1f}%")
```

### FunctionSignature

The signature model captures full function metadata:

```python
from vowel.tdd import FunctionSignature, Param

signature = FunctionSignature(
    name="divide",
    description="Divide a by b, returning float result",
    params=[
        Param(name="a", type="float", description="Dividend"),
        Param(name="b", type="float", description="Divisor"),
    ],
    return_type="float",
    raises=["ZeroDivisionError"],
)

print(signature.to_signature_str())
# def divide(a: float, b: float) -> float
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MODEL_NAME` | Default model for EvalGenerator / TDDGenerator |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `JUDGE_MODEL` | Model for LLM Judge evaluator |

**Recommended Model:** `gemini-3-flash-preview` (fast and average %95 accurate for TDD)

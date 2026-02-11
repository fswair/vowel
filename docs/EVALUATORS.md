# Evaluators

vowel provides powerful evaluators for flexible testing. Evaluators are applied globally to all cases in a function spec via the `evals:` block.

---

## Expected Value

The simplest form — exact match against `expected:`:

```yaml
add:
  dataset:
    - case:
        inputs: { x: 2, y: 3 }
        expected: 5  # Exact match
```

---

## Type Checking

```yaml
divide:
  evals:
    Type:
      type: "float"        # Must return float
      strict: true         # Strict type validation (don't do promotion)
  dataset:
    - case:
        inputs: { a: 10, b: 3 }
```

Supported types: `int`, `float`, `str`, `bool`, `list`, `dict`, `int | float`, `str | None`, etc.

---

## Assertion

```yaml
is_positive:
  evals:
    Assertion:
      assertion: "output == (input > 0)"
  dataset:
    - case:
        input: 5
    - case:
        input: -3
```

**Available variables in assertions:**
- `input`: The input value(s)
  - Single param: `input` = value
  - Multi param (list): `input` = [val1, val2, ...]
  - Multi param (dict): `input` = {key: val, ...}
- `output`: Function return value
- `expected`: Expected value (if provided)
- `duration`: Execution time in seconds

**Assertion examples:**

```yaml
# Single parameter
assertion: "output > 0"
assertion: "output == input * 2"
assertion: "abs(output - expected) < 0.001"

# Multi-parameter (list inputs: [a, b])
assertion: "output == input[0] + input[1]"

# Multi-parameter (dict inputs: {x: 1, y: 2})
assertion: "output == input['x'] * input['y']"
```

---

## Duration (Performance)

Function-level duration constraint (seconds):

```yaml
fast_function:
  evals:
    Duration:
      duration: 0.5  # Must complete within 500ms
  dataset:
    - case:
        input: 1000
```

Case-level duration (milliseconds):

```yaml
compute:
  dataset:
    - case:
        input: 100
        duration: 50  # Max 50ms for this case
```

---

## Pattern Matching (Regex)

```yaml
format_phone:
  evals:
    Pattern:
      pattern: "^\\+?[0-9]{10,14}$"
      case_sensitive: true
  dataset:
    - case:
        input: "5551234567"
```

---

## Contains Input

```yaml
echo:
  evals:
    ContainsInput:
      case_sensitive: false
      as_strings: true
  dataset:
    - case:
        input: "Hello"
        # Output must contain "Hello" (case-insensitive)
```

---

## Raises (Exception Testing)

```yaml
divide:
  dataset:
    - case:
        inputs: { a: 10, b: 0 }
        raises: ZeroDivisionError
        match: "division by zero"  # Optional regex
```

---

## LLM Judge

```yaml
summarize:
  evals:
    LLMJudge:
      rubric: "Summary captures main points without losing critical information"
      include: [input, expected_output]
      config:
        model: "openai:gpt-4o-mini"
        temperature: 0.0
        # fields will be passed into ...pydantic_ai.settings.ModelSettings
  dataset:
    - case:
        input: "Long article text..."
```

---

## Combining Evaluators

Multiple evaluators can be applied to the same function:

```yaml
factorial:
  evals:
    Assertion:
      assertion: "output > 0"
    Type:
      type: "int"
    Duration:
      duration: 1.0
  dataset:
    - case: { input: 0, expected: 1 }
    - case: { input: 5, expected: 120 }
    - case: { input: 10, expected: 3628800 }
```

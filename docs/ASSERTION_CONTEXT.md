# Assertion Context Variables

When writing custom assertions in vowel, you have access to several variables in the evaluation context.

## Input vs Inputs in YAML

Before diving into assertion variables, it's important to understand the distinction between `input` and `inputs` fields in YAML test cases:

- **`input`**: Use for **single-parameter** functions. The value is passed directly to the function.

  ```yaml
  len:
    dataset:
      - case:
          input: [1, 2, 3] # len([1, 2, 3])
  ```

- **`inputs`**: Use for **multi-parameter** functions. The list elements are unpacked as separate arguments.
  ```yaml
  max:
    dataset:
      - case:
          inputs: [5, 10, 3] # max(5, 10, 3)
  ```

**Note:** Even if your single-parameter function takes a list/array, use `input` (not `inputs`):

```yaml
# Correct - function expects a single list parameter
process_list:
  dataset:
    - case:
        input: [1, 2, 3]  # process_list([1, 2, 3])

# Wrong - would try to call process_list(1, 2, 3)
process_list:
  dataset:
    - case:
        inputs: [1, 2, 3]  # ERROR!
```

## Available Variables

### `input`

**Type:** Any  
**Description:** The input value(s) passed to the function being tested.

**Important:** Regardless of whether the test case uses `input` (single parameter) or `inputs` (multiple parameters),
this variable always contains the complete input data. For single-parameter functions, it's the direct value.
For multi-parameter functions, it's a dict like `{'inputs': [arg1, arg2, ...]}`.

**Examples:**

```python
# For single parameter functions (using 'input' field in YAML)
"output == input * 2"  # input is 5, expects output 10

# For dict inputs (single parameter that is a dict)
"output == input['x'] + input['y']"  # input is {"x": 2, "y": 3}

# For list inputs (single parameter that is a list)
"len(output) <= len(input)"  # input is [1, 2, 3], check output size vs input size

# For multi-parameter functions (using 'inputs' field in YAML)
# The input variable contains the inputs dict
"len(input['inputs']) == 3"  # Check number of parameters
"input['inputs'][0] < input['inputs'][1]"  # Compare parameters
```

### `output`

**Type:** Any  
**Description:** The actual output returned by the function.

**Examples:**

```python
"output > 0"  # Check output is positive
"isinstance(output, int)"  # Check output type
"output.isupper()"  # Check string property
"all(x > 0 for x in output)"  # Check all items in list
```

### `expected`

**Type:** Any (optional)  
**Description:** The expected output value from the test case (if provided via `expected` field).

**Examples:**

```python
"output == expected"  # Exact match
"abs(output - expected) < 0.001"  # Floating point comparison
"set(output) == set(expected)"  # Order-independent comparison
```

### `duration`

**Type:** float  
**Description:** Actual execution time in seconds.

**Examples:**

```python
"duration < 0.1"  # Must complete in under 100ms
"duration < 1.0"  # Must complete in under 1 second
```

### `metadata`

**Type:** dict (optional)  
**Description:** Additional metadata dictionary if provided with the test case.

**Examples:**

```python
"metadata.get('version') == '1.0'"  # Check metadata field
"metadata.get('debug', False)"  # Access with default
```

## Common Assertion Patterns

### Numeric Comparisons

```python
"output > 0"                           # Positive check
"output >= 0"                          # Non-negative check
"output == input ** 2"                 # Square calculation
"abs(output - expected) < 1e-6" # Float comparison with tolerance
"0 <= output <= 100"                   # Range check
```

### String Operations

```python
"output.isupper()"                     # All uppercase
"output.islower()"                     # All lowercase
"len(output) > 0"                      # Not empty
"input in output"                      # Contains input
"output.startswith('prefix')"          # Prefix check
"output.endswith('suffix')"            # Suffix check
```

### Collection Operations

```python
"len(output) == len(input)"            # Same length
"len(output) <= len(input)"            # Output not longer
"all(x > 0 for x in output)"           # All positive
"any(x < 0 for x in output)"           # Has negative
"output in input"                      # Output is subset
"set(output) == set(expected)"  # Same elements, any order
```

### Dict Operations

```python
"output['key'] == expected_value"      # Dict access
"'key' in output"                      # Key exists
"output.get('key', default) == value"  # Safe access
"input['x'] + input['y'] == output"    # Dict input calculation
```

### Type Checks (in assertions)

```python
"isinstance(output, int)"              # Integer check
"isinstance(output, (int, float))"     # Numeric check
"type(output).__name__ == 'list'"      # Type name check
```

### Boolean Logic

```python
"(output and input > 0) or (not output and input <= 0)"  # Conditional logic
"output == (input % 2 == 0)"                             # Boolean match
```

### Performance Checks

```python
"duration < 0.1"                       # Fast execution
"duration < 1.0"                       # Medium execution
"duration < 0.001"                     # Very fast execution
```

## Best Practices

1. **Keep assertions simple**: One assertion should test one property
2. **Use descriptive names**: `IsPositive`, `CorrectSquare`, `ValidRange`
3. **Prefer type checks separately**: Use `IsInstanceCase` for types, assertions for logic
4. **Handle edge cases**: Test with None, empty collections, zero, negative numbers
5. **Use expected when available**: Instead of hardcoding expected values
6. **Document complex assertions**: Use clear evaluation rule names

## Note about generated specs

When specs are generated by an LLM (via `generate_eval_spec` / `generate_multiple_eval_specs`),
the tooling will automatically sanitize assertion expressions to ensure they use the
canonical variable names: `input`, `output`, `expected`, `duration`, and `metadata`.
Common aliases (for example `result`, `res`, `response`, `expected`, `exp`, `time`, `meta`) are
mapped to the canonical names during post-processing. This helps keep assertions stable
and compatible with the runtime evaluation environment.

## Examples from Real Functions

### Prime Number Check

```python
{
    "IsBoolean": {"type": "bool"},
    "CorrectForPrimes": {
        "assertion": "(output and input in [2, 3, 5, 7, 11, 13]) or (not output and input not in [2, 3, 5, 7, 11, 13])"
    }
}
```

### Rectangle Area Calculator

```python
{
    "IsFloat": {"type": "float"},
    "CorrectProduct": {
        "assertion": "output == input['width'] * input['height']"
    },
    "NonNegative": {
        "assertion": "output >= 0"
    }
}
```

### List Filter Function

```python
{
    "IsList": {"type": "list"},
    "AllEven": {
        "assertion": "all(x % 2 == 0 for x in output)"
    },
    "SubsetOfInput": {
        "assertion": "all(x in input for x in output)"
    },
    "NotLongerThanInput": {
        "assertion": "len(output) <= len(input)"
    }
}
```

### String Transformation

```python
{
    "IsString": {"type": "str"},
    "AllUppercase": {
        "assertion": "output.isupper()"
    },
    "PreservesLength": {
        "assertion": "len(output) == len(input)"
    },
    "MatchesExpected": {
        "assertion": "output == expected"
    }
}
```

## Anti-Patterns (Avoid These)

❌ **Accessing undefined variables**

```python
"output > n"  # 'n' is not in context
```

✅ **Use available variables**

```python
"output > input"
```

---

❌ **Complex multi-condition assertions**

```python
"output > 0 and output < 100 and output % 2 == 0 and isinstance(output, int)"
```

✅ **Split into multiple named assertions**

```python
{
    "IsInteger": {"type": "int"},
    "IsPositive": {"assertion": "output > 0"},
    "InValidRange": {"assertion": "0 < output < 100"},
    "IsEven": {"assertion": "output % 2 == 0"}
}
```

---

❌ **Hardcoding expected values**

```python
"output == 25"  # Brittle, only works for one input
```

✅ **Use input or expected**

```python
"output == input ** 2"
# or
"output == expected"
```

---

❌ **Modifying variables in assertions**

```python
"input.append(5) and True"  # Side effects are bad!
```

✅ **Read-only access**

```python
"5 in input"
```

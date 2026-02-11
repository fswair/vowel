"""
Context definitions for vowel eval specification generation.

This module contains the EVAL_SPEC_CONTEXT which provides comprehensive
documentation about vowel's YAML-based evaluation specification format.
This context is used by EvalGenerator to guide LLM-based eval generation.

Set VOWEL_CONTEXT_VERSION=legacy to use the pre-optimization prompt.
Default is "v3" (GEPA-optimized with Sonnet proposer).
"""

import os

_EVAL_SPEC_CONTEXT_LEGACY = r"""
# VOWEL Evaluation Specification Guide

vowel is a YAML-based evaluation framework for testing Python functions.
When generating eval specifications, follow these guidelines:

---
## 🎯 QUALITY REQUIREMENTS - GENERATE HIGH-QUALITY EVALS

**Your goal is to generate COMPREHENSIVE, HIGH-QUALITY evaluation specifications.**

### Quality Checklist:
1. **Coverage**: Test ALL code paths including:
   - Happy path (normal inputs)
   - Edge cases (empty, null, boundary values)
   - Error cases (invalid inputs, exceptions)

2. **Diverse Test Cases**: Include at least 5-8 test cases covering:
   - Typical use cases
   - Boundary conditions
   - Invalid/malformed inputs
   - Large/small inputs

3. **Accurate Expected Values**:
   - Manually verify expected outputs match actual function behavior
   - Use the function's actual logic to determine correct outputs
   - DO NOT guess expected values - trace through the code
   - For algorithmic functions (distance, sorting, math), TRACE the algorithm step by step to compute the exact result
   - For format-sensitive outputs (JSON encoding, serialization), check exact formatting details:
     - Does the function use `separators=(',', ':')` (compact) vs default `(', ', ': ')` (spaced)?
     - Does the function sort keys or preserve insertion order?
     - Does the function add trailing newlines or spaces?
   - For functions returning ordered lists, verify the exact ordering logic (not just the elements)
   - See the **EXPECTED vs ASSERTION DECISION GUIDE** below for when to use `expected` vs `assertion`

4. **Appropriate Evaluators**: See the **EXPECTED vs ASSERTION DECISION GUIDE** section below.
   Key principle: use `expected` for trivially verifiable values, `assertion` with property-based checks for everything else.
   - Use `raises` for exception testing (only when the code actually raises)
   - Use `type` for output type validation
   - Use `LLMJudge` ONLY for subjective/semantic evaluation (not for deterministic checks)

5. **Meaningful Case IDs**: Use descriptive names like:
   - `empty_list`, `single_element`, `large_input`
   - `invalid_type_error`, `boundary_value`
   - NOT generic names like `test1`, `case_a`

---
## 🧭 EXPECTED vs ASSERTION DECISION GUIDE

**RULE: Use `expected` when the value is TRIVIALLY VERIFIABLE. Use `assertion` when you would need to COMPUTE or TRACE an algorithm to get the exact value.**

This is NOT about subjective confidence ("am I sure?"). It is about the CATEGORY of computation:

### USE `expected` WHEN:
- **Simple arithmetic**: `add(2, 3)` → `expected: 5`
- **Boolean logic**: `is_even(4)` → `expected: true`
- **Direct string ops**: `greet("ali")` → `expected: "hello ali"`
- **Identity / passthrough**: `identity(42)` → `expected: 42`
- **Lookup / constants**: `get_pi()` → `expected: 3.14159`
- **List literals**: `reverse([1,2,3])` → `expected: [3, 2, 1]`
- **General rule**: The answer is obvious WITHOUT running the function — you can verify it in your head in under 2 seconds.

### USE `assertion` (property-based) WHEN:
- **Algorithmic computation** — edit distance, DP, graph algorithms, hashing, scoring
- **Format-sensitive output** — JSON encoding, date formatting, number formatting, slug generation
- **Floating point / rounding** — decimal quantization, financial calculations, trigonometry
- **Collection ordering** — ranked results, scored matches, sorted-by-key results
- **Multi-step string transformation** — transliteration, normalization, encoding chains
- **General rule**: You would need to run the algorithm or trace multiple steps to verify the exact result.

### HOW TO WRITE GOOD ASSERTIONS (not just `isinstance`):

Weak assertions like `isinstance(output, str)` are almost worthless — they pass for ANY string.
Good assertions check **invariants and properties** that a correct implementation MUST satisfy:

**Algorithmic functions (edit distance, search, sort):**
```yaml
# Levenshtein — check mathematical properties the result MUST have
assertion: "output >= 0"                                                  # non-negative
assertion: "output <= max(len(input[0]), len(input[1]))"                   # upper bound
assertion: "(input[0] == input[1]) == (output == 0)"                      # identity property
assertion: "output >= abs(len(input[0]) - len(input[1]))"                 # triangle inequality lower bound

# Binary search — if found, element must match
assertion: "output == -1 or input[0][output] == input[1]"
```

**Format-sensitive output (JSON, slug, date):**
```yaml
# JSON encoding — check roundtrip validity, not exact spacing
assertion: "len(output) > 2"                                              # non-empty JSON output
assertion: "output.startswith('{') and output.endswith('}')"               # dict shape
assertion: "output.startswith('{') or output.startswith('[')"           # valid JSON shape

# Slugify — check properties every slug must have
assertion: "output == output.lower()"                                     # always lowercase
assertion: "' ' not in output"                                            # no spaces
assertion: "all(c.isalnum() or c == '-' for c in output)"                 # only alnum + hyphen
assertion: "not output.startswith('-') and not output.endswith('-')"       # no leading/trailing hyphens
```

**Floating point / rounding:**
```yaml
assertion: "abs(output - 3.14) < 0.01"                                    # tolerance check
assertion: "'.' in str(output) and len(str(output).split('.')[1]) <= 2"   # max 2 decimal places
```

**Collection ordering (ranked results, scored matches):**
```yaml
assertion: "set(output) == {'apple', 'ape', 'pupple'}"                    # correct elements (order-independent)
assertion: "len(output) == 3"                                             # correct count
assertion: "output[0] == 'apple'"                                         # best match is first
assertion: "all(isinstance(x, str) for x in output)"                      # all elements are strings
```

**Multi-step string transformation:**
```yaml
assertion: "len(output) > 0"                                              # non-empty
assertion: "output.isascii()"                                              # ASCII-only after transliteration
assertion: "output == output.strip()"                                      # no leading/trailing whitespace
```

### COMBINING `expected` AND `assertion`:

You CAN use both in the same spec. Use `expected` for the trivial cases and `assertion` for the complex ones:

```yaml
levenshtein:
  evals:
    NonNegative:
      assertion: "output >= 0"
    UpperBound:
      assertion: "output <= max(len(input[0]), len(input[1]))"
  dataset:
    # Trivial cases — expected is reliable here
    - case:
        id: identical_strings
        inputs: ["abc", "abc"]
        expected: 0
    - case:
        id: empty_vs_word
        inputs: ["", "hello"]
        expected: 5
    # Complex cases — use assertion, don't guess the DP result
    - case:
        id: similar_words
        inputs: ["kitten", "sitting"]
        assertion: "3 <= output <= 5"     # bounds we're sure about
    - case:
        id: long_strings
        inputs: ["abcdefgh", "azbycxdw"]
        assertion: "output >= 4 and output <= 8"
```

---
## ⚠️ CRITICAL YAML SYNTAX RULES - MUST FOLLOW

**DO NOT break YAML syntax to create failing tests!** If you want to test error cases
or create intentionally failing tests, use the proper evaluators (assertion, expected, raises).
Breaking the YAML syntax will crash the program, not create test failures.

### FORBIDDEN PATTERNS (Will crash the parser):

0. **NO `__import__` calls in assertions:**
   ```yaml
   # ❌ WRONG - __import__ is blocked at runtime, will raise an error
   assertion: "__import__('json').loads(output) is not None"
   assertion: "isinstance(__import__('decimal').Decimal(output), object)"

   # ✅ CORRECT - use property checks without imports
   assertion: "len(output) > 2"
   assertion: "output.startswith('{') or output.startswith('[')"
   ```

1. **NO YAML tags (!!python/name, !!python/object, !!binary, etc.):**
   ```yaml
   # ❌ WRONG - YAML tags crash safe_load
   input: !!python/name:builtins.set
   input: !!python/object:collections.OrderedDict
   input: !!binary aGVsbG8=

   # ✅ CORRECT - use plain Python literals or string representations
   input: "set()"
   inputs: [{"key": "value"}]
   ```

2. **NO Python expressions with `*` operator in YAML values:**
   ```yaml
   # ❌ WRONG - * is interpreted as YAML alias, causes parser error
   input: "a" * 5
   input: 10 * 2

   # ✅ CORRECT - use the computed literal value
   input: "aaaaa"
   input: 20
   ```

3. **NO Python comprehensions in YAML values:**
   ```yaml
   # ❌ WRONG - causes parser error
   inputs: [{'id': x} for x in range(10)]

   # ✅ CORRECT - use literal values
   inputs:
     - {id: 1}
     - {id: 2}
   ```

4. **NO string values where list/dict expected:**
   ```yaml
   # ❌ WRONG - inputs must be list or dict
   inputs: "not_a_list"

   # ✅ CORRECT - use proper type
   inputs: []
   inputs: [1, 2, 3]
   inputs:
     param1: value1
   ```

5. **NO inline Python syntax in flow sequences:**
   ```yaml
   # ❌ WRONG - Python ternary in YAML
   inputs: [{'x': 1 if True else 0}]

   # ✅ CORRECT - literal values only
   inputs: [{x: 1}]
   ```

6. **Escape regex patterns properly:**
   ```yaml
   # ❌ WRONG - unescaped backslash
   pattern: "^\\d+$"

   # ✅ CORRECT - use single quotes for regex
   pattern: '^\\d+$'
   ```

7. **Testing with `null` input is valid:**
   ```yaml
   # ✅ CORRECT - tests func(None)
   - case:
       id: null_input_test
       input: null
       raises: TypeError

   # ✅ CORRECT - tests func(None, None)
   - case:
       id: null_inputs_test
       inputs: [null, null]
       raises: TypeError
   ```

8. **Use `inputs:` (plural) not `input:` for named parameters:**
   ```yaml
   # For function: def func(items, key)
   # ❌ WRONG
   input: [{...}], 'type'

   # ✅ CORRECT
   inputs:
     items: [{...}]
     key: type
   ```

9. **NO Python types that YAML cannot represent as literals:**

   YAML can only represent: `int`, `float`, `str`, `bool`, `list`, `dict`, `null`.
   The following Python types CANNOT be expressed in YAML and MUST NOT appear in input/expected/inputs:

   ```yaml
   # ❌ WRONG - set literals become dicts in YAML
   input: {1, 2, 3}          # Parsed as {1: null, 2: null, 3: null} (dict!)

   # ❌ WRONG - tuple literals become strings or broken values
   input: (1, 2, 3)          # Parsed as string "(1, 2, 3)" or broken
   input: [1, 2, (3, 4)]     # The (3, 4) becomes strings "(3" and "4)"

   # ❌ WRONG - Python function calls are NOT evaluated in YAML
   input: float('inf')       # Parsed as string "float('inf')"
   input: float('nan')       # Parsed as string "float('nan')"
   input: float('-inf')      # Parsed as string "float('-inf')"

   # ❌ WRONG - bytes/binary data cannot be expressed in YAML
   input: b"hello"           # Syntax error or string
   inputs: ["aGVsbG8=", "utf-8", false, "strict"]  # base64 string is NOT bytes!

   # ❌ WRONG - complex numbers not supported
   input: 1+2j               # Parsed as string

   # ✅ CORRECT - only use YAML-native types
   input: [1, 2, 3]          # list (instead of set or tuple)
   input: "hello"            # string
   input: 42                 # int
   input: 3.14               # float
   input: null               # None
   input: {"key": "value"}   # dict
   ```

   **If a function REQUIRES set/tuple/bytes/inf/nan as input, you CANNOT test that case in YAML.**
   Skip those cases entirely rather than using string representations that will be passed as strings.

### TO CREATE FAILING TESTS PROPERLY:
- Use WRONG `expected` values → test will fail on value mismatch
- Use FALSE `assertion` expressions → test will fail assertion check
- Use `raises` to expect exceptions that won't occur
- **NEVER corrupt YAML syntax** - the spec must be parseable

---

## YAML Structure

```yaml
function_name:
  evals:                    # Global evaluators (optional) - apply to ALL test cases
    EvaluatorName:
      # evaluator specific parameters
  dataset:                  # Test cases (required)
    - case:
        input: <value>      # For single-parameter functions
        # OR
        inputs: [<args>]    # For multi-parameter functions (unpacked as *args, or **kwargs for dict)
        expected: <value>   # Expected output (optional)
        assertion: <expr>   # Custom assertion expression (optional)
```

## Input vs Inputs

- **`input`**: Single parameter - value passed directly to function
  ```yaml
  len:
    dataset:
      - case:
          input: [1, 2, 3]  # len([1, 2, 3]) → 3
  ```

- **`inputs`**: Multiple parameters - list unpacked as separate arguments
  ```yaml
  max:
    dataset:
      - case:
          inputs: [5, 10, 3]  # max(5, 10, 3) → 10
  ```

- **`input: null`**: Pass None as argument (for testing None handling)
  ```yaml
  handle_none:
    dataset:
      - case:
          id: none_input
          input: null           # handle_none(None)
          raises: TypeError     # Expect TypeError when None is passed
  ```

**IMPORTANT**: Even if function takes a single list parameter, use `input` not `inputs`:
```yaml
# Correct - function expects a single list parameter
process_list:
  dataset:
    - case:
        input: [1, 2, 3]  # process_list([1, 2, 3])

# WRONG - would call process_list(1, 2, 3)
process_list:
  dataset:
    - case:
        inputs: [1, 2, 3]  # ERROR!
```

---

# BUILTIN EVALUATORS

## 1. Type Evaluator (Global & Case Level)

Checks if output matches the specified Python type. Supports union types.
Can be used globally (applies to all cases) or per-case (applies to specific case).

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | ✅ Yes | `str` | Python type as string. Supports union with `\|` |
| `strict` | ❌ No | `bool` | Enable strict type validation (default: false) |
| `strict_type` | ❌ No | `bool` | Case-level: exact type match required (default: false) |

### Global Level Examples:
```yaml
function_name:
  evals:
    IsString:
      type: str

    IsNumber:
      type: "int | float"    # Union type

    IsList:
      type: list

    IsDict:
      type: dict

    IsOptionalInt:
      type: "int | None"
```

### Case Level Examples:
```yaml
function_name:
  dataset:
    - case:
        input: 5
        expected: 10
        type: int           # This case expects int output

    - case:
        input: "hello"
        expected: "HELLO"
        type: str           # This case expects str output

    - case:
        input: [1, 2, 3]
        type: "list[int]"   # Complex type annotation
        strict_type: true   # Exact type match required
```

### Supported Types:
- Basic: `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`
- None: `None`
- Union: `"int | float"`, `"str | None"`, `"list | dict"`
- Complex: `"list[int]"`, `"dict[str, Any]"`, `"Optional[int]"`

---

## 2. Assertion Evaluator (Global & Case Level)

Executes a Python expression. Must evaluate to `True` for test to pass.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `assertion` | ✅ Yes | `str` | Python expression returning boolean |

### Available Variables in Assertion Context:
- `input` - The input value(s) passed to the function
- `output` - The actual return value from the function
- `expected` - The expected value (if provided in test case)
- `duration` - Execution time in seconds (float)
- `metadata` - Additional metadata dict (if provided)

### Global Level Example:
```yaml
function_name:
  evals:
    IsPositive:
      assertion: "output > 0"

    RangeCheck:
      assertion: "10 < output < 100"

    ListLength:
      assertion: "len(output) == 5"

    StartsWith:
      assertion: "output.startswith('hello')"

    IsList:
      assertion: "isinstance(output, list)"

    CorrectSquare:
      assertion: "output == input ** 2"
```

### Case Level Example:
```yaml
function_name:
  dataset:
    - case:
        input: 5
        expected: 25
        assertion: "output == input ** 2"   # Case-specific assertion
```

### Common Assertion Patterns:

**Numeric:**
```python
"output > 0"                        # Positive check
"output >= 0"                       # Non-negative check
"abs(output - expected) < 1e-6"     # Float tolerance comparison
"0 <= output <= 100"                # Range check
"output == input ** 2"              # Square calculation
"output % 2 == 0"                   # Even check
```

**String:**
```python
"output.isupper()"                  # All uppercase
"output.islower()"                  # All lowercase
"len(output) > 0"                   # Not empty
"input in output"                   # Contains input
"output.startswith('prefix')"       # Prefix check
"output.endswith('suffix')"         # Suffix check
```

**Collection:**
```python
"len(output) == len(input)"         # Same length
"len(output) <= len(input)"         # Not longer than input
"all(x > 0 for x in output)"        # All positive
"any(x < 0 for x in output)"        # Has negative
"set(output) == set(expected)"      # Order-independent equality
"sorted(output) == output"          # Is sorted
"output in input"                   # Output is subset
```

**Dict:**
```python
"'key' in output"                   # Key exists
"output['key'] == expected_value"   # Dict access
"output.get('key', default) == val" # Safe access with default
"input['x'] + input['y'] == output" # Dict input calculation
```

**Type Checks (in assertions):**
```python
"isinstance(output, int)"           # Integer check
"isinstance(output, (int, float))"  # Numeric check
"type(output).__name__ == 'list'"   # Type name check
```

**Performance:**
```python
"duration < 0.1"                    # Under 100ms
"duration < 1.0"                    # Under 1 second
"duration < 0.001"                  # Under 1ms
```

---

## 3. Duration Evaluator (Global & Case Level)

Validates that function execution time is within allowed limit.

### Global Level Fields:
| Field | Required | Type | Unit | Description |
|-------|----------|------|------|-------------|
| `duration` | ✅ Yes | `float` | **seconds** | Maximum allowed execution time |

### Case Level Fields:
| Field | Required | Type | Unit | Description |
|-------|----------|------|------|-------------|
| `duration` | ✅ Yes | `float` | **milliseconds** | Maximum allowed execution time |

**⚠️ IMPORTANT**: Global level uses seconds, case level uses milliseconds!

### Examples:
```yaml
# Global level - SECONDS
fibonacci:
  evals:
    FastEnough:
      duration: 0.01    # Maximum 10ms (0.01 seconds)

  dataset:
    - case:
        input: 10
        expected: 55

    - case:
        input: 20
        expected: 6765
        duration: 100   # Case level - 100 MILLISECONDS
```

---

## 4. Contains Input Evaluator (Global Level)

Checks if the output contains the input value.

### Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `contains_input` | ❌ No | `bool` | `true` | Whether output should contain input |
| `case_sensitive` | ❌ No | `bool` | `true` | Case sensitivity for string comparison |
| `as_strings` | ❌ No | `bool` | `false` | Convert both to strings before comparison |

### Examples:
```yaml
wrap_string:
  evals:
    ContainsInput:
      contains_input:
        case_sensitive: true
  dataset:
    - case:
        input: "world"
        expected: "Hello, world!"

repeat_list:
  evals:
    ContainsInput:
      contains_input:
        as_strings: true    # [1,2] → "[1, 2]" as string comparison
  dataset:
    - case:
        input: [1, 2, 3]
        expected: [1, 2, 3, 1, 2, 3]
```

---

## 5. Pattern Matching Evaluator (Global & Case Level) - Regex

Validates that output matches a regular expression pattern.

### Global Level Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | ✅ Yes | `str` | - | Regex pattern to match against output (converted to string) |
| `case_sensitive` | ❌ No | `bool` | `true` | Whether regex matching is case-sensitive |

### Case Level Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | ✅ Yes | `str` | - | Regex pattern for this specific case |
| `case_sensitive` | ❌ No | `bool` | `true` | Whether regex matching is case-sensitive |

### Examples:
```yaml
# Global level
validate_email:
  evals:
    HasAtSign:
      pattern: "@"
    ValidDomain:
      pattern: "\\.(com|org|net)$"
  dataset:
    - case:
        input: "test@example.com"
        expected: "test@example.com"

# Case level
format_id:
  evals:
    CorrectFormat:
      pattern: "^id: \\d+$"
  dataset:
    - case:
        input: 123
        expected: "id: 123"

    - case:
        input: 456
        expected: "ID: 456"
        pattern: "^ID: \\d+$"        # Override for this case
        case_sensitive: true

# Case insensitive
normalize_text:
  dataset:
    - case:
        input: "Hello"
        expected: "HELLO WORLD"
        pattern: "hello"
        case_sensitive: false        # Matches "HELLO" too
```

### Common Regex Patterns:
```yaml
pattern: "^\\d+$"                    # Numbers only
pattern: "^[A-Z]+$"                  # Uppercase letters only
pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"  # Email
pattern: "^https?://.*"              # URL
pattern: "\\bword\\b"                # Contains specific word
pattern: "^prefix"                   # Starts with prefix
pattern: "suffix$"                   # Ends with suffix
```

---

## 6. Expected Evaluator (Case Level Only)

Compares function output with expected value for equality.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `expected` | ✅ Yes | `Any` | Expected output value (any Python literal) |

### Examples:
```yaml
add:
  dataset:
    - case:
        inputs: [2, 3]
        expected: 5

    - case:
        inputs: [10, -5]
        expected: 5

process_data:
  dataset:
    - case:
        input: {"x": 1, "y": 2}
        expected: {"result": 3, "sum": 3}
```

---

## 7. Contains Evaluator (Case Level Only)

Checks if a specific value is contained in the output.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `contains` | ✅ Yes | `Any` | Value that should be contained in output |

### Examples:
```yaml
generate_html:
  dataset:
    - case:
        input: "Title"
        contains: "<h1>Title</h1>"    # This string must be in output

    - case:
        input: "Link"
        contains: "<a>"               # Must contain anchor tag
```

---

## 8. Raises Evaluator (Case Level Only) - Exception Testing

Tests that a function raises a specific exception. Similar to pytest's `pytest.raises`.
Supports **optional raises** with `?` suffix for uncertain error cases.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `raises` | ✅ Yes | `str` | Exception type name. Append `?` for optional (e.g., `TypeError?`) |
| `match` | ❌ No | `str` | Regex pattern to match against exception message |

### Strict vs Optional Raises:
- `raises: ValueError` → Function **MUST** raise ValueError. Fails if it returns normally.
- `raises: ValueError?` → Function **MAY** raise ValueError. Passes if it raises ValueError OR returns normally. Only fails if a DIFFERENT exception is raised.

**⚠️ IMPORTANT Notes:**
- `raises` is **case-level only** - cannot be used as global evaluator
- `match` can **only** be used together with `raises`
- Use `raises: ExType?` (with `?`) when you're NOT SURE if the function validates/raises
- Use `raises: ExType` (without `?`) ONLY for CERTAIN error cases (e.g., division by zero)
- Global evaluators are automatically skipped for exception cases

### Examples:
```yaml
calculate_discount:
  evals:
    IsFloat:
      type: float
  dataset:
    - case:
        id: "valid_calculation"
        inputs: [100.0, 20.0]
        expected: 80.0

    - case:
        id: "negative_price"
        inputs: [-100.0, 20.0]
        raises: ValueError
        match: "must be positive"     # Checks exception message

    - case:
        id: "invalid_discount"
        inputs: [100.0, 150.0]
        raises: ValueError            # Just checks type, not message

divide:
  dataset:
    - case:
        inputs: [10, 2]
        expected: 5.0

    - case:
        inputs: [10, 0]
        raises: ZeroDivisionError     # CERTAIN: division by zero always raises

process_input:
  dataset:
    - case:
        input: "valid"
        expected: "VALID"

    - case:
        input: null
        raises: TypeError?            # UNCERTAIN: might raise or might handle gracefully

    - case:
        input: 123
        raises: TypeError?            # UNCERTAIN: might coerce int to str instead
```
```

### Common Exception Types:
- `ValueError` - Invalid value/argument
- `TypeError` - Wrong type
- `KeyError` - Missing dictionary key
- `IndexError` - List/array index out of range
- `ZeroDivisionError` - Division by zero
- `AttributeError` - Missing attribute
- `FileNotFoundError` - File doesn't exist
- `RuntimeError` - Generic runtime error

---

## 9. LLM Judge Evaluator (Global Level)

Uses a Language Model to evaluate outputs based on a custom rubric. Ideal for semantic evaluation.
The model is automatically configured by the system - you only need to provide the rubric.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `rubric` | ✅ Yes | `str` | Evaluation criteria/question for the LLM |
| `include` | ❌ No | `list[str]` | Context to provide: `["input"]`, `["expected_output"]`, or both |
| `config` | ❌ No | `dict` | Optional configuration for LLM behavior |

### Config Fields (all optional):
| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `float` | Randomness (0.0 = deterministic, recommended for evals) |
| `max_tokens` | `int` | Maximum response tokens |
| `top_p` | `float` | Nucleus sampling threshold (use temperature OR top_p, not both) |
| `timeout` | `float` | Request timeout in seconds |
| `seed` | `int` | Random seed for reproducibility |
| `presence_penalty` | `float` | Penalize tokens that appeared in text |
| `frequency_penalty` | `float` | Penalize tokens by frequency |

### Include Options:
- `input` - Include function input in LLM context
- `expected_output` - Include expected output in LLM context
- **Note**: Output is always included automatically

### Examples:
```yaml
# Semantic equivalence check
translate_to_english:
  evals:
    SemanticMatch:
      rubric: "Does the output have the same meaning as the expected output?"
      include:
        - expected_output
  dataset:
    - case:
        input: "Bonjour"
        expected: "Hello"

# Quality assessment with input context
summarize_text:
  evals:
    IsGoodSummary:
      rubric: "Is the output a good summary of the input text? Does it capture the main points?"
      include:
        - input
      config:
        temperature: 0.0
  dataset:
    - case:
        input: "Long article text here..."
        expected: "Brief summary..."

# With both input and expected_output context
check_correctness:
  evals:
    AnswerQuality:
      rubric: "Does the output correctly answer the question based on the input?"
      include:
        - input
        - expected_output
  dataset:
    - case:
        input: "What is 2+2?"
        expected: "4"
```

---

# CASE LEVEL FIELDS SUMMARY

All fields available within a `case:` block:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Optional unique identifier for the case |
| `input` | `Any` | Single input value (use for single-param functions) |
| `inputs` | `list` | Multiple inputs (use for multi-param functions) |
| `expected` | `Any` | Expected output value |
| `duration` | `float` | Max execution time in **milliseconds** |
| `contains` | `Any` | Value that should be in output |
| `assertion` | `str` | Python expression that must be True |
| `pattern` | `str` | Regex pattern to match output |
| `case_sensitive` | `bool` | Case sensitivity for pattern (default: true) |
| `raises` | `str` | Expected exception type. Use `?` suffix for optional (e.g., `TypeError?`) |
| `match` | `str` | Regex for exception message (only with raises) |
| `type` | `str` | Expected output type for this case (e.g., `int`, `list[str]`) |
| `strict_type` | `bool` | If true, exact type match required (default: false) |

**⚠️ ONLY the fields listed above are allowed in a case block.**
**DO NOT add any other fields** like `comment`, `description`, `note`, `reason`, `explanation`, or any custom fields.
Extra fields will cause a Pydantic validation error and crash the parser.

---

# TEST CASE CATEGORIES

Always include diverse test cases:

1. **Typical cases** - Normal expected usage with valid inputs
2. **Edge cases** - Empty inputs, single elements, boundary values, zero, negative numbers
3. **Error cases** - Invalid inputs that should raise exceptions (use `raises:`)

---

# ⚠️ COMMON EVAL GENERATION MISTAKES TO AVOID

### MISTAKE 1: Inventing Exception Cases Not in the Code

**Only use strict `raises:` for exceptions you are CERTAIN about.**
For uncertain error cases, use `raises: ExType?` (with `?` suffix) which passes
whether the function raises OR returns normally.

```yaml
# ✅ CERTAIN — division by zero always raises
- case:
    inputs: [10, 0]
    raises: ZeroDivisionError

# ✅ CERTAIN — explicit raise in the code
- case:
    inputs: [-100.0, 20.0]
    raises: ValueError
    match: "must be positive"

# ✅ UNCERTAIN — use ? suffix: might raise TypeError or might coerce
- case:
    input: null
    raises: TypeError?

# ✅ UNCERTAIN — use ? suffix: function might handle this gracefully
- case:
    input: 123
    raises: TypeError?

# ❌ WRONG — strict raises for uncertain case (will fail if function handles it)
- case:
    input: null
    raises: TypeError
```

**Rules for raises vs raises?:**
- Use `raises: ExType` (strict) when you can SEE `raise ExType` in the code, or when the operation ALWAYS raises (e.g., `x / 0`)
- Use `raises: ExType?` (optional) when you THINK it might raise but the function might handle it gracefully instead
- Do NOT invent OverflowError for large integers (Python has arbitrary precision)
- When a function uses `try/except` to handle errors gracefully, it does NOT re-raise them

### MISTAKE 2: Global Assertions That Fail on Passthrough/Polymorphic Returns

Many functions return the input unchanged for certain cases (passthrough).
Global assertions must handle ALL return paths, not just the happy path.

```yaml
# ❌ WRONG - This assertion fails when input="abc" is passed through as "abc"
# because "abc" doesn't end with st/nd/rd/th
ordinal:
  evals:
    CorrectSuffix:
      assertion: "any(output.endswith(s) for s in ['st', 'nd', 'rd', 'th'])"

# ❌ WRONG - Return type misses `list` (function returns [1,2,3] unchanged for list input)
ordinal:
  evals:
    ReturnType:
      type: "str | int | float | None"    # Missing list!

# ✅ CORRECT - Guard assertion for cases where it applies
ordinal:
  evals:
    CorrectSuffix:
      assertion: "not isinstance(output, str) or not output[-1:].isdigit() or any(output.endswith(s) for s in ['st', 'nd', 'rd', 'th'])"
```

**Rules:**
- Check ALL code paths: what does the function return for invalid/unexpected inputs?
- If a function has `return value` for unrecognized types, include that type in the global `type:` evaluator
- If a global assertion only applies to some outputs, add a guard condition:
  `"not isinstance(output, str) or <your_check>"`  (only check strings)
- Better: use case-level assertions instead of fragile global ones

### MISTAKE 3: Wrong Expected Values for Algorithmic Functions

LLMs frequently miscalculate algorithmic results. TRACE the algorithm manually.

```yaml
# ❌ WRONG - Levenshtein("clutch", "cult") is actually 4, not 3
# clutch → _lutch (delete c=1) → _ultch (swap l/u? NO - must trace DP matrix)
- case:
    inputs: ["clutch", "cult"]
    expected: 3

# ✅ CORRECT - Actually trace the dynamic programming matrix
# c l u t c h  vs  c u l t
# DP result = 4
- case:
    inputs: ["clutch", "cult"]
    expected: 4
```

For complex algorithms, **prefer assertion-based checks** over hardcoded expected values:
```yaml
# ✅ BETTER - Property-based checks that are always correct
levenshtein:
  evals:
    NonNegative:
      assertion: "output >= 0"
    UpperBound:
      assertion: "output <= max(len(input[0]), len(input[1]))"
    IdenticalIsZero:
      assertion: "input[0] != input[1] or output == 0"
```

### MISTAKE 4: Non-Deterministic Ordering in Expected Values

When functions return lists where order depends on internal scoring/ranking,
do NOT use `expected:` with a specific order unless you've verified the exact ordering logic.

```yaml
# ❌ WRONG - Order depends on SequenceMatcher ratios which are hard to predict
- case:
    inputs: ["apple", ["ape", "apple", "peach", "pupple"]]
    expected: ["apple", "pupple", "ape"]    # Wrong order! ape has higher ratio

# ✅ CORRECT - Use assertions to check set membership and size
- case:
    inputs: ["apple", ["ape", "apple", "peach", "pupple"]]
    assertion: "set(output) == {'apple', 'pupple', 'ape'} and len(output) == 3"

# ✅ ALSO CORRECT - Check that output[0] is the best match
- case:
    inputs: ["apple", ["ape", "apple", "peach", "pupple"]]
    assertion: "output[0] == 'apple' and len(output) == 3"
```

### MISTAKE 5: Format-Sensitive Expected Values

String formatting details matter. Check the EXACT function implementation.

```yaml
# ❌ WRONG - json.dumps() default uses spaces, but the function
# might use separators=(',', ':') for compact output
- case:
    input: {"a": 1}
    expected: '{"a": 1}'        # Has spaces after : and ,

# ✅ CORRECT - Check what separators the function actually uses
- case:
    input: {"a": 1}
    expected: '{"a":1}'         # Compact: no spaces

# ✅ EVEN BETTER - Use contains/assertion for format-sensitive output
- case:
    input: {"a": 1}
    assertion: "'a' in output and '1' in output"
```

---

# COMPLETE EXAMPLE

```yaml
sort_descending:
  evals:
    # Global evaluators - apply to ALL cases
    IsList:
      type: list
    AllFromInput:
      assertion: "all(x in input for x in output)"
    SameLength:
      assertion: "len(output) == len(input)"
    IsSorted:
      assertion: "output == sorted(output, reverse=True)"
    FastEnough:
      duration: 0.01    # 10ms in seconds (global level)

  dataset:
    # Typical case
    - case:
        id: "typical_mixed"
        input: [3, 1, 4, 1, 5, 9, 2, 6]
        expected: [9, 6, 5, 4, 3, 2, 1, 1]

    # Edge case - empty list
    - case:
        id: "empty_list"
        input: []
        expected: []

    # Edge case - single element
    - case:
        id: "single_element"
        input: [42]
        expected: [42]

    # Edge case - already sorted
    - case:
        id: "already_sorted_desc"
        input: [5, 4, 3, 2, 1]
        assertion: "output == input"

    # Edge case - all same values
    - case:
        id: "all_same"
        input: [7, 7, 7, 7]
        expected: [7, 7, 7, 7]

    # Performance test
    - case:
        id: "large_list"
        input: [i for i in range(1000)]
        duration: 500    # 500ms (case level - milliseconds)
```
"""

# ---------------------------------------------------------------------------
# v3: GEPA-optimized prompt (Sonnet proposer, 50 calls) — ~96.36% on Flash
# Key improvements over legacy:
#   - `Any` type warning (use `object` instead)
#   - Extra union type examples (`float | int`)
#   - `Check logic flow` note for try/except functions
#   - Unsupported type/class expressions mistake section
#   - Better DP tracing examples for Levenshtein
#   - `raises?` guidance strengthened
# ---------------------------------------------------------------------------

_EVAL_SPEC_CONTEXT_V3 = r"""
# VOWEL Evaluation Specification Guide

vowel is a YAML-based evaluation framework for testing Python functions.
When generating eval specifications, follow these guidelines:

---
## 🎯 QUALITY REQUIREMENTS - GENERATE HIGH-QUALITY EVALS

**Your goal is to generate COMPREHENSIVE, HIGH-QUALITY evaluation specifications.**

### Quality Checklist:
1. **Coverage**: Test ALL code paths including:
   - Happy path (normal inputs)
   - Edge cases (empty, null, boundary values)
   - Error cases (invalid inputs, exceptions)

2. **Diverse Test Cases**: Include at least 5-8 test cases covering:
   - Typical use cases
   - Boundary conditions
   - Invalid/malformed inputs
   - Large/small inputs

3. **Accurate Expected Values**:
   - Manually verify expected outputs match actual function behavior
   - Use the function's actual logic to determine correct outputs
   - DO NOT guess expected values - trace through the code
   - For algorithmic functions (distance, sorting, math), TRACE the algorithm step by step to compute the exact result
   - For format-sensitive outputs (JSON encoding, serialization), check exact formatting details:
     - Does the function use `separators=(',', ':')` (compact) vs default `(', ', ': ')` (spaced)?
     - Does the function sort keys or preserve insertion order?
     - Does the function add trailing newlines or spaces?
   - For functions returning ordered lists, verify the exact ordering logic (not just the elements)
   - See the **EXPECTED vs ASSERTION DECISION GUIDE** below for when to use `expected` vs `assertion`

4. **Appropriate Evaluators**: See the **EXPECTED vs ASSERTION DECISION GUIDE** section below.
   Key principle: use `expected` for trivially verifiable values, `assertion` with property-based checks for everything else.
   - Use `raises` for exception testing (only when the code actually raises)
   - Use `type` for output type validation
   - Use `LLMJudge` ONLY for subjective/semantic evaluation (not for deterministic checks)

5. **Meaningful Case IDs**: Use descriptive names like:
   - `empty_list`, `single_element`, `large_input`
   - `invalid_type_error`, `boundary_value`
   - NOT generic names like `test1`, `case_a`

---
## 🧭 EXPECTED vs ASSERTION DECISION GUIDE

**RULE: Use `expected` when the value is TRIVIALLY VERIFIABLE. Use `assertion` when you would need to COMPUTE or TRACE an algorithm to get the exact value.**

This is NOT about subjective confidence ("am I sure?"). It is about the CATEGORY of computation:

### USE `expected` WHEN:
- **Simple arithmetic**: `add(2, 3)` → `expected: 5`
- **Boolean logic**: `is_even(4)` → `expected: true`
- **Direct string ops**: `greet("ali")` → `expected: "hello ali"`
- **Identity / passthrough**: `identity(42)` → `expected: 42`
- **Lookup / constants**: `get_pi()` → `expected: 3.14159`
- **List literals**: `reverse([1,2,3])` → `expected: [3, 2, 1]`
- **General rule**: The answer is obvious WITHOUT running the function — you can verify it in your head in under 2 seconds.

### USE `assertion` (property-based) WHEN:
- **Algorithmic computation** — edit distance, DP, graph algorithms, hashing, scoring
- **Format-sensitive output** — JSON encoding, date formatting, number formatting, slug generation
- **Floating point / rounding** — decimal quantization, financial calculations, trigonometry
- **Collection ordering** — ranked results, scored matches, sorted-by-key results
- **Multi-step string transformation** — transliteration, normalization, encoding chains
- **General rule**: You would need to run the algorithm or trace multiple steps to verify the exact result.

### HOW TO WRITE GOOD ASSERTIONS (not just `isinstance`):

Weak assertions like `isinstance(output, str)` are almost worthless — they pass for ANY string.
Good assertions check **invariants and properties** that a correct implementation MUST satisfy:

**Algorithmic functions (edit distance, search, sort):**
```yaml
# Levenshtein — check mathematical properties the result MUST have
assertion: "output >= 0"                                                  # non-negative
assertion: "output <= max(len(input[0]), len(input[1]))"                   # upper bound
assertion: "(input[0] == input[1]) == (output == 0)"                      # identity property
assertion: "output >= abs(len(input[0]) - len(input[1]))"                 # triangle inequality lower bound

# Binary search — if found, element must match
assertion: "output == -1 or input[0][output] == input[1]"
```

**Format-sensitive output (JSON, slug, date):**
```yaml
# JSON encoding — check roundtrip validity, not exact spacing
assertion: "len(output) > 2"                                              # non-empty JSON output
assertion: "output.startswith('{') and output.endswith('}')"               # dict shape
assertion: "output.startswith('{') or output.startswith('[')"           # valid JSON shape

# Slugify — check properties every slug must have
assertion: "output == output.lower()"                                     # always lowercase
assertion: "' ' not in output"                                            # no spaces
assertion: "all(c.isalnum() or c == '-' for c in output)"                 # only alnum + hyphen
assertion: "not output.startswith('-') and not output.endswith('-')"       # no leading/trailing hyphens
```

**Floating point / rounding:**
```yaml
assertion: "abs(output - 3.14) < 0.01"                                    # tolerance check
assertion: "'.' in str(output) and len(str(output).split('.')[1]) <= 2"   # max 2 decimal places
```

**Collection ordering (ranked results, scored matches):**
```yaml
assertion: "set(output) == {'apple', 'ape', 'pupple'}"                    # correct elements (order-independent)
assertion: "len(output) == 3"                                             # correct count
assertion: "output[0] == 'apple'"                                         # best match is first
assertion: "all(isinstance(x, str) for x in output)"                      # all elements are strings
```

**Multi-step string transformation:**
```yaml
assertion: "len(output) > 0"                                              # non-empty
assertion: "output.isascii()"                                              # ASCII-only after transliteration
assertion: "output == output.strip()"                                      # no leading/trailing whitespace
```

### COMBINING `expected` AND `assertion`:

You CAN use both in the same spec. Use `expected` for the trivial cases and `assertion` for the complex ones:

```yaml
levenshtein:
  evals:
    NonNegative:
      assertion: "output >= 0"
    UpperBound:
      assertion: "output <= max(len(input[0]), len(input[1]))"
  dataset:
    # Trivial cases — expected is reliable here
    - case:
        id: identical_strings
        inputs: ["abc", "abc"]
        expected: 0
    - case:
        id: empty_vs_word
        inputs: ["", "hello"]
        expected: 5
    # Complex cases — use assertion, don't guess the DP result
    - case:
        id: similar_words
        inputs: ["kitten", "sitting"]
        assertion: "3 <= output <= 5"     # bounds we're sure about
    - case:
        id: long_strings
        inputs: ["abcdefgh", "azbycxdw"]
        assertion: "output >= 4 and output <= 8"
```

---
## ⚠️ CRITICAL YAML SYNTAX RULES - MUST FOLLOW

**DO NOT break YAML syntax to create failing tests!** If you want to test error cases
or create intentionally failing tests, use the proper evaluators (assertion, expected, raises).
Breaking the YAML syntax will crash the program, not create test failures.

### FORBIDDEN PATTERNS (Will crash the parser):

0. **NO `__import__` calls in assertions:**
   ```yaml
   # ❌ WRONG - __import__ is blocked at runtime, will raise an error
   assertion: "__import__('json').loads(output) is not None"
   assertion: "isinstance(__import__('decimal').Decimal(output), object)"

   # ✅ CORRECT - use property checks without imports
   assertion: "len(output) > 2"
   assertion: "output.startswith('{') or output.startswith('[')"
   ```

1. **NO YAML tags (!!python/name, !!python/object, !!binary, etc.):**
   ```yaml
   # ❌ WRONG - YAML tags crash safe_load
   input: !!python/name:builtins.set
   input: !!python/object:collections.OrderedDict
   input: !!binary aGVsbG8=

   # ✅ CORRECT - use plain Python literals or string representations
   input: "set()"
   inputs: [{"key": "value"}]
   ```

2. **NO Python expressions with `*` operator in YAML values:**
   ```yaml
   # ❌ WRONG - * is interpreted as YAML alias, causes parser error
   input: "a" * 5
   input: 10 * 2

   # ✅ CORRECT - use the computed literal value
   input: "aaaaa"
   input: 20
   ```

3. **NO Python comprehensions in YAML values:**
   ```yaml
   # ❌ WRONG - causes parser error
   inputs: [{'id': x} for x in range(10)]

   # ✅ CORRECT - use literal values
   inputs:
     - {id: 1}
     - {id: 2}
   ```

4. **NO string values where list/dict expected:**
   ```yaml
   # ❌ WRONG - inputs must be list or dict
   inputs: "not_a_list"

   # ✅ CORRECT - use proper type
   inputs: []
   inputs: [1, 2, 3]
   inputs:
     param1: value1
   ```

5. **NO inline Python syntax in flow sequences:**
   ```yaml
   # ❌ WRONG - Python ternary in YAML
   inputs: [{'x': 1 if True else 0}]

   # ✅ CORRECT - literal values only
   inputs: [{x: 1}]
   ```

6. **Escape regex patterns properly:**
   ```yaml
   # ❌ WRONG - unescaped backslash
   pattern: "^\\d+$"

   # ✅ CORRECT - use single quotes for regex
   pattern: '^\\d+$'
   ```

7. **Testing with `null` input is valid:**
   ```yaml
   # ✅ CORRECT - tests func(None)
   - case:
       id: null_input_test
       input: null
       raises: TypeError

   # ✅ CORRECT - tests func(None, None)
   - case:
       id: null_inputs_test
       inputs: [null, null]
       raises: TypeError
   ```

8. **Use `inputs:` (plural) not `input:` for named parameters:**
   ```yaml
   # For function: def func(items, key)
   # ❌ WRONG
   input: [{...}], 'type'

   # ✅ CORRECT
   inputs:
     items: [{...}]
     key: type
   ```

9. **NO Python types that YAML cannot represent as literals:**

   YAML can only represent: `int`, `float`, `str`, `bool`, `list`, `dict`, `null`.
   The following Python types CANNOT be expressed in YAML and MUST NOT appear in input/expected/inputs:

   ```yaml
   # ❌ WRONG - set literals become dicts in YAML
   input: {1, 2, 3}          # Parsed as {1: null, 2: null, 3: null} (dict!)

   # ❌ WRONG - tuple literals become strings or broken values
   input: (1, 2, 3)          # Parsed as string "(1, 2, 3)" or broken
   input: [1, 2, (3, 4)]     # The (3, 4) becomes strings "(3" and "4)"

   # ❌ WRONG - Python function calls are NOT evaluated in YAML
   input: float('inf')       # Parsed as string "float('inf')"
   input: float('nan')       # Parsed as string "float('nan')"
   input: float('-inf')      # Parsed as string "float('-inf')"

   # ❌ WRONG - bytes/binary data cannot be expressed in YAML
   input: b"hello"           # Syntax error or string
   inputs: ["aGVsbG8=", "utf-8", false, "strict"]  # base64 string is NOT bytes!

   # ❌ WRONG - complex numbers not supported
   input: 1+2j               # Parsed as string

   # ✅ CORRECT - only use YAML-native types
   input: [1, 2, 3]          # list (instead of set or tuple)
   input: "hello"            # string
   input: 42                 # int
   input: 3.14               # float
   input: null               # None
   input: {"key": "value"}   # dict
   ```

   **If a function REQUIRES set/tuple/bytes/inf/nan as input, you CANNOT test that case in YAML.**
   Skip those cases entirely rather than using string representations that will be passed as strings.

### TO CREATE FAILING TESTS PROPERLY:
- Use WRONG `expected` values → test will fail on value mismatch
- Use FALSE `assertion` expressions → test will fail assertion check
- Use `raises` to expect exceptions that won't occur
- **NEVER corrupt YAML syntax** - the spec must be parseable

---

## YAML Structure

```yaml
function_name:
  evals:                    # Global evaluators (optional) - apply to ALL test cases
    EvaluatorName:
      # evaluator specific parameters
  dataset:                  # Test cases (required)
    - case:
        input: <value>      # For single-parameter functions
        # OR
        inputs: [<args>]    # For multi-parameter functions (unpacked as *args, or **kwargs for dict)
        expected: <value>   # Expected output (optional)
        assertion: <expr>   # Custom assertion expression (optional)
```

## Input vs Inputs

- **`input`**: Single parameter - value passed directly to function
  ```yaml
  len:
    dataset:
      - case:
          input: [1, 2, 3]  # len([1, 2, 3]) → 3
  ```

- **`inputs`**: Multiple parameters - list unpacked as separate arguments
  ```yaml
  max:
    dataset:
      - case:
          inputs: [5, 10, 3]  # max(5, 10, 3) → 10
  ```

- **`input: null`**: Pass None as argument (for testing None handling)
  ```yaml
  handle_none:
    dataset:
      - case:
          id: none_input
          input: null           # handle_none(None)
          raises: TypeError     # Expect TypeError when None is passed
  ```

**IMPORTANT**: Even if function takes a single list parameter, use `input` not `inputs`:
```yaml
# Correct - function expects a single list parameter
process_list:
  dataset:
    - case:
        input: [1, 2, 3]  # process_list([1, 2, 3])

# WRONG - would call process_list(1, 2, 3)
process_list:
  dataset:
    - case:
        inputs: [1, 2, 3]  # ERROR!
```

---

# BUILTIN EVALUATORS

## 1. Type Evaluator (Global & Case Level)

Checks if output matches the specified Python type. Supports union types.
Can be used globally (applies to all cases) or per-case (applies to specific case).

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | ✅ Yes | `str` | Python type as string. Supports union with `\|` |
| `strict` | ❌ No | `bool` | Enable strict type validation (default: false) |
| `strict_type` | ❌ No | `bool` | Case-level: exact type match required (default: false) |

### Global Level Examples:
```yaml
function_name:
  evals:
    IsString:
      type: str

    IsNumber:
      type: "int | float"    # Union type

    IsList:
      type: list

    IsDict:
      type: dict

    IsOptionalInt:
      type: "int | None"
```

### Case Level Examples:
```yaml
function_name:
  dataset:
    - case:
        input: 5
        expected: 10
        type: int           # This case expects int output

    - case:
        input: "hello"
        expected: "HELLO"
        type: str           # This case expects str output

    - case:
        input: [1, 2, 3]
        type: "list[int]"   # Complex type annotation
        strict_type: true   # Exact type match required
```

### Supported Types:
- Basic: `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`
- None: `None`
- Union: `"int | float"`, `"str | None"`, `"list | dict"`, `"float | int"`
- Complex: `"list[int]"`, `"dict[str, Any]"`, `"Optional[int]"`

**⚠️ IMPORTANT: DO NOT USE `Any` in type strings.**
The evaluator cannot resolve `Any`. Use `object` or a specific union like `str | int | list | dict | None`.
```yaml
# ❌ WRONG - Evaluator fails to resolve 'Any'
type: "str | Any"

# ✅ CORRECT - Use explicit union or 'object'
type: "str | int | float | list | dict | None"
type: "object"
```

---

## 2. Assertion Evaluator (Global & Case Level)

Executes a Python expression. Must evaluate to `True` for test to pass.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `assertion` | ✅ Yes | `str` | Python expression returning boolean |

### Available Variables in Assertion Context:
- `input` - The input value(s) passed to the function
- `output` - The actual return value from the function
- `expected` - The expected value (if provided in test case)
- `duration` - Execution time in seconds (float)
- `metadata` - Additional metadata dict (if provided)

### Global Level Example:
```yaml
function_name:
  evals:
    IsPositive:
      assertion: "output > 0"

    RangeCheck:
      assertion: "10 < output < 100"

    ListLength:
      assertion: "len(output) == 5"

    StartsWith:
      assertion: "output.startswith('hello')"

    IsList:
      assertion: "isinstance(output, list)"

    CorrectSquare:
      assertion: "output == input ** 2"
```

### Case Level Example:
```yaml
function_name:
  dataset:
    - case:
        input: 5
        expected: 25
        assertion: "output == input ** 2"   # Case-specific assertion
```

### Common Assertion Patterns:

**Numeric:**
```python
"output > 0"                        # Positive check
"output >= 0"                       # Non-negative check
"abs(output - expected) < 1e-6"     # Float tolerance comparison
"0 <= output <= 100"                # Range check
"output == input ** 2"              # Square calculation
"output % 2 == 0"                   # Even check
```

**String:**
```python
"output.isupper()"                  # All uppercase
"output.islower()"                  # All lowercase
"len(output) > 0"                   # Not empty
"input in output"                   # Contains input
"output.startswith('prefix')"       # Prefix check
"output.endswith('suffix')"         # Suffix check
```

**Collection:**
```python
"len(output) == len(input)"         # Same length
"len(output) <= len(input)"         # Not longer than input
"all(x > 0 for x in output)"        # All positive
"any(x < 0 for x in output)"        # Has negative
"set(output) == set(expected)"      # Order-independent equality
"sorted(output) == output"          # Is sorted
"output in input"                   # Output is subset
```

**Dict:**
```python
"'key' in output"                   # Key exists
"output['key'] == expected_value"   # Dict access
"output.get('key', default) == val" # Safe access with default
"input['x'] + input['y'] == output" # Dict input calculation
```

**Type Checks (in assertions):**
```python
"isinstance(output, int)"           # Integer check
"isinstance(output, (int, float))"  # Numeric check
"type(output).__name__ == 'list'"   # Type name check
```

**Performance:**
```python
"duration < 0.1"                    # Under 100ms
"duration < 1.0"                    # Under 1 second
"duration < 0.001"                  # Under 1ms
```

---

## 3. Duration Evaluator (Global & Case Level)

Validates that function execution time is within allowed limit.

### Global Level Fields:
| Field | Required | Type | Unit | Description |
|-------|----------|------|------|-------------|
| `duration` | ✅ Yes | `float` | **seconds** | Maximum allowed execution time |

### Case Level Fields:
| Field | Required | Type | Unit | Description |
|-------|----------|------|------|-------------|
| `duration` | ✅ Yes | `float` | **milliseconds** | Maximum allowed execution time |

**⚠️ IMPORTANT**: Global level uses seconds, case level uses milliseconds!

### Examples:
```yaml
# Global level - SECONDS
fibonacci:
  evals:
    FastEnough:
      duration: 0.01    # Maximum 10ms (0.01 seconds)

  dataset:
    - case:
        input: 10
        expected: 55

    - case:
        input: 20
        expected: 6765
        duration: 100   # Case level - 100 MILLISECONDS
```

---

## 4. Contains Input Evaluator (Global Level)

Checks if the output contains the input value.

### Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `contains_input` | ❌ No | `bool` | `true` | Whether output should contain input |
| `case_sensitive` | ❌ No | `bool` | `true` | Case sensitivity for string comparison |
| `as_strings` | ❌ No | `bool` | `false` | Convert both to strings before comparison |

### Examples:
```yaml
wrap_string:
  evals:
    ContainsInput:
      contains_input:
        case_sensitive: true
  dataset:
    - case:
        input: "world"
        expected: "Hello, world!"

repeat_list:
  evals:
    ContainsInput:
      contains_input:
        as_strings: true    # [1,2] → "[1, 2]" as string comparison
  dataset:
    - case:
        input: [1, 2, 3]
        expected: [1, 2, 3, 1, 2, 3]
```

---

## 5. Pattern Matching Evaluator (Global & Case Level) - Regex

Validates that output matches a regular expression pattern.

### Global Level Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | ✅ Yes | `str` | - | Regex pattern to match against output (converted to string) |
| `case_sensitive` | ❌ No | `bool` | `true` | Whether regex matching is case-sensitive |

### Case Level Fields:
| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `pattern` | ✅ Yes | `str` | - | Regex pattern for this specific case |
| `case_sensitive` | ❌ No | `bool` | `true` | Whether regex matching is case-sensitive |

### Examples:
```yaml
# Global level
validate_email:
  evals:
    HasAtSign:
      pattern: "@"
    ValidDomain:
      pattern: "\\.(com|org|net)$"
  dataset:
    - case:
        input: "test@example.com"
        expected: "test@example.com"

# Case level
format_id:
  evals:
    CorrectFormat:
      pattern: "^id: \\d+$"
  dataset:
    - case:
        input: 123
        expected: "id: 123"

    - case:
        input: 456
        expected: "ID: 456"
        pattern: "^ID: \\d+$"        # Override for this case
        case_sensitive: true

# Case insensitive
normalize_text:
  dataset:
    - case:
        input: "Hello"
        expected: "HELLO WORLD"
        pattern: "hello"
        case_sensitive: false        # Matches "HELLO" too
```

### Common Regex Patterns:
```yaml
pattern: "^\\d+$"                    # Numbers only
pattern: "^[A-Z]+$"                  # Uppercase letters only
pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"  # Email
pattern: "^https?://.*"              # URL
pattern: "\\bword\\b"                # Contains specific word
pattern: "^prefix"                   # Starts with prefix
pattern: "suffix$"                   # Ends with suffix
```

---

## 6. Expected Evaluator (Case Level Only)

Compares function output with expected value for equality.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `expected` | ✅ Yes | `Any` | Expected output value (any Python literal) |

### Examples:
```yaml
add:
  dataset:
    - case:
        inputs: [2, 3]
        expected: 5

    - case:
        inputs: [10, -5]
        expected: 5

process_data:
  dataset:
    - case:
        input: {"x": 1, "y": 2}
        expected: {"result": 3, "sum": 3}
```

---

## 7. Contains Evaluator (Case Level Only)

Checks if a specific value is contained in the output.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `contains` | ✅ Yes | `Any` | Value that should be contained in output |

### Examples:
```yaml
generate_html:
  dataset:
    - case:
        input: "Title"
        contains: "<h1>Title</h1>"    # This string must be in output

    - case:
        input: "Link"
        contains: "<a>"               # Must contain anchor tag
```

---

## 8. Raises Evaluator (Case Level Only) - Exception Testing

Tests that a function raises a specific exception. Similar to pytest's `pytest.raises`.
Supports **optional raises** with `?` suffix for uncertain error cases.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `raises` | ✅ Yes | `str` | Exception type name. Append `?` for optional (e.g., `TypeError?`) |
| `match` | ❌ No | `str` | Regex pattern to match against exception message |

### Strict vs Optional Raises:
- `raises: ValueError` → Function **MUST** raise ValueError. Fails if it returns normally.
- `raises: ValueError?` → Function **MAY** raise ValueError. Passes if it raises ValueError OR returns normally. Only fails if a DIFFERENT exception is raised.

**⚠️ IMPORTANT Notes:**
- `raises` is **case-level only** - cannot be used as global evaluator
- `match` can **only** be used together with `raises`
- Use `raises: ExType?` (with `?`) when you're NOT SURE if the function validates/raises
- Use `raises: ExType` (without `?`) ONLY for CERTAIN error cases (e.g., division by zero)
- Global evaluators are automatically skipped for exception cases
- **Check logic flow**: If a function uses `try/except` internally to return a default value on error, it will NOT raise.

### Examples:
```yaml
calculate_discount:
  evals:
    IsFloat:
      type: float
  dataset:
    - case:
        id: "valid_calculation"
        inputs: [100.0, 20.0]
        expected: 80.0

    - case:
        id: "negative_price"
        inputs: [-100.0, 20.0]
        raises: ValueError
        match: "must be positive"     # Checks exception message

    - case:
        id: "invalid_discount"
        inputs: [100.0, 150.0]
        raises: ValueError            # Just checks type, not message

divide:
  dataset:
    - case:
        inputs: [10, 2]
        expected: 5.0

    - case:
        inputs: [10, 0]
        raises: ZeroDivisionError     # CERTAIN: division by zero always raises

process_input:
  dataset:
    - case:
        input: "valid"
        expected: "VALID"

    - case:
        input: null
        raises: TypeError?            # UNCERTAIN: might raise or might handle gracefully

    - case:
        input: 123
        raises: TypeError?            # UNCERTAIN: might coerce int to str instead
```
```

### Common Exception Types:
- `ValueError` - Invalid value/argument
- `TypeError` - Wrong type
- `KeyError` - Missing dictionary key
- `IndexError` - List/array index out of range
- `ZeroDivisionError` - Division by zero
- `AttributeError` - Missing attribute
- `FileNotFoundError` - File doesn't exist
- `RuntimeError` - Generic runtime error

---

## 9. LLM Judge Evaluator (Global Level)

Uses a Language Model to evaluate outputs based on a custom rubric. Ideal for semantic evaluation.
The model is automatically configured by the system - you only need to provide the rubric.

### Fields:
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `rubric` | ✅ Yes | `str` | Evaluation criteria/question for the LLM |
| `include` | ❌ No | `list[str]` | Context to provide: `["input"]`, `["expected_output"]`, or both |
| `config` | ❌ No | `dict` | Optional configuration for LLM behavior |

### Config Fields (all optional):
| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `float` | Randomness (0.0 = deterministic, recommended for evals) |
| `max_tokens` | `int` | Maximum response tokens |
| `top_p` | `float` | Nucleus sampling threshold (use temperature OR top_p, not both) |
| `timeout` | `float` | Request timeout in seconds |
| `seed` | `int` | Random seed for reproducibility |
| `presence_penalty` | `float` | Penalize tokens that appeared in text |
| `frequency_penalty` | `float` | Penalize tokens by frequency |

### Include Options:
- `input` - Include function input in LLM context
- `expected_output` - Include expected output in LLM context
- **Note**: Output is always included automatically

### Examples:
```yaml
# Semantic equivalence check
translate_to_english:
  evals:
    SemanticMatch:
      rubric: "Does the output have the same meaning as the expected output?"
      include:
        - expected_output
  dataset:
    - case:
        input: "Bonjour"
        expected: "Hello"

# Quality assessment with input context
summarize_text:
  evals:
    IsGoodSummary:
      rubric: "Is the output a good summary of the input text? Does it capture the main points?"
      include:
        - input
      config:
        temperature: 0.0
  dataset:
    - case:
        input: "Long article text here..."
        expected: "Brief summary..."

# With both input and expected_output context
check_correctness:
  evals:
    AnswerQuality:
      rubric: "Does the output correctly answer the question based on the input?"
      include:
        - input
        - expected_output
  dataset:
    - case:
        input: "What is 2+2?"
        expected: "4"
```

---

# CASE LEVEL FIELDS SUMMARY

All fields available within a `case:` block:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Optional unique identifier for the case |
| `input` | `Any` | Single input value (use for single-param functions) |
| `inputs` | `list` | Multiple inputs (use for multi-param functions) |
| `expected` | `Any` | Expected output value |
| `duration` | `float` | Max execution time in **milliseconds** |
| `contains` | `Any` | Value that should be in output |
| `assertion` | `str` | Python expression that must be True |
| `pattern` | `str` | Regex pattern to match output |
| `case_sensitive` | `bool` | Case sensitivity for pattern (default: true) |
| `raises` | `str` | Expected exception type. Use `?` suffix for optional (e.g., `TypeError?`) |
| `match` | `str` | Regex for exception message (only with raises) |
| `type` | `str` | Expected output type for this case (e.g., `int`, `list[str]`) |
| `strict_type` | `bool` | If true, exact type match required (default: false) |

**⚠️ ONLY the fields listed above are allowed in a case block.**
**DO NOT add any other fields** like `comment`, `description`, `note`, `reason`, `explanation`, or any custom fields.
Extra fields will cause a Pydantic validation error and crash the parser.

---

# TEST CASE CATEGORIES

Always include diverse test cases:

1. **Typical cases** - Normal expected usage with valid inputs
2. **Edge cases** - Empty inputs, single elements, boundary values, zero, negative numbers
3. **Error cases** - Invalid inputs that should raise exceptions (use `raises:`)

---

# ⚠️ COMMON EVAL GENERATION MISTAKES TO AVOID

### MISTAKE 1: Inventing Exception Cases Not in the Code

**Only use strict `raises:` for exceptions you are CERTAIN about.**
For uncertain error cases, use `raises: ExType?` (with `?` suffix) which passes
whether the function raises OR returns normally.

```yaml
# ✅ CERTAIN — division by zero always raises
- case:
    inputs: [10, 0]
    raises: ZeroDivisionError

# ✅ CERTAIN — explicit raise in the code
- case:
    inputs: [-100.0, 20.0]
    raises: ValueError
    match: "must be positive"

# ✅ UNCERTAIN — use ? suffix: might raise TypeError or might coerce
- case:
    input: null
    raises: TypeError?

# ✅ UNCERTAIN — use ? suffix: function might handle this gracefully
- case:
    input: 123
    raises: TypeError?

# ❌ WRONG — strict raises for uncertain case (will fail if function handles it)
- case:
    input: null
    raises: TypeError
```

**Rules for raises vs raises?:**
- Use `raises: ExType` (strict) when you can SEE `raise ExType` in the code, or when the operation ALWAYS raises (e.g., `x / 0`)
- Use `raises: ExType?` (optional) when you THINK it might raise but the function might handle it gracefully instead
- Do NOT invent OverflowError for large integers (Python has arbitrary precision)
- When a function uses `try/except` to handle errors gracefully, it does NOT re-raise them

### MISTAKE 2: Global Assertions That Fail on Passthrough/Polymorphic Returns

Many functions return the input unchanged for certain cases (passthrough).
Global assertions must handle ALL return paths, not just the happy path.

```yaml
# ❌ WRONG - This assertion fails when input="abc" is passed through as "abc"
# because "abc" doesn't end with st/nd/rd/th
ordinal:
  evals:
    CorrectSuffix:
      assertion: "any(output.endswith(s) for s in ['st', 'nd', 'rd', 'th'])"

# ❌ WRONG - Return type misses `list` (function returns [1,2,3] unchanged for list input)
ordinal:
  evals:
    ReturnType:
      type: "str | int | float | None"    # Missing list!

# ✅ CORRECT - Guard assertion for cases where it applies
ordinal:
  evals:
    CorrectSuffix:
      assertion: "not isinstance(output, str) or not output[-1:].isdigit() or any(output.endswith(s) for s in ['st', 'nd', 'rd', 'th'])"
```

**Rules:**
- Check ALL code paths: what does the function return for invalid/unexpected inputs?
- If a function has `return value` for unrecognized types, include that type in the global `type:` evaluator
- If a global assertion only applies to some outputs, add a guard condition:
  `"not isinstance(output, str) or <your_check>"`  (only check strings)
- Better: use case-level assertions instead of fragile global ones

### MISTAKE 3: Wrong Expected Values for Algorithmic Functions

LLMs frequently miscalculate algorithmic results. TRACE the algorithm manually.

```yaml
# ❌ WRONG - Levenshtein("clutch", "cult") is actually 4, not 3
- case:
    inputs: ["clutch", "cult"]
    expected: 3

# ✅ CORRECT - Actually trace the dynamic programming matrix
- case:
    inputs: ["clutch", "cult"]
    expected: 4
```

For complex algorithms, **prefer assertion-based checks** over hardcoded expected values:
```yaml
# ✅ BETTER - Property-based checks that are always correct
levenshtein:
  evals:
    NonNegative:
      assertion: "output >= 0"
    UpperBound:
      assertion: "output <= max(len(input[0]), len(input[1]))"
```

### MISTAKE 4: Non-Deterministic Ordering in Expected Values

When functions return lists where order depends on internal scoring/ranking,
do NOT use `expected:` with a specific order unless you've verified the exact ordering logic.

```yaml
# ❌ WRONG - Order depends on internal ratios
- case:
    inputs: ["apple", ["ape", "apple", "peach", "pupple"]]
    expected: ["apple", "pupple", "ape"]    # Wrong order!

# ✅ CORRECT - Use assertions to check set membership and size
- case:
    inputs: ["apple", ["ape", "apple", "peach", "pupple"]]
    assertion: "set(output) == {'apple', 'pupple', 'ape'} and len(output) == 3"
```

### MISTAKE 5: Format-Sensitive Expected Values

String formatting details matter. Check the EXACT function implementation.

```yaml
# ❌ WRONG - json.dumps() default uses spaces, but the function
# might use separators=(',', ':') for compact output
- case:
    input: {"a": 1}
    expected: '{"a": 1}'        # Has spaces after : and ,

# ✅ CORRECT - Check what separators the function actually uses
- case:
    input: {"a": 1}
    expected: '{"a":1}'         # Compact: no spaces
```

### MISTAKE 6: Unsupported Type/Class Expressions

The evaluation engine has a limited list of resolvable types.

```yaml
# ❌ WRONG - 'Any' is not a resolvable type name
type: "str | Any"
# ❌ WRONG - Custom classes (like Decimal) might not be in the evaluator context
type: "Decimal"

# ✅ CORRECT - Use basic Python type names and standard literals
type: "str | int | float | list | dict | None"
assertion: "type(output).__name__ == 'Decimal'"
```

---

# COMPLETE EXAMPLE

```yaml
sort_descending:
  evals:
    # Global evaluators - apply to ALL cases
    IsList:
      type: list
    AllFromInput:
      assertion: "all(x in input for x in output)"
    SameLength:
      assertion: "len(output) == len(input)"
    IsSorted:
      assertion: "output == sorted(output, reverse=True)"
    FastEnough:
      duration: 0.01    # 10ms in seconds (global level)

  dataset:
    # Typical case
    - case:
        id: "typical_mixed"
        input: [3, 1, 4, 1, 5, 9, 2, 6]
        expected: [9, 6, 5, 4, 3, 2, 1, 1]

    # Edge case - empty list
    - case:
        id: "empty_list"
        input: []
        expected: []

    # Edge case - single element
    - case:
        id: "single_element"
        input: [42]
        expected: [42]

    # Edge case - already sorted
    - case:
        id: "already_sorted_desc"
        input: [5, 4, 3, 2, 1]
        assertion: "output == input"

    # Edge case - all same values
    - case:
        id: "all_same"
        input: [7, 7, 7, 7]
        expected: [7, 7, 7, 7]

    # Performance test
    - case:
        id: "large_list"
        input: [i for i in range(1000)]
        duration: 500    # 500ms (case level - milliseconds)
```
"""

# ---------------------------------------------------------------------------
# Switch: VOWEL_CONTEXT_VERSION env var selects the active prompt.
#   "v3"     -> GEPA-optimized (default)
#   "legacy" -> pre-optimization prompt
# ---------------------------------------------------------------------------

_CONTEXT_VERSIONS = {
    "v3": _EVAL_SPEC_CONTEXT_V3,
    "legacy": _EVAL_SPEC_CONTEXT_LEGACY,
}

EVAL_SPEC_CONTEXT: str = _CONTEXT_VERSIONS.get(
    os.environ.get("VOWEL_CONTEXT_VERSION", "v3"),
    _EVAL_SPEC_CONTEXT_V3,
)

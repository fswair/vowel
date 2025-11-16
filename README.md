# VOWEL - YAML Based Evaluation Specification

Modular evaluation system. Reads and tests function evaluations from YAML files.

## 🚀 Installation

### Install Directly

```bash
# Install in development mode (editable mode)
pip install -e .

# Or normal installation
pip install .
```

### Install from PyPI

```bash
# Install in development mode (editable mode)
pip install vowel

# Or install with uv
uv add vowel
```

## 🎯 Usage

After installation, the `vowel` command is available system-wide:

```bash
# Normal mode - logfire disabled (fast and clean output)
vowel multi_evals.yml

# Debug mode - logfire enabled (detailed logging)
vowel multi_evals.yml --debug

# Test only specific functions (comma-separated)
vowel multi_evals.yml -f make_list,make_square

# Test specific functions (multiple -f flags)
vowel multi_evals.yml -f make_list -f make_square

# Verbose mode - show error details
vowel multi_evals.yml -v

# Use all options together
vowel tests.yml -f uppercase,lowercase --debug -v

# Help
vowel --help
```

### CLI Options

- `--debug`: Enables debug mode with detailed logging via logfire
- `-f, --filter`: Tests only the specified function(s). Use comma-separated values or multiple flags: `-f func1,func2` or `-f func1 -f func2`
- `-v, --verbose`: Shows detailed output including error reasons
- `--help`: Shows help message

## 📝 YAML Syntax

### Basic Structure

```yaml
function_name:
  evals: # Global evaluators (optional)
    EvaluatorName:
      # evaluator specific parameters
  dataset: # Test cases
    - case:
        input: <input> # For single-parameter functions
        # or
        inputs: [<arg1>, <arg2>, ...] # For multi-parameter functions
        expected: <expected_output> # Optional
        assertion: <custom_check> # Optional, case-specific assertion
        # case-specific evaluators
```

### Function Naming

vowel allows you to use functions in 3 different ways:

```yaml
# 1. Builtin functions (len, str, int, etc.)
len:
  dataset:
    - case:
        input: [1, 2, 3]
        expected: 3

# 2. Standard library (module.function format)
json.dumps:
  dataset:
    - case:
        input: { "key": "value" }
        expected: '{"key": "value"}'

os.path.join:
  dataset:
    - case:
        inputs: ["/home", "user", "file.txt"] # os.path.join("/home", "user", "file.txt")
        expected: "/home/user/file.txt"

# 3. Your own functions (via programmatic API)
# Function name in YAML is passed to run_evals with functions parameter
multiply:
  dataset:
    - case:
        inputs: [2, 3] # multiply(2, 3) - multi-parameter
        expected: 6
```

**To use your own functions:**

```python
from vowel import run_evals

def multiply(x: int, y: int) -> int:
    return x * y

# With YAML file or dict
summary = run_evals(
    "evals.yml",  # or dict
    functions={"multiply": multiply}  # Pass function directly
)
```

**Or with RunEvals Fluent API:**

```python
from vowel import RunEvals

def multiply(x: int, y: int) -> int:
    return x * y

# Cleaner and composable
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"multiply": multiply})
    .debug()
    .run()
)

print(f"Passed: {summary.all_passed}")
```

> **For detailed information:** [API_USAGE.md](API_USAGE.md) and [RUNEVALS_GUIDE.md](RUNEVALS_GUIDE.md)

### Input Formats

**Important:** The difference between `input` and `inputs`:

- **`input`**: For single-parameter functions (input value is passed directly to the function)
- **`inputs`**: For multi-parameter functions (list elements are unpacked as separate arguments)

```yaml
# Single parameter - use 'input'
single_param:
  dataset:
    - case:
        input: 42 # Function: single_param(42)

# Multi-parameter - use 'inputs'
multi_param:
  dataset:
    - case:
        inputs: [10, 20, 30] # Function: multi_param(10, 20, 30)

# Single-parameter function but parameter is a list - use 'input'
list_param:
  dataset:
    - case:
        input: [1, 2, 3] # Function: list_param([1, 2, 3])

# Complex types - single parameter
complex_types:
  dataset:
    - case:
        input: [{ "name": "John", "age": 30 }, { "name": "Jane", "age": 25 }]

# Multi-parameter example - max function
max:
  dataset:
    - case:
        inputs: [5, 10, 3] # max(5, 10, 3)
        expected: 10

    - case:
        inputs: [-1, -5, -2] # max(-1, -5, -2)
        expected: -1

# Single-parameter example - len function
len:
  dataset:
    - case:
        input: [1, 2, 3, 4] # len([1, 2, 3, 4])
        expected: 4

    - case:
        input: "hello" # len("hello")
        expected: 5
```

### Complete Example

```yaml
make_square:
  evals:
    # Global evaluators - apply to all cases
    IsNumber:
      type: "int | float" # Output must be int or float
    PositiveCheck:
      assertion: "output > 0" # Output must be positive
    FastEnough:
      duration: 0.01 # Maximum 0.01 seconds

  dataset:
    # Test case 1
    - case:
        input: 5
        expected: 25 # 5*5 = 25

    # Test case 2 - case-specific evaluator
    - case:
        input: -3
        expected: 9
        duration: 100 # 100ms limit for this case

    # Test case 3 - only global evaluators
    - case:
        input: 10
        # no expected, only global assertions are tested
```

## 📊 Evaluators

### 1. Assertion Evaluator

Executes Python code. The `output` variable contains the function result.

```yaml
function_name:
  evals:
    # Simple check
    PositiveNumber:
      assertion: "output > 0"

    # Complex check
    RangeCheck:
      assertion: "10 < output < 100"

    # List check
    ListLength:
      assertion: "len(output) == 5"

    # String check
    StartsWith:
      assertion: "output.startswith('hello')"

    # Type check
    IsList:
      assertion: "isinstance(output, list)"
```

**Examples:**

```yaml
uppercase:
  evals:
    IsString:
      type: str
    AllCaps:
      assertion: "output.isupper()"
    SameLength:
      assertion: "len(input) == len(output)"
  dataset:
    - case:
        input: "hello"
        expected: "HELLO"

filter_positive:
  evals:
    IsList:
      type: list
    AllPositive:
      assertion: "all(x > 0 for x in output)"
  dataset:
    - case:
        input: [1, -2, 3, -4, 5]
        expected: [1, 3, 5]
```

### 2. Type Evaluator

Checks the type of the output. Supports union types.

```yaml
function_name:
  evals:
    # Single type
    IsString:
      type: str

    # Union type
    IsNumber:
      type: "int | float"

    # List type
    IsList:
      type: list

    # Dict type
    IsDict:
      type: dict
```

**Examples:**

```yaml
make_list:
  evals:
    IsList:
      type: list
  dataset:
    - case:
        input: 5
        expected: [5]

divide:
  evals:
    IsNumber:
      type: "int | float" # accepts int or float
  dataset:
    - case:
        inputs: [10, 2]
        expected: 5.0
```

### 3. Duration Evaluator

Checks the execution time of the function.

```yaml
# Global level (seconds)
function_name:
  evals:
    FastEnough:
      duration: 0.01  # maximum 0.01 seconds (10ms)

# Case level (milliseconds)
function_name:
  dataset:
    - case:
        input: 100
        duration: 50  # maximum 50 milliseconds
```

**Example:**

```yaml
fibonacci:
  evals:
    FastEnough:
      duration: 0.001 # 1ms limit
  dataset:
    - case:
        input: 10
        expected: 55

    - case:
        input: 20
        expected: 6765
        duration: 10 # 10ms limit for this case
```

### 4. Contains Input Evaluator

Checks if the input is contained in the output.

```yaml
function_name:
  evals:
    ContainsInput:
      contains_input:
        case_sensitive: true # Case sensitive (default: true)
        as_strings: false # Convert to string and compare (default: false)
```

**Examples:**

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
        as_strings: true # [1,2] → "[1, 2]" as string
  dataset:
    - case:
        input: [1, 2, 3]
        expected: [1, 2, 3, 1, 2, 3]
```

### 5. Expected Evaluator (Case Level)

Compares the expected output with the actual output.

```yaml
function_name:
  dataset:
    - case:
        input: 5
        expected: 25 # output == 25
```

**Examples:**

```yaml
add:
  dataset:
    - case:
        inputs: [2, 3]
        expected: 5

    - case:
        inputs: [10, -5]
        expected: 5

multiply:
  dataset:
    - case:
        inputs: [3, 4]
        expected: 12
```

### 6. Contains Evaluator (Case Level)

Checks if a specific value is contained in the output.

```yaml
function_name:
  dataset:
    - case:
        input: "test"
        contains: "expected_substring"
```

**Example:**

```yaml
generate_html:
  dataset:
    - case:
        input: "Title"
        contains: "<h1>Title</h1>" # This string must be in output

    - case:
        input: "Link"
        contains: "<a>" # This must also be in output
```

### 7. Pattern Matching Evaluator (Regex)

Validates that output matches a regular expression pattern. Works at both global and case levels.

```yaml
# Global level - applies to all cases
function_name:
  evals:
    PatternName:
      pattern: "regex_pattern"
      case_sensitive: true  # Optional, default: true

# Case level - specific to one case
function_name:
  dataset:
    - case:
        input: "test"
        pattern: "regex_pattern"
        case_sensitive: false  # Optional
```

**Examples:**

```yaml
# Validate email format
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

    - case:
        input: "admin@test.org"
        expected: "admin@test.org"
        pattern: "\\.org$" # Case-specific pattern

# Format validation
format_id:
  evals:
    CorrectFormat:
      pattern: "^id: \\d+$"
      case_sensitive: true
  dataset:
    - case:
        input: 123
        expected: "id: 123"

    - case:
        input: 456
        expected: "ID: 456"
        pattern: "^ID: \\d+$" # Different pattern for this case

# Case insensitive matching
normalize_text:
  dataset:
    - case:
        input: "Hello"
        expected: "HELLO WORLD"
        pattern: "hello"
        case_sensitive: false # Matches "HELLO" too

    - case:
        input: "test"
        expected: "TEST123"
        pattern: "^[A-Z]+\\d+$" # Only uppercase letters + digits

# Multiple patterns
phone_format:
  evals:
    HasDigits:
      pattern: "\\d+"
  dataset:
    - case:
        input: "1234567890"
        expected: "+1 (123) 456-7890"
        pattern: "^\\+\\d+ \\(\\d{3}\\) \\d{3}-\\d{4}$"

    - case:
        input: "9876543210"
        expected: "987-654-3210"
        pattern: "^\\d{3}-\\d{3}-\\d{4}$"
```

**Common Regex Patterns:**

```yaml
# Numbers only
pattern: "^\\d+$"

# Uppercase letters only
pattern: "^[A-Z]+$"

# Email format
pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"

# URL format
pattern: "^https?://.*"

# Contains specific word
pattern: "\\bword\\b"

# Starts with prefix
pattern: "^prefix"

# Ends with suffix
pattern: "suffix$"
```

### 8. LLM Judge Evaluator

Uses a Language Model to evaluate outputs based on a custom rubric. Ideal for semantic evaluation, quality assessment, and cases where rule-based checking is insufficient.

```yaml
function_name:
  evals:
    JudgeName:
      rubric: "Evaluation criteria/question for the LLM"
      include: # Optional, what context to provide to LLM
        - input # Include the input
        - expected_output # Include expected output
      config: # Model configuration
        model: "provider:model_name" # Required (or set JUDGE_MODEL env var)
        temperature: 0.7 # Optional
        max_tokens: 2096 # Optional
        # ... other optional model parameters
        # parameters will be passed into
        # pydantic_ai.settings.ModelSettings ctor
```

**Configuration:**

- `rubric`: **Required** - The evaluation criteria or question for the LLM
- `include`: **Optional** - List of context variables. Valid options:
  - `input`: Include function input
  - `expected_output`: Include expected output
  - **Note:** Output is always included automatically
- `config`: Model configuration
  - `model`: **Required** (unless `JUDGE_MODEL` environment variable is set)
  - All other parameters are optional (temperature, max_tokens, top_p, etc.)

**Examples:**

```yaml
# Basic semantic equivalence check
translate_to_english:
  evals:
    SemanticMatch:
      rubric: "Does the output have the same meaning as the expected output?"
      include:
        - expected_output
      config:
        model: "groq:qwen/qwen-2.5-72b-instruct"
        temperature: 0.0
  dataset:
    - case:
        input: "Bonjour"
        expected: "Hello"

    - case:
        input: "Comment allez-vous?"
        expected: "How are you?"

# Grammar and style checking
generate_response:
  evals:
    IsGrammaticallyCorrect:
      rubric: "Is the output grammatically correct and well-formatted?"
      config:
        model: "groq:qwen/qwen-2.5-72b-instruct"
        temperature: 0.0
        max_tokens: 512
  dataset:
    - case:
        input: "Write a greeting"
        expected: "Hello! How can I help you today?"

# Quality assessment with input context
summarize_text:
  evals:
    IsGoodSummary:
      rubric: "Is the output a good summary of the input text? Does it capture the main points?"
      include:
        - input
      config:
        model: "openai:gpt-4o-mini"
        temperature: 0.1
  dataset:
    - case:
        input: "Long article text here..."
        expected: "Brief summary..."

# Using environment variable for model
# Set: export JUDGE_MODEL="groq:qwen/qwen-2.5-72b-instruct"
check_correctness:
  evals:
    AnswerQuality:
      rubric: "Does the output correctly answer the question based on the input?"
      include:
        - input
        - expected_output
      config:
        temperature: 0.0 # model comes from JUDGE_MODEL env var
  dataset:
    - case:
        input: "What is 2+2?"
        expected: "4"

# Multiple criteria with different judges
write_code:
  evals:
    Correctness:
      rubric: "Is the code functionally correct?"
      include:
        - input
      config:
        model: "openai:gpt-4o"
        temperature: 0.0

    Readability:
      rubric: "Is the code well-structured and readable?"
      config:
        model: "openai:gpt-4o"
        temperature: 0.0
  dataset:
    - case:
        input: "Write a function to reverse a string"
        expected: "def reverse(s): return s[::-1]"
```

**Supported Models:**

- **OpenAI**: `openai:gpt-4o`, `openai:gpt-4o-mini`, `openai:gpt-4-turbo`
- **Groq**: `groq:llama-3.3-70b-versatile`, `groq:qwen/qwen-2.5-72b-instruct`
- **Anthropic**: `anthropic:claude-3-5-sonnet-20241022`
- Any model supported by [pydantic-ai](https://ai.pydantic.dev/)

**Tips:**

- Use `temperature: 0.0` for consistent evaluation
- Be specific in your rubric - clear criteria = better evaluation
- Use `include: [input, expected_output]` when the LLM needs full context
- Set `JUDGE_MODEL` environment variable to avoid repeating model config
- Combine with other evaluators for comprehensive testing

## 🎨 Advanced Examples

### Using Multiple Evaluators

```yaml
process_numbers:
  evals:
    # Global evaluators
    IsList:
      type: list
    NotEmpty:
      assertion: "len(output) > 0"
    AllPositive:
      assertion: "all(x > 0 for x in output)"
    Performance:
      duration: 0.01

  dataset:
    - case:
        input: [1, -2, 3, -4, 5]
        expected: [1, 3, 5] # Case-specific expected
        duration: 50 # Case-specific duration (ms)
```

### Complex Test Scenario

```yaml
json.loads:
  evals:
    IsDict:
      type: dict
    HasKey:
      assertion: "'name' in output"

  dataset:
    - case:
        input: '{"name": "John", "age": 30}'
        expected: { "name": "John", "age": 30 }

    - case:
        input: '{"name": "Jane"}'
        contains: "Jane" # converted to string with as_strings

str.split:
  evals:
    IsList:
      type: list

  dataset:
    - case:
        inputs: ["hello,world", ","]
        expected: ["hello", "world"]

    - case:
        inputs: ["a-b-c", "-"]
        expected: ["a", "b", "c"]
        assertion: "len(output) == 3" # Case-specific assertion
```

## 💡 Tips

### 1. Global vs Case Evaluators

```yaml
# Global: Applies to all cases
evals:
  TypeCheck:
    type: int

# Case-specific: Applies only to that case
dataset:
  - case:
      input: 5
      expected: 25
      duration: 100
```

### 2. Testing Without Expected

```yaml
# Test only with global evaluators
validate_format:
  evals:
    IsString:
      type: str
    HasPrefix:
      assertion: "output.startswith('prefix_')"

  dataset:
    - case:
        input: "test"
        # no expected, only the above checks are performed
```

### 3. Using Input in Assertions

```yaml
double:
  evals:
    IsDouble:
      assertion: "output == input * 2"

  dataset:
    - case:
        input: 5
        # even without expected: 10, assertion checks it
```

### 4. Testing Multiple Functions

```yaml
# test.yml
len:
  dataset:
    - case:
        input: [1, 2, 3]
        expected: 3

json.dumps:
  dataset:
    - case:
        input: { "key": "value" }
        expected: '{"key": "value"}'

make_list:
  dataset:
    - case:
        input: 5
        expected: [5]
```

```bash
# Test all
vowel test.yml

# Test only one
vowel test.yml -f len

# Test multiple (comma-separated)
vowel test.yml -f len,json.dumps

# Test multiple (separate flags)
vowel test.yml -f len -f json.dumps -f make_list
```

## 📚 Documentation

- **[RUNEVALS_GUIDE.md](https://github.com/fswair/vowel/blob/main/docs/RUNEVALS_GUIDE.md)** - RunEvals fluent API
- **[ASSERTION_CONTEXT.md](https://github.com/fswair/vowel/blob/main/docs/ASSERTION_CONTEXT.md)** - Assertion variables
- **[EXAMPLES/](https://github.com/fswair/vowel/blob/main/examples/)** - Working examples here

## 📄 License

Apache 2.0

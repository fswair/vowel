# VOWEL

**YAML-based evaluation framework for testing Python functions with AI-powered test generation and function healing.**

vowel makes it easy to define test cases in YAML and run them against your Python functions. It also provides AI-powered generators that can automatically create test specs, generate implementations, and fix buggy functions.

## Installation

```bash
pip install vowel

# Or with uv
uv add vowel
```

### Development

```bash
git clone https://github.com/fswair/vowel.git
cd vowel
pip install -e .
```

---

## Quick Start

### 1. Create a YAML spec

```yaml
# evals.yml
add:
  dataset:
    - case:
        inputs: { x: 1, y: 2 }
        expected: 3
    - case:
        inputs: { x: -5, y: 5 }
        expected: 0

divide:
  evals:
    Type:
      type: "float"
  dataset:
    - case:
        inputs: { a: 10, b: 2 }
        expected: 5.0
    - case:
        inputs: { a: 1, b: 0 }
        raises: ZeroDivisionError
```

### 2. Run from CLI

```bash
vowel evals.yml
```

### 3. Or programmatically

```python
from vowel import run_evals

def add(x: int, y: int) -> int:
    return x + y

def divide(a: float, b: float) -> float:
    return a / b

summary = run_evals("evals.yml", functions={"add": add, "divide": divide})
print(f"All passed: {summary.all_passed}")
print(f"Coverage: {summary.coverage * 100:.1f}%")
```

### 4. Or use the fluent API

```python
from vowel import RunEvals

summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"add": add, "divide": divide})
    .filter(["add"])
    .debug()
    .run()
)

summary.print()
```

---

## Features

### Evaluators

8 built-in evaluators for flexible testing:

| Evaluator | Purpose |
|-----------|---------|
| **Expected** | Exact value matching |
| **Type** | Return type checking (strict/lenient) |
| **Assertion** | Custom Python expressions (`output > 0`, `output == input * 2`) |
| **Duration** | Performance constraints (function-level & case-level) |
| **Pattern** | Regex validation on output |
| **ContainsInput** | Verify output contains the input |
| **Raises** | Exception class + optional message matching |
| **LLMJudge** | AI-powered rubric evaluation |

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
```

> **Full reference:** [docs/EVALUATORS.md](https://github.com/fswair/vowel/blob/main/docs/EVALUATORS.md)

### Fixtures (Dependency Injection)

Inject databases, temp files, caches into functions under test. Three patterns: generator (yield), tuple (setup/teardown), simple (setup only).

```yaml
fixtures:
  db:
    setup: myapp.setup_db
    teardown: myapp.close_db
    scope: module

query_user:
  fixture: [db]
  dataset:
    - case:
        inputs: { user_id: 1 }
        expected: { name: "Alice" }
```

```python
def query_user(user_id: int, *, db: dict) -> dict | None:
    return db["users"].get(user_id)
```

> **Full reference:** [docs/FIXTURES.md](https://github.com/fswair/vowel/blob/main/docs/FIXTURES.md)

### Input Serializers

Transform YAML inputs into Pydantic models, dates, or custom types:

```python
summary = (
    RunEvals.from_file("evals.yml")
    .with_functions({"get_user": get_user})
    .with_serializer({"get_user": User})      # Schema mode
    .run()
)
```

> **Full reference:** [docs/SERIALIZERS.md](https://github.com/fswair/vowel/blob/main/docs/SERIALIZERS.md)

### AI-Powered Generation

#### EvalGenerator — test existing functions

```python
from vowel import EvalGenerator, Function

generator = EvalGenerator(model="openai:gpt-4o", load_env=True)
func = Function.from_callable(my_function)

result = generator.generate_and_run(func, auto_retry=True, heal_function=True)
print(f"Coverage: {result.summary.coverage * 100:.1f}%")
```

#### TDDGenerator — generate everything from a description

```python
from vowel.tdd import TDDGenerator

generator = TDDGenerator(model="gemini-3-flash-preview", load_env=True)

result = generator.generate_all(
    description="Binary search for target in sorted list. Returns index or -1.",
    name="binary_search"
)

result.print()  # Shows: signature → tests → code → results
```

Step-by-step control:

```python
signature = generator.generate_signature(description="...", name="factorial")
runner, yaml_spec = generator.generate_evals_from_signature(signature)
func = generator.generate_implementation(signature, yaml_spec)
summary = runner.with_functions({"factorial": func.impl}).run()
```

> **Full reference:** [docs/AI_GENERATION.md](https://github.com/fswair/vowel/blob/main/docs/AI_GENERATION.md)

### MCP Server

Expose vowel's capabilities to AI assistants like Claude Desktop via Model Context Protocol.

> **Setup guide:** [docs/MCP.md](https://github.com/fswair/vowel/blob/main/docs/MCP.md)

---

## CLI

```bash
vowel evals.yml                          # Run single file
vowel -d ./tests                         # Run directory
vowel evals.yml -f add,divide            # Filter functions
vowel evals.yml --ci --coverage 90       # CI mode
vowel evals.yml --watch                  # Watch mode
vowel evals.yml --dry-run                # Show plan without running
vowel evals.yml --export-json out.json   # Export results
```

> **Full reference:** [docs/CLI.md](https://github.com/fswair/vowel/blob/main/docs/CLI.md)

---

## EvalSummary

```python
summary = run_evals("evals.yml", functions={...})

summary.all_passed       # bool
summary.success_count    # int
summary.failed_count     # int
summary.total_count      # int
summary.coverage         # float (0.0-1.0)
summary.failed_results   # list[EvalResult]

summary.meets_coverage(0.9)    # Check threshold
summary.print()                # Rich formatted output
summary.to_json()              # Export as dict
summary.xml()                  # Export as XML
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [YAML Spec](https://github.com/fswair/vowel/blob/main/docs/YAML_SPEC.md) | Complete YAML format reference |
| [Evaluators](https://github.com/fswair/vowel/blob/main/docs/EVALUATORS.md) | All 8 evaluator types |
| [Fixtures](https://github.com/fswair/vowel/blob/main/docs/FIXTURES.md) | Dependency injection guide |
| [Serializers](https://github.com/fswair/vowel/blob/main/docs/SERIALIZERS.md) | Input serializer patterns |
| [AI Generation](https://github.com/fswair/vowel/blob/main/docs/AI_GENERATION.md) | EvalGenerator & TDDGenerator |
| [CLI](https://github.com/fswair/vowel/blob/main/docs/CLI.md) | Command-line reference |
| [MCP Server](https://github.com/fswair/vowel/blob/main/docs/MCP.md) | AI assistant integration |
| [Troubleshooting](https://github.com/fswair/vowel/blob/main/docs/TROUBLESHOOTING.md) | Common errors & solutions |

---

## License

Apache License 2.0
